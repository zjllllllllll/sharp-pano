"""Convert this project's XYZRGB clouds, verify all points, then optionally delete PLY."""
from pathlib import Path
import argparse
import hashlib
import gc
import json
import time

import laspy
import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ('depthmaster-results',)
FIELDS = ('x', 'y', 'z', 'red', 'green', 'blue')
CHUNK = 500_000
SCALE_M = 0.000001


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_source(path):
    path = path.resolve(strict=True)
    if path.parent not in [(ROOT / folder).resolve(strict=True) for folder in FOLDERS]:
        raise ValueError(f'Source outside the configured result directories: {path}')
    if path.suffix.lower() != '.ply' or not path.is_file():
        raise ValueError(f'Not a PLY file: {path}')
    return path


def convert_and_verify(source):
    source = validate_source(source)
    target = source.with_suffix('.las')
    partial = source.with_suffix('.las.partial')
    if target.exists() or partial.exists():
        raise FileExistsError(f'Refusing to overwrite {target} or its partial file')
    source_hash = digest(source)
    # Read-only mapping avoids plyfile's slow per-point binary fallback. No
    # mapped arrays escape this function; release them before batch deletion.
    ply = PlyData.read(source, mmap='r')
    if len(ply.elements) != 1 or ply.elements[0].name != 'vertex':
        raise ValueError(f'Not a vertex-only point cloud: {source}')
    vertices = ply['vertex'].data
    if set(vertices.dtype.names) != set(FIELDS):
        raise ValueError(f'Unexpected PLY fields; do not silently discard attributes: {source}')
    if len(vertices) == 0:
        raise ValueError(f'Empty point cloud: {source}')
    for field in ('x', 'y', 'z'):
        values = vertices[field]
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite coordinates: {source}')
        if np.abs(values.astype(np.float64)).max() / SCALE_M > np.iinfo(np.int32).max - 1:
            raise ValueError(f'Coordinates exceed int32 range at requested precision: {source}')
    for field in ('red', 'green', 'blue'):
        if vertices[field].dtype != np.dtype('u1'):
            raise ValueError(f'Expected 8-bit RGB: {source}')
    header = laspy.LasHeader(point_format=2, version='1.2')
    header.scales = np.full(3, SCALE_M)
    header.offsets = np.zeros(3)
    header.system_identifier = 'Local camera XYZRGB'
    header.generating_software = 'sharp-pano PLY to LAS'
    provenance = {'source': source.name, 'source_sha256': source_hash,
                  'units': 'meters', 'axes': 'X right, Y up, Z forward; camera origin',
                  'crs': 'local camera coordinates, no georeferenced CRS assigned',
                  'color_encoding': 'uint16 = original uint8 * 257',
                  'original_ply_comments': list(ply.comments)}
    header.vlrs.append(laspy.VLR(user_id='sharp-pano', record_id=1,
                               description='Local coordinates and provenance',
                               record_data=json.dumps(provenance).encode('utf-8')))
    with laspy.open(partial, mode='w', header=header, do_compress=False) as writer:
        for start in range(0, len(vertices), CHUNK):
            original = vertices[start:start + CHUNK]
            points = laspy.ScaleAwarePointRecord.zeros(len(original), header=header)
            for field in ('x', 'y', 'z'):
                setattr(points, field, original[field].astype(np.float64))
            for field in ('red', 'green', 'blue'):
                setattr(points, field, original[field].astype(np.uint16) * 257)
            points.return_number[:] = 1
            points.number_of_returns[:] = 1
            writer.write_points(points)
    max_error = np.zeros(3)
    cursor = 0
    with laspy.open(partial) as reader:
        if reader.header.point_count != len(vertices):
            raise AssertionError('Point count changed')
        np.testing.assert_array_equal(reader.header.scales, header.scales)
        np.testing.assert_array_equal(reader.header.offsets, header.offsets)
        if reader.header.point_format.id != 2 or str(reader.header.version) != '1.2':
            raise AssertionError('Unexpected LAS format')
        for points in reader.chunk_iterator(CHUNK):
            original = vertices[cursor:cursor + len(points)]
            for axis, field in enumerate(('x', 'y', 'z')):
                difference = np.abs(np.asarray(getattr(points, field)) - original[field].astype(np.float64))
                max_error[axis] = max(max_error[axis], float(difference.max()))
                if max_error[axis] > SCALE_M / 2 + 1e-12:
                    raise AssertionError(f'Coordinate error exceeds rounding tolerance: {max_error}')
            for field in ('red', 'green', 'blue'):
                np.testing.assert_array_equal(getattr(points, field), original[field].astype(np.uint16) * 257)
            cursor += len(points)
    if cursor != len(vertices):
        raise AssertionError('Incomplete LAS point stream')
    if digest(source) != source_hash:
        raise RuntimeError(f'Source changed during conversion: {source}')
    if target.exists():
        raise FileExistsError(target)
    partial.rename(target)
    return {'source': str(source), 'target': str(target), 'points': len(vertices),
            'source_sha256': source_hash, 'las_sha256': digest(target),
            'max_coordinate_error_xyz_m': max_error.tolist(),
            'rgb_exact_after_uint16_mapping': True, 'point_order_preserved': True,
            'source_bytes': source.stat().st_size, 'las_bytes': target.stat().st_size,
            'source_deleted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delete-originals', action='store_true',
                        help='Delete original PLY only after every conversion and verification succeeds')
    args = parser.parse_args()
    sources = sorted(p for folder in FOLDERS for p in (ROOT / folder).iterdir()
                     if p.is_file() and p.suffix.lower() == '.ply')
    if not sources:
        raise FileNotFoundError('No PLY clouds in the configured result directories')
    for source in sources:
        validate_source(source)
        if source.with_suffix('.las').exists() or source.with_suffix('.las.partial').exists():
            raise FileExistsError(f'An output already exists for {source}')
    report_path = Path(__file__).resolve().parent / 'las-conversion-report.json'
    if report_path.exists():
        raise FileExistsError(f'Report already exists: {report_path}')
    protected = [p for folder in FOLDERS for p in (ROOT / folder).iterdir()
                 if p.is_file() and p.suffix.lower() != '.ply']
    protected_before = {str(p): digest(p) for p in protected}
    started = time.perf_counter()
    records = []
    report = {'las_version': '1.2', 'point_format': 2, 'coordinate_scale_m': SCALE_M,
              'rgb_mapping': 'uint16 = original uint8 * 257', 'records': records,
              'delete_originals_requested': args.delete_originals, 'completed': False}

    def save_report():
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    for index, source in enumerate(sources, 1):
        record = convert_and_verify(source)
        records.append(record)
        save_report()
        print(f'VERIFIED {index}/{len(sources)} {record["target"]} points={record["points"]} '
              f'xyz_error_m={record["max_coordinate_error_xyz_m"]}', flush=True)
    # Final guard for the entire batch before any source is removed.
    for record in records:
        if digest(Path(record['source'])) != record['source_sha256']:
            raise RuntimeError('Original PLY changed; keeping all originals')
        if digest(Path(record['target'])) != record['las_sha256']:
            raise RuntimeError('Verified LAS changed; keeping all originals')
    if {str(p): digest(p) for p in protected} != protected_before:
        raise RuntimeError('A depth/preview artifact changed; keeping all originals')
    report['depth_and_preview_files_unchanged'] = True
    save_report()
    if args.delete_originals:
        gc.collect()  # Release any finished read-only mapping on Windows.
        for record in records:
            source = validate_source(Path(record['source']))
            source.unlink()
            record['source_deleted'] = True
            save_report()
            print(f'DELETED_VERIFIED_SOURCE {source}', flush=True)
    report['completed'] = True
    report['elapsed_seconds'] = time.perf_counter() - started
    report['total_points'] = sum(record['points'] for record in records)
    save_report()
    print(f'COMPLETE clouds={len(records)} points={report["total_points"]} '
          f'deleted_originals={args.delete_originals} seconds={report["elapsed_seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
