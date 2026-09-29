"""Append subpixel children; copy one parent's final fields except XYZ.

The network and ALL existing Gaussian parameters remain untouched. Positions
use the point-stage interpolation and, for stage 5, the parent's world offset.
"""
from pathlib import Path
import argparse
import gc
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sharp-sixface'))
from point_densify import BLEND, INDEX, build_densification, cached_result as cached_points
from gaussian_ops import DTYPE, validate_vertices, digest
from gaussian_alignment import align_short_axes, VERSION as ALIGNMENT_VERSION

VERSION = 3
PARTS = ('common', 'removed', 'children')


def stencil(ids, width):
    ids = np.asarray(ids, dtype=np.int64)
    return ids[:, None]//3 + np.array([0, 1, width, width+1]), BLEND[ids % 3]


def inherit_children(original, source, keep, base, ids, cam, width, adjusted):
    """Use the source pixel encoded in id//3, not a nearest spatial neighbor.

    Copy the exported vertex AFTER covariance transforms and edge feathering,
    guaranteeing exact equality of scales, quaternion, SH color and opacity.
    Missing/culled parents have no child. A four-corner Mask hole never fills.
    """
    from point_densify import interpolate_children
    count = len(base['means'])
    ids = np.asarray(ids, dtype=np.int64)
    source = np.asarray(source, dtype=np.int64)
    if keep.shape != (count,) or len(source) != len(original):
        raise ValueError('父高斯源索引 / Mask 长度不一致')
    if np.any(source < 0) or np.any(source >= count) or len(np.unique(source)) != len(source):
        raise ValueError('父高斯源索引越界或重复')
    parent = ids//3
    if (np.any(ids < 0) or np.any(parent >= count) or count % (width*width)
            or np.any(parent % width == width-1) or np.any((parent//width) % width == width-1)):
        raise ValueError('子像素索引越界或跨越网格 / 层边界')
    corners, _ = stencil(ids, width)
    mask_valid = keep[corners].all(axis=1)
    row_for_source = np.full(count, -1, dtype=np.int64)
    row_for_source[source] = np.arange(len(source))
    parent_rows = row_for_source[parent]
    valid = mask_valid & (parent_rows >= 0)
    chosen = np.flatnonzero(valid)
    parent_rows = parent_rows[chosen]
    children = original[parent_rows].copy()
    xyz = interpolate_children(base['means'], base['colors'], ids[chosen], cam, width)[:, :3]
    if adjusted:
        final_parent = np.column_stack([original[k][parent_rows] for k in ('x','y','z')])
        xyz += final_parent-base['means'][parent[chosen]]
    for axis, key in enumerate(('x','y','z')):
        children[key] = xyz[:, axis]
    validate_vertices(children, allow_empty=True)
    stats = dict(candidates=len(ids), added=len(children), masked_parent_cells=int((~mask_valid).sum()),
                 missing_native_parent=int((mask_valid & (row_for_source[parent]<0)).sum()),
                 inherited_fields_bit_exact=True, existing_gaussians_unchanged=True)
    return children, stats


def inputs(job, stage):
    names = ['point-densify.json', INDEX, 'point-mask.json',
             'initialized.ply' if stage == 4 else 'adjusted.ply']
    return {n: [(job/n).stat().st_size, (job/n).stat().st_mtime_ns] for n in names if (job/n).exists()}


def cached_result(job, stage):
    try:
        info = json.loads((job/f'gaussian-dense-{stage}.json').read_text(encoding='utf-8'))
        if (info['version'] != VERSION or not info.get('baseline_variants')
                or (stage == 5 and info.get('normal_alignment') != ALIGNMENT_VERSION)
                or not cached_points(job) or info['inputs'] != inputs(job, stage)):
            return None
        for name, expected in info['files'].items():
            stat = (job/name).stat()
            if [stat.st_size, stat.st_mtime_ns] != expected:
                return None
        return info
    except (OSError, KeyError, ValueError, TypeError):
        return None


class ComparisonWriter:
    """Stream face-sized chunks, avoiding multiple complete scene copies."""
    def __init__(self, job, stage):
        self.job, self.stage = Path(job), stage
        self.stem = 'initialized' if stage == 4 else 'adjusted'
        self.counts = {key: 0 for key in (*PARTS, 'dense', 'baseline', 'masked')}
        self.paths = {key: self.job/f'{self.stem}-dense-{key}.raw.tmp' for key in self.counts}
        self.streams = {key: path.open('wb') for key, path in self.paths.items()}
        self.inheritance = []

    def append(self, original, source, keep, children, stats):
        valid = keep[source]
        buckets = dict(common=original[valid], removed=original[~valid], children=children)
        self.inheritance.append(stats)
        for key, value in buckets.items():
            value.tofile(self.streams[key]); self.counts[key] += len(value)
        for key in ('common', 'children'):
            buckets[key].tofile(self.streams['dense']); self.counts['dense'] += len(buckets[key])
        original.tofile(self.streams['baseline']);self.counts['baseline']+=len(original)
        original[valid].tofile(self.streams['masked']);self.counts['masked']+=int(valid.sum())

    def close(self):
        for stream in self.streams.values():
            stream.close()

    def finish(self):
        import shutil
        self.close()
        files, views = {}, {}
        try:
            for key, raw in self.paths.items():
                name = (f'{self.stem}-dense.ply' if key == 'dense' else
                        f'{self.stem}-compare-{key}.ply' if key in ('baseline','masked') else f'{self.stem}-view-{key}.ply')
                target = self.job/name
                staging = target.with_suffix('.new.ply')
                header = ('ply\nformat binary_little_endian 1.0\ncomment SHARP parent-inherited densification v3\n'
                          f'element vertex {self.counts[key]}\n'+
                          ''.join(f'property float {field}\n' for field in DTYPE.names)+'end_header\n')
                with staging.open('wb') as out, raw.open('rb') as src:
                    out.write(header.encode('ascii')); shutil.copyfileobj(src, out, 1024*1024)
                staging.replace(target)
                files[name] = [target.stat().st_size, target.stat().st_mtime_ns]
                views[key] = dict(preview=name, download=name, count=self.counts[key],
                                  bytes=target.stat().st_size, sha256=digest(target))
        finally:
            for raw in self.paths.values():
                raw.unlink(missing_ok=True)
        return views, files


def build(job, stage, progress=print):
    """Backfill existing stages, pairing dense/base from one forward per face."""
    from PIL import Image
    from geometry import camera, split_face
    from gaussian_ops import ndc_to_camera, world_transform, to_vertices
    from point_mask import load_face
    job = Path(job)
    existing = cached_result(job, stage)
    if existing:
        return existing
    start = time.perf_counter()
    density = build_densification(job, progress)
    stage3 = json.loads((job/'stage-3.json').read_text(encoding='utf-8'))
    height, width = stage3['diagnostics']['initializer_grid']
    face_size = height*width*stage3['layers']
    all_ids = np.fromfile(job/INDEX, dtype='<u4')
    writer = ComparisonWriter(job, stage)
    model = None
    signature = json.dumps(dict(inputs=inputs(job, stage),geometry='panorama_initializer_native_prediction_v1',
                                normal_alignment=ALIGNMENT_VERSION if stage==5 else None),sort_keys=True)
    try:
        for i in range(6):
            progress(f'父高斯继承 STEP {stage} · {i+1}/6')
            cam = camera(i)
            ids = all_ids[(all_ids//(face_size*3)) == i].astype('i8')-i*face_size*3
            keep = load_face(job, i, face_size)
            with np.load(job/'base'/f'{i}.npz') as data:
                base = {k: data[k] for k in data.files}
            if stage == 5:
                cache = job/f'dense-work-{stage}-{i}.npz'
                cached = False
                if cache.exists():
                    with np.load(cache) as data:
                        if str(data['signature']) == signature:
                            vertices,source=data['vertices'],data['source']
                            cached=True
                if not cached:
                    if model is None:
                        import torch
                        from sharp_model import load_model
                        torch.set_num_threads(8); torch.manual_seed(42)
                        model, _ = load_model()
                        rgb = np.asarray(Image.open(job/'input.png').convert('RGB'))
                        radial = np.load(job/'conditioning.npy')
                    face, z, _ = split_face(rgb, radial, cam)
                    from sharp_model import infer_panorama_depth
                    scene, _ = infer_panorama_depth(model, face, z, cam)
                    scene, _ = ndc_to_camera(scene, cam); scene = world_transform(scene, cam)
                    scene, _ = align_short_axes(scene, base, cam, width)
                    vertices, _, source = to_vertices(scene, i, cam['hfov_deg'], return_indices=True)
                    temp = cache.with_suffix('.tmp.npz')
                    np.savez(temp,signature=signature,vertices=vertices,source=source)
                    temp.replace(cache)
                    del scene
            else:
                vertices, _, source = to_vertices(base, i, cam['hfov_deg'], return_indices=True)
            # Both compared outputs use THIS forward and its exact source IDs.
            # Do not join rows to a previous run: tiny GPU differences may move
            # an opacity/edge value across a culling threshold and shift rows.
            children, stats = inherit_children(vertices, source, keep, base, ids, cam, width, stage==5)
            writer.append(vertices, source, keep, children, stats)
            del vertices, children, base
            gc.collect()
        if model is not None:
            del model, rgb, radial
            gc.collect();torch.cuda.empty_cache()
        views, files = writer.finish()
    finally:
        writer.close()
    return publish(job, stage, views, files, density, start, writer.inheritance)


def publish(job, stage, views, files, density, start, inheritance):
    info = dict(version=VERSION, inputs=inputs(job, stage), files=files, dense=views['dense'],
                normal_alignment=ALIGNMENT_VERSION if stage==5 else None,
                parts={k: views[k] for k in PARTS}, point_added=density['added'],
                baseline_variants=dict(unmasked=views['baseline'],masked=views['masked']),
                comparison_baseline='same forward; older original PLYs preserved separately',
                added=views['children']['count'], seconds=time.perf_counter()-start,
                method='append-only single source-pixel parent inheritance; no model hooks or delta interpolation',
                parent='id // 3: encoded cell anchor, same face/layer/pixel; no nearest-neighbor matching',
                position='point-stage child + parent world offset; no final depth reprojection' if stage==5 else 'point-stage child center',
                depth_geometry='panorama_initializer_native_prediction_v1' if stage==5 else 'direct_initialization',
                footprint='unchanged: no scale or opacity redistribution for existing or new Gaussians',
                inherited_fields='all PLY fields except x/y/z; exact copy AFTER edge feathering',
                inheritance=inheritance,
                limitations='No new observations; additive overlap may increase local opacity',
                mask_sha256=density['mask_sha256'])
    path = job/f'gaussian-dense-{stage}.json'
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8'); temp.replace(path)
    return info


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('job', type=Path); p.add_argument('stage', type=int, choices=[4, 5])
    args = p.parse_args(); build(args.job, args.stage)
