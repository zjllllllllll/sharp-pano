"""Audit actual offline outputs, coordinates, rendering and unchanged inference code."""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
from PIL import Image
import laspy

from batch import digest, write_json, ROOT, HERE
from shapely.geometry import Polygon, Point, shape


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate(run, expected_inset=.5):
    inputs = read(run / 'inputs.json')
    catalog = read(run / 'catalog.json')
    assert catalog['rules']['inset_m'] == expected_inset
    current = {p.name for p in (ROOT / 'pano-tests').iterdir()
               if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}}
    assert current == {row['name'] for row in inputs}, 'Input inventory changed'
    fingerprints = read(run / 'source-fingerprints.json')
    changed = [name for name, sha in fingerprints.items() if digest(ROOT / name) != sha]
    assert set(changed) <= {'pano-web/server.py', 'pano-web/static/index.html'}, changed
    rows = []
    for item in inputs:
        job = run / item['stem']
        assert digest(Path(item['source'])) == item['sha256'], item['name']
        assert digest(job / 'input.png') == item['rgb_sha256'], item['name']
        for stage, function in [(2, 'depth_stage'), (3, 'cloud_stage')]:
            marker = read(job / f'stage-{stage}.json')
            assert marker['source_sha256'] == item['sha256']
            assert marker['canonical_stage'] == f'pano-web.worker.{function}'
        depth = np.load(job / 'depth.npy', mmap_mode='r')
        conditioning = np.load(job / 'conditioning.npy', mmap_mode='r')
        assert depth.shape == (item['size'][1], item['size'][0])
        assert conditioning.shape == depth.shape
        assert np.isfinite(conditioning).all() and (conditioning > 0).all()
        stage = read(job / 'stage-3.json')
        count = stage['count']
        assert (job / 'cloud-full.bin').stat().st_size == count * 24
        cloud = np.memmap(job / 'cloud-full.bin', dtype='<f4', mode='r').reshape(-1, 6)
        for start in range(0, count, 500000):
            chunk = cloud[start:start+500000]
            assert np.isfinite(chunk).all()
            assert (chunk[:, 3:] >= 0).all() and (chunk[:, 3:] <= 1).all()
        with laspy.open(job / 'cloud.las') as reader:
            assert reader.header.point_count == count
            for index in np.linspace(0, count-1, 17).astype(int):
                reader.seek(int(index))
                point = reader.read_points(1)
                np.testing.assert_allclose([point.x[0], point.y[0], point.z[0]],
                                           cloud[index, :3], atol=1e-5, rtol=0)
        native_cameras = read(job / 'cameras.json')
        assert len(native_cameras) == 6
        assert all(c['hfov_deg'] == 100 for c in native_cameras)
        for i in range(6):
            assert (job / 'base' / f'{i}.npz').stat().st_size > 0
        outline = read(job / 'outline.json')
        raw = Polygon(outline['polygon_m'])
        sample = read(job / 'sample-cameras.json')
        room, inset = shape(sample['room_geometry']), shape(sample['inset_geometry'])
        assert room.is_valid and not room.is_empty
        assert raw.is_valid or sample['outline_repaired']
        assert sample['spacing_m'] == .5 and sample['inward_offset_m'] == expected_inset
        assert sample['height_world_y_m'] == 0 and sample['render_complete']
        ids, positions, coverages = set(), set(), []
        for camera in sample['cameras']:
            ident = camera['id']
            assert ident not in ids
            ids.add(ident)
            x, y, z = camera['position_world_m']
            assert y == 0 and x/.5 == round(x/.5) and z/.5 == round(z/.5)
            assert (x, z) not in positions
            positions.add((x, z))
            point = Point(x, z)
            assert room.contains(point) and inset.covers(point)
            assert point.distance(room.boundary) >= expected_inset-1e-9
            with Image.open(job / camera['panorama']) as image:
                assert image.size == (1024, 512) and image.mode == 'RGB'
                image.load()
            with Image.open(job / 'panoramas' / f'{ident}-coverage.png') as mask:
                assert mask.size == (1024, 512)
                coverage = (np.asarray(mask) > 0).mean()
            assert abs(coverage - camera['coverage']) < 1e-9
            coverages.append(coverage)
        assert len(ids) == sample['count']
        if not ids:
            assert sample['empty_reason'] in ('offset_erased_room', 'no_grid_node_inside_offset')
        for filename in ['depth.png', 'bev.png', 'bev-full.png', 'overlay.png',
                         'overview.png', 'evidence.png', 'cameras.png']:
            with Image.open(job / filename) as image:
                image.verify()
        row = dict(scene=item['stem'], points=count, cameras=len(ids),
                   outline_valid=raw.is_valid, area_m2=room.area,
                   empty_reason=sample['empty_reason'],
                   depth_valid_fraction=float((np.isfinite(depth) & (depth > 0)).mean()),
                   panorama_coverage_min=min(coverages) if coverages else None)
        rows.append(row)
        print(f'VALIDATED {item["stem"]}: {len(ids)} cameras', flush=True)
        del cloud, depth, conditioning
    assert catalog['total'] == len(inputs) == catalog['outline_ready'] == catalog['render_complete']
    assert catalog['camera_count'] == sum(r['cameras'] for r in rows)
    assert not list(run.glob('*/analysis-error.json'))
    result = dict(passed=True, checked_at=time.time(), scene_count=len(rows), inset_m=expected_inset,
                  camera_count=sum(r['cameras'] for r in rows),
                  zero_camera_scenes=[r['scene'] for r in rows if not r['cameras']],
                  modified_existing_sources=changed, rows=rows,
                  note='Geometric/numerical audit, not a claim that inferred contours are true walls or cameras collision-free.')
    write_json(run / 'validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=HERE/'runs/20260921-all')
    parser.add_argument('--inset',type=float,default=.5)
    args=parser.parse_args()
    result = validate(args.run,args.inset)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, ensure_ascii=False, indent=2))
