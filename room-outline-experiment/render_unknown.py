"""Render saved camera poses with the RGB-D unknown-volume mask renderer."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import time

import numpy as np
from PIL import Image
from batch import digest, write_json
from occlusion import RGBDVolume, Config, VERSION, REASONS

RENDER_SOURCE='Original radial RGB-D reliable surfaces + conservative unknown-volume occlusion; not Gaussians'


def render_job(job, width=1024, height=512, config=None, limit=None):
    job=Path(job);config=config or Config()
    info=json.loads((job/'sample-cameras.json').read_text(encoding='utf-8'))
    fingerprint=dict(depth=digest(job/'depth.npy'),rgb=digest(job/'input.png'),
                     validity=digest(job/'depth-valid.npy') if (job/'depth-valid.npy').exists() else None,
                     implementation=digest(Path(__file__).with_name('occlusion.py')),
                     output_implementation=digest(Path(__file__)),config=asdict(config),size=[width,height])
    if info.get('render_fingerprint') != fingerprint:
        for camera in info['cameras']:
            for key in ['panorama','views','mask_stats','coverage','render_seconds','reused_camera_id']:
                camera.pop(key,None)
    info.update(render_version=VERSION,render_source=RENDER_SOURCE,render_resolution=[width,height],
                render_fingerprint=fingerprint,render_complete=False,
                mask_reason_bits=REASONS,reused_render_count=0)
    write_json(job/'sample-cameras.json',info)
    (job/'panoramas').mkdir(exist_ok=True)
    volume=None
    for index,camera in enumerate(info['cameras'],1):
        if limit is not None and index>limit:break
        if camera.get('views') and all((job/path).is_file() for path in camera['views'].values()) and (job/f'panoramas/{camera["id"]}-depths.npz').exists():
            continue
        if volume is None:
            volume=RGBDVolume.from_job(job,config)
            info['volume']=volume.stats
            print(f'VOLUME {job.name}: {volume.stats}',flush=True)
        result=volume.render(camera['position_world_m'],width,height)
        if camera.get('coincides_with_original') and volume.stats['source_valid_fraction']==1:
            # Regression gate must include lost surfaces, not just occlusion
            # over whatever geometry survived. No source-view special case.
            if np.any(result['mask']):
                raise AssertionError(f'{job.name}: original-view valid footprint lost: {result["stats"]}')
        mapping={'masked':'','raw':'-raw','mask':'-mask','overlay':'-overlay',
                 'reason_color':'-reasons','unknown':'-unknown','uncertain':'-uncertain',
                 'holes':'-holes','coverage':'-coverage','reason':'-reason-bits'}
        views={}
        for kind,suffix in mapping.items():
            relative=f'panoramas/{camera["id"]}{suffix}.png'
            path=job/relative
            partial=path.with_suffix('.png.partial')
            Image.fromarray(result[kind]).save(partial,format='PNG',compress_level=2)
            partial.replace(path)
            views[kind]=relative
        depth_path=job/f'panoramas/{camera["id"]}-depths.npz'
        with depth_path.with_suffix('.partial').open('wb') as stream:
            np.savez_compressed(stream,surface_range_m=result['surface_depth'],unknown_entry_range_m=result['boundary_depth'])
        depth_path.with_suffix('.partial').replace(depth_path)
        camera.update(panorama=views['masked'],views=views,mask_stats=result['stats'],
                      coverage=result['stats']['coverage'],render_seconds=result['stats']['render_seconds'])
        if index%5==0:
            write_json(job/'sample-cameras.json',info)
            print(f'MASK_PROGRESS {job.name}: {index}/{info["count"]} last={camera["render_seconds"]:.2f}s',flush=True)
    info['render_complete']=all(c.get('views') for c in info['cameras'])
    write_json(job/'sample-cameras.json',info)
    print(f'MASK_SCENE {job.name}: {info["count"]} complete={info["render_complete"]}',flush=True)
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('--limit',type=int)
    a=p.parse_args();render_job(a.job,limit=a.limit)
