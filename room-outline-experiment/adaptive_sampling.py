"""Publish adaptive camera candidates; reuse immutable previews by exact pose.

No RGB-D/SHARP inference or occlusion algorithm changes. Output provenance stays
per camera because old fully-valid renders use the audited equivalent renderer.
"""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import os
import shutil
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from batch import HERE, digest, write_json
from cameras import sample_cameras, SamplingError, draw_camera_map, SAMPLING_VERSION, MASK_THRESHOLD
from candidate_selection import apply_selection, write_candidates, VERSION as SELECTION_VERSION
from postprocess import publish, make_report
from render_unknown import render_job, RENDER_SOURCE
from occlusion import VERSION

SOURCE=HERE/'runs/20260922-unknown-v2'
TARGET=HERE/'runs/20260922-adaptive-sampling'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def link_once(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():os.link(source,target)
    elif not source.samefile(target):raise ValueError(f'Unexpected existing cache file: {target}')


def prepare(source,target):
    root=(HERE/'runs').resolve()
    if source.resolve().parent!=root or target.resolve().parent!=root or source.resolve()==target.resolve():
        raise ValueError('Distinct immediate experiment/runs children required')
    target.mkdir(exist_ok=True)
    manifest_path=target/'adaptive-sampling.json'
    if manifest_path.exists():
        manifest=read(manifest_path)
        if manifest['source_name']!=source.name or manifest['version']!=SAMPLING_VERSION:
            raise ValueError('Different revision already exists')
    else:
        manifest=dict(source_name=source.name,version=SAMPLING_VERSION,created=time.time(),
            source_catalog_sha256=digest(source/'catalog.json'),scenes={},
            note='Only camera sampling/figures change; cached renders are immutable hardlinks. New poses are rendered with unchanged occlusion code. No AI inpainting.')
        write_json(manifest_path,manifest)
    for name in ['inputs.json','source-fingerprints.json','experiment.json']:
        if not (target/name).exists():shutil.copy2(source/name,target/name)
    # Keep the evidence establishing old/current render equivalence, not pretend
    # that copied images were produced by the current implementation.
    if not (target/'implementation').exists():shutil.copytree(source/'implementation',target/'implementation')
    for item in read(source/'inputs.json'):
        name=item['stem'];old=source/name;job=target/name;job.mkdir(exist_ok=True)
        if (job/'sample-cameras.json').exists():continue
        for path in old.rglob('*'):
            rel=path.relative_to(old)
            if not path.is_file() or rel.parts[0]=='panoramas' or path.name in {'sample-cameras.json','cameras.png','camera-map-base.png','analysis-error.json'} or path.suffix=='.partial':continue
            dst=job/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix in {'.bin','.las','.npy','.npz'}:link_once(path,dst)
            elif not dst.exists():shutil.copy2(path,dst)
        previous=read(old/'sample-cameras.json')
        try:info=sample_cameras(read(job/'outline.json')['polygon_m'])
        except SamplingError as error:info=error.info
        prior={tuple(c['position_world_m']):c for c in previous['cameras']}
        reused=0
        for camera in info['cameras']:
            cached=prior.get(tuple(camera['position_world_m']))
            if cached is None:continue
            views={}
            for kind,relative in cached['views'].items():
                suffix=Path(relative).name[len(cached['id']):]
                destination=f'panoramas/{camera["id"]}{suffix}'
                link_once(old/relative,job/destination);views[kind]=destination
            link_once(old/f'panoramas/{cached["id"]}-depths.npz',job/f'panoramas/{camera["id"]}-depths.npz')
            camera.update(views=views,panorama=views['masked'],mask_stats=deepcopy(cached['mask_stats']),
                coverage=cached['coverage'],render_seconds=cached['render_seconds'],
                render_fingerprint=deepcopy(previous['render_fingerprint']),reused_camera_id=cached['id'])
            reused+=1
        for key in ['render_fingerprint','volume','mask_reason_bits','panorama_projection']:
            if key in previous:info[key]=deepcopy(previous[key])
        info.update(render_complete=all(c.get('views') for c in info['cameras']),render_version=VERSION,
            render_source=RENDER_SOURCE,render_resolution=[1024,512],reused_render_count=reused,
            provenance='Per-camera render_fingerprint is authoritative; equivalent old renders are preserved by pose.')
        manifest['scenes'][name]=dict(source_metadata_sha256=digest(old/'sample-cameras.json'),
            source_map_sha256=digest(old/'cameras.png'),source_outline_sha256=digest(old/'outline.json'),
            count=info['count'],reused=reused,new=info['count']-reused)
        write_json(manifest_path,manifest)
        write_json(job/'sample-cameras.json',info)
        print(f'SAMPLED {name}: offset={info["inward_offset_m"]} neighbors={info["neighbor_count"]}/8 count={info["count"]} new={info["count"]-reused}',flush=True)
    publish(target)


def finish_scene(job):
    info=read(job/'sample-cameras.json')
    info['cameras']=info.get('all_cameras',info['cameras'])
    missing=[deepcopy(c) for c in info['cameras'] if not c.get('views')]
    if missing:
        stage=job/'new-pose-render';stage.mkdir(exist_ok=True)
        for name in ['input.png','depth.npy','depth-valid.npy']:
            if (job/name).exists():link_once(job/name,stage/name)
        subset_path=stage/'sample-cameras.json'
        if not subset_path.exists():
            subset=deepcopy(info);subset.update(cameras=missing,count=len(missing),render_complete=False)
            subset.pop('render_fingerprint',None);write_json(subset_path,subset)
        result=render_job(stage)
        by_id={c['id']:c for c in result['cameras']}
        for camera in info['cameras']:
            if camera['id'] not in by_id:continue
            rendered=by_id[camera['id']]
            for relative in rendered['views'].values():link_once(stage/relative,job/relative)
            relative=f'panoramas/{camera["id"]}-depths.npz';link_once(stage/relative,job/relative)
            camera.update(rendered,render_fingerprint=result['render_fingerprint'])
        info['new_render_fingerprint']=result['render_fingerprint']
    info['render_complete']=all(c.get('views') for c in info['cameras'])
    info=apply_selection(info)
    draw_camera_map(job,info)
    write_json(job/'sample-cameras.json',info)
    write_candidates(job,info)
    print(f'ADAPTIVE_COMPLETE {job.name}: {info["count"]} candidates / {len(info["default_retained_camera_ids"])} retained at {MASK_THRESHOLD:.0%}',flush=True)
    return info


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--target',type=Path,default=TARGET);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    prepare(a.source,a.target)
    if a.prepare_only:return
    errors=[]
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        futures={}
        for item in read(a.target/'inputs.json'):
            job=a.target/item['stem'];info=read(job/'sample-cameras.json')
            if info['render_complete'] and info.get('camera_map_projection') and info.get('selection_version')==SELECTION_VERSION and (job/'candidate-cameras.json').exists():continue
            futures[pool.submit(finish_scene,job)]=item['stem']
        for future in as_completed(futures):
            try:future.result()
            except Exception as error:
                import traceback
                traceback.print_exc();errors.append(dict(scene=futures[future],error=str(error)))
            publish(a.target)
    rows=publish(a.target);make_report(a.target,rows)
    write_json(a.target/'render-status.json',dict(complete=not errors and all(r['render_complete'] for r in rows),
        errors=errors,sampling_errors=[dict(scene=r['id'],error=r['sample']['sampling_error']) for r in rows if r['sample'].get('sampling_error')],
        selection_errors=[dict(scene=r['id'],error=r['sample']['selection_error']) for r in rows if r['sample'].get('selection_error')],
        selection_version=SELECTION_VERSION,geometric_camera_count=sum(r['sample'].get('geometric_camera_count',len(r['cameras'])) for r in rows),
        finished=time.time(),camera_count=sum(len(r['cameras']) for r in rows)))
    if errors:raise RuntimeError(errors)
    print('ADAPTIVE_ALL_COMPLETE',flush=True)


if __name__=='__main__':main()
