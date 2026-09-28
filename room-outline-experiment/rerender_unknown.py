"""Stage a render-only revision, keeping exact existing camera IDs and poses."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import os
import shutil
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

from batch import HERE, digest, write_json
from occlusion import VERSION
from render_unknown import render_job, RENDER_SOURCE
from postprocess import publish, make_report
from cameras import number_cameras, draw_camera_map, NUMBERING


def prepare(source,target):
    if source.resolve().parent != (HERE/'runs').resolve() or target.resolve().parent != (HERE/'runs').resolve():
        raise ValueError('Run paths must be immediate experiment/runs children')
    if source.resolve()==target.resolve():raise ValueError('Do not overwrite source run')
    manifest=target/'rerender.json'
    if target.exists():
        previous=json.loads(manifest.read_text(encoding='utf-8'))
        if previous['source_name']!=source.name or previous['version']!=VERSION:raise ValueError('Different output revision')
        return
    inputs=json.loads((source/'inputs.json').read_text(encoding='utf-8'))
    target.mkdir()
    for filename in ['inputs.json','source-fingerprints.json','experiment.json']:
        shutil.copy2(source/filename,target/filename)
    snapshot={};linked=[]
    for row in inputs:
        old,new=source/row['stem'],target/row['stem'];new.mkdir()
        for path in old.rglob('*'):
            rel=path.relative_to(old)
            if not path.is_file() or rel.parts[0]=='panoramas' or path.name in {'sample-cameras.json','analysis-error.json'} or path.suffix=='.partial':continue
            dst=new/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix in {'.npy','.npz','.bin','.las'}:
                os.link(path,dst);linked.append(dst.relative_to(target).as_posix())
            else:shutil.copy2(path,dst)
        for path in [old/'sample-cameras.json',old/'outline.json',old/'cameras.png',*(old/'panoramas').glob('*.png')]:
            snapshot[path.relative_to(source).as_posix()]=digest(path)
        info=json.loads((old/'sample-cameras.json').read_text(encoding='utf-8'))
        for cam in info['cameras']:
            for key in ['panorama','coverage','render_seconds','reused_camera_id','views','mask_stats']:cam.pop(key,None)
        info=number_cameras(info)
        info.update(render_complete=False,render_source=RENDER_SOURCE,render_version=VERSION,reused_render_count=0)
        info.pop('render_fingerprint',None)
        write_json(new/'sample-cameras.json',info)
        draw_camera_map(new,info)
        print(f'STAGED {row["stem"]}: {info["count"]} cameras, C000 origin',flush=True)
    write_json(target/'previous-render-fingerprints.json',snapshot)
    write_json(manifest,dict(source_name=source.name,version=VERSION,created=time.time(),
        immutable_hardlinks=linked,plan_sha256=digest(HERE.parent/'novel-view-occlusion-mask-plan.md'),
        numbering=NUMBERING,
        note='Preview geometry fixed; camera IDs/map labels renumbered; missing origin references added. Existing positions, outlines, RGB-D, SHARP arrays untouched.'))
    publish(target)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=HERE/'runs/20260921-unknown-v1')
    p.add_argument('--target',type=Path,default=HERE/'runs/20260922-unknown-v2')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--only',nargs='+');p.add_argument('--limit',type=int)
    a=p.parse_args();prepare(a.source,a.target)
    inputs=json.loads((a.target/'inputs.json').read_text(encoding='utf-8'))
    errors=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={}
        for row in inputs:
            if a.only and row['stem'] not in a.only:continue
            future=pool.submit(render_job,a.target/row['stem'],limit=a.limit)
            futures[future]=row['stem']
        for future in as_completed(futures):
            name=futures[future]
            try:future.result()
            except Exception as error:
                import traceback
                traceback.print_exc();errors.append(dict(scene=name,error=repr(error)))
            publish(a.target)
    # A Windows commit-limit failure must not silently leave missing previews.
    # Retry failed scenes once, serially, after all parallel workers have exited.
    if errors and a.limit is None:
        failed=errors;errors=[]
        with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as pool:
            for row in failed:
                name=row['scene'];print(f'SERIAL_RETRY {name}',flush=True)
                try:pool.submit(render_job,a.target/name).result()
                except Exception as error:errors.append(dict(scene=name,error=repr(error)))
                publish(a.target)
    rows=publish(a.target);make_report(a.target,rows)
    write_json(a.target/'render-status.json',dict(complete=not errors and all(r['render_complete'] for r in rows),
        errors=errors,finished=time.time(),camera_count=sum(len(r['cameras']) for r in rows)))
    if errors:raise RuntimeError(errors)
    print('UNKNOWN_RENDER_FINISHED',flush=True)


if __name__=='__main__':main()
