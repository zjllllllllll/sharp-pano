"""Build a separate camera-only revision without rerunning depth or SHARP."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import os
import shutil
import time

from batch import HERE, digest, write_json
from cameras import process, sample_cameras
from postprocess import publish, make_report


def prepare(source, target, inset):
    if target.exists():
        manifest=json.loads((target/'resample.json').read_text(encoding='utf-8'))
        if manifest['inset_m'] != inset or manifest['source_name'] != source.name:
            raise ValueError('Existing output is a different revision')
        return
    if source.resolve().parent != (HERE/'runs').resolve() or target.resolve().parent != (HERE/'runs').resolve():
        raise ValueError('Both run directories must be immediate children of experiment/runs')
    if source.resolve() == target.resolve():
        raise ValueError('New revision must not overwrite the source')
    inputs=json.loads((source/'inputs.json').read_text(encoding='utf-8'))
    snapshot={}
    for row in inputs:
        old=source/row['stem']
        for path in [old/'sample-cameras.json',old/'cameras.png',*(old/'panoramas').glob('*.png')]:
            snapshot[path.relative_to(source).as_posix()]=digest(path)
    target.mkdir()
    write_json(target/'previous-camera-fingerprints.json',snapshot)
    for filename in ['inputs.json','source-fingerprints.json','experiment.json']:
        shutil.copy2(source/filename,target/filename)
    linked=[]
    for row in inputs:
        old,new=source/row['stem'],target/row['stem']
        new.mkdir()
        for original in old.rglob('*'):
            relative=original.relative_to(old)
            if not original.is_file() or relative.parts[0]=='panoramas':
                continue
            if original.name in {'sample-cameras.json','cameras.png','analysis-error.json'} or original.suffix=='.partial':
                continue
            destination=new/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            # Only immutable inference/evidence arrays are shared. Camera files,
            # PNGs and metadata are independent copies and never written through links.
            if original.suffix in {'.bin','.las','.npy','.npz'}:
                os.link(original,destination)
                linked.append(destination.relative_to(target).as_posix())
            else:
                shutil.copy2(original,destination)
    write_json(target/'resample.json',dict(source_name=source.name,inset_m=inset,
        spacing_m=.5,height_world_y_m=0.,created=time.time(),immutable_hardlinks=linked,
        note='Depth, native SHARP point cloud and fitted outlines are reused unchanged; render cache is matched by exact world position.'))
    experiment=json.loads((target/'experiment.json').read_text(encoding='utf-8'))
    experiment.update(sampling=f'Same source height, source-anchored 0.5 m grid, {inset:g} m inward offset',
                      camera_revision=f'inset-{inset:g}',models_rerun=False)
    write_json(target/'experiment.json',experiment)
    # Prepare every scene's sampling metadata before launching renders.
    for row in inputs:
        job=target/row['stem']
        outline=json.loads((job/'outline.json').read_text(encoding='utf-8'))
        info=sample_cameras(outline['polygon_m'],inset=inset)
        info['render_complete']=False
        write_json(job/'sample-cameras.json',info)
    publish(target)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=HERE/'runs/20260921-all')
    parser.add_argument('--target',type=Path,default=HERE/'runs/20260921-inset-0p5')
    parser.add_argument('--inset',type=float,default=.5)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    prepare(args.source,args.target,args.inset)
    rows=json.loads((args.target/'inputs.json').read_text(encoding='utf-8'))
    errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={}
        for row in rows:
            job=args.target/row['stem']
            metadata=job/'sample-cameras.json'
            if metadata.exists() and json.loads(metadata.read_text(encoding='utf-8')).get('render_complete'):
                print('SKIP',row['stem'],flush=True)
                continue
            future=pool.submit(process,job,True,args.inset,args.source/row['stem'])
            futures[future]=row['stem']
        for future in as_completed(futures):
            name=futures[future]
            try:
                info=future.result()
                print(f'SCENE_COMPLETE {name}: {info["count"]} cameras, reused {info["reused_render_count"]}',flush=True)
            except Exception as error:
                errors.append(dict(scene=name,error=repr(error)))
                print(f'SCENE_ERROR {name}: {error!r}',flush=True)
            publish(args.target)
    result=publish(args.target)
    make_report(args.target,result)
    if errors:
        write_json(args.target/'resample-errors.json',errors)
        raise RuntimeError(f'{len(errors)} scenes failed')
    print(f'RESAMPLE_COMPLETE {len(result)} scenes / {sum(len(r["cameras"]) for r in result)} cameras',flush=True)


if __name__=='__main__':
    main()
