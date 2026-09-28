"""Freeze existing rendered geometric pools without re-running inference.

Back up mutable prior metadata/maps first. Never rewrite linked model arrays or
panorama payloads. Re-running is idempotent, including after an interrupted run.
"""
from pathlib import Path
import argparse
import shutil
import time

from batch import HERE, write_json
from adaptive_sampling import TARGET, read
from cameras import draw_camera_map
from candidate_selection import apply_selection, write_candidates, VERSION
from postprocess import publish, make_report


def finalize(run=TARGET):
    run=Path(run).resolve()
    if run.parent!=(HERE/'runs').resolve():
        raise ValueError('An immediate experiment/runs child is required')
    backup=run/'selection-history/before-mask15-connected'
    backup.mkdir(parents=True,exist_ok=True)
    for name in ['catalog.json','report.md','render-status.json','sampling-validation.json','browser-validation.json','experiment.json']:
        if (run/name).exists() and not (backup/name).exists():shutil.copy2(run/name,backup/name)
    for item in read(run/'inputs.json'):
        job=run/item['stem'];previous=read(job/'sample-cameras.json')
        if previous.get('selection_version')==VERSION and previous.get('selection_complete') and (job/'candidate-cameras.json').exists():
            continue
        saved=backup/item['stem'];saved.mkdir(exist_ok=True)
        for name in ['sample-cameras.json','cameras.png','camera-map-base.png']:
            if (job/name).exists() and not (saved/name).exists():shutil.copy2(job/name,saved/name)
        info=apply_selection(previous)
        if not info['selection_complete']:raise ValueError(f'{job.name}: incomplete Mask statistics; render first')
        draw_camera_map(job,info)
        write_json(job/'sample-cameras.json',info)
        write_candidates(job,info)
        print(f'FIXED {job.name}: {info["geometric_camera_count"]} -> {info["count"]}; {info["selection"]["counts"]}',flush=True)
    rows=publish(run);make_report(run,rows)
    write_json(run/'render-status.json',dict(complete=all(r['render_complete'] for r in rows),errors=[],
        finished=time.time(),selection_version=VERSION,camera_count=sum(len(r['cameras']) for r in rows),
        geometric_camera_count=sum(r['sample']['geometric_camera_count'] for r in rows),
        sampling_errors=[dict(scene=r['id'],error=r['sample']['sampling_error']) for r in rows if r['sample'].get('sampling_error')],
        selection_errors=[dict(scene=r['id'],error=r['sample']['selection_error']) for r in rows if r['sample'].get('selection_error')]))
    config=read(run/'experiment.json')
    config['camera_selection']=dict(version=VERSION,mask_threshold=.15,connectivity=4,spacing_m=.5,
        start_offset_m=.5,offset_step_m=.05,offset_stop_below_m=.1,minimum_eight_neighbors=4)
    config.update(sampling='Adaptive offset 0.5 -> 0.05 m; eight-neighbor stop, then Mask <=15% and origin four-connected component',
        preview='1024x512 RGB-D reliable surfaces with unknown-space first occlusion Mask; not Gaussian renders',camera_revision=VERSION)
    write_json(run/'experiment.json',config)
    print('FIXED_CANDIDATES_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=TARGET)
    finalize(parser.parse_args().run)
