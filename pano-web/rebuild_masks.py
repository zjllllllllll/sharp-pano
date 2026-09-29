"""Rebuild stage-6/7 masks from saved captures, without model calls.

Publish only a prepared mask revision. Previous RGB/depth/Gaussian artifacts
remain in their old revision and JSON snapshots; stages 7–9 become pending.
"""
import argparse
import base64
from copy import deepcopy
from pathlib import Path
import shutil
import time
import uuid
import numpy as np
from PIL import Image
import novel_pipeline as n
from gpt_green_repair import validate_capture


def rebuild(job):
    job=job.resolve()
    if not job.is_relative_to(n.APP/'runs'):raise ValueError('Expected a local run')
    prior=n.read(job/'job.json')
    if prior.get('busy'):raise ValueError('A job is running')
    old_folder=n.work_folder(job);old_selection=n.read(old_folder/'selection.json')
    current=old_selection.get('current')
    if not current:raise ValueError('A confirmed capture is required')
    if prior.get('mask_update_pending') and current['preview'].get('capture_version')==n.CAPTURE_VERSION:
        return prior['mask_update_pending']
    run_id=uuid.uuid4().hex
    backup=job/'reruns'/('geometry-mask-smooth1-'+time.strftime('%Y%m%d-%H%M%S')+'-'+run_id[:6])
    backup.mkdir(parents=True)
    for name in ('job.json','stage-6.json','stage-7.json','stage-8.json','stage-9.json'):
        if (job/name).exists():shutil.copy2(job/name,backup/name)
    shutil.copy2(old_folder/'selection.json',backup/'selection.json')
    running=deepcopy(prior);running.update(busy=True,running_stage=None,error=None,
        mask_update_run=run_id,message='正在更新几何 Mask 及边缘平滑（不调用模型）')
    n.write_json(job/'job.json',running)
    try:
        stage6=n.prepare(job);folder=job/'novel'/stage6['revision']
        if folder==old_folder:raise ValueError('Mask update must create a new scene revision')
        captures={};reports=[]
        candidates={c['id'] for c in stage6['cameras']}
        previews={**old_selection['previews'],current['camera_id']:current['preview']}
        for cid,old in previews.items():
            if cid not in candidates:continue
            source=old_folder/old['path']
            if any(not (source/f'{face}-render-alpha.png').is_file() for face in n.FACE_NAMES):continue
            # Raw renders are read unchanged; no browser/GPU recapture is needed.
            data_uri=lambda p:'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode('ascii')
            payload=dict(revision=stage6['revision'],source_sha256=old['source']['sha256'],
                capture_version=n.CAPTURE_VERSION,
                faces=[data_uri(source/f'{face}-render-rgb.png') for face in n.FACE_NAMES],
                alpha_faces=[data_uri(source/f'{face}-render-alpha.png') for face in n.FACE_NAMES])
            preview=n.capture(job,cid,payload,folder=folder)
            if preview['raw_sha256']!=old['raw_sha256']:raise ValueError('Raw RGB changed during mask rebuild')
            target=validate_capture(folder,preview)
            old_mask=np.asarray(Image.open(source/'mask.png'))>0
            new_mask=np.asarray(Image.open(target/'mask.png'))>0
            captures[cid]=preview
            face_counts={face:int((np.asarray(Image.open(target/f'{face}-mask.png'))>0).sum()) for face in n.FACE_NAMES}
            reports.append(dict(camera_id=cid,old_capture=old['path'],new_capture=preview['path'],
                old_mask_pixels=round(old['mask_fraction']*1024*512),
                geometry_before_cleanup=preview['before_cleanup_pixels'],
                mask_pixels=round(preview['mask_fraction']*1024*512),mask_fraction=preview['mask_fraction'],
                cleanup_removed_pixels=preview['cleanup_removed_pixels'],face_mask_pixels=face_counts))
            reports[-1].update(added_vs_previous=int((new_mask&~old_mask).sum()),
                removed_vs_previous=int((old_mask&~new_mask).sum()))
        cid=current['camera_id']
        if cid not in captures:raise ValueError('Confirmed camera has no reusable raw alpha capture')
        selection=n.read(folder/'selection.json')
        selection.update(selected_ids=old_selection['selected_ids'],completed_ids=[],
            current={**current,'preview':captures[cid]},previews=captures)
        n.write_json(folder/'selection.json',selection)
        state=n.read(job/'job.json')
        if state.get('mask_update_run')!=run_id or state['stages']!=prior['stages']:
            raise ValueError('Job changed during mask rebuild; outputs retained, not published')
        stage6=n.public_state(job,folder)
        pending=dict(revision=stage6['revision'],camera_id=cid,capture=captures[cid],
            backup=str(backup),mask_cleanup=n.MASK_CLEANUP,
            models_called=False,previous_revision=old_folder.name)
        state['stages']={k:v for k,v in state['stages'].items() if int(k)<7};state['stages']['6']=stage6
        state.update(busy=False,running_stage=None,error=None,mask_update_pending=pending,
            message='几何 Mask 已完成边缘平滑；未运行第七至九步模型')
        state.pop('mask_update_run',None)
        n.write_json(job/'stage-6.json',stage6)
        # Backups already exist. Retire stale stage records so standalone tools
        # cannot mistake old generations for the updated mask revision.
        for stage in (7,8,9):
            path=job/f'stage-{stage}.json'
            if path.exists():path.replace(backup/path.name)
        n.write_json(job/'job.json',state)
        report=dict(passed=True,**{k:v for k,v in pending.items() if k!='capture'},captures=reports,
            candidate_count=stage6['count'],raw_rgb_unchanged=True,original_scene=stage6['source'])
        n.write_json(backup/'mask-update.json',report)
        return report
    except Exception:
        state=n.read(job/'job.json')
        if state.get('mask_update_run')==run_id:
            for name in ('stage-6.json','stage-7.json','stage-8.json','stage-9.json'):
                if (backup/name).exists():shutil.copy2(backup/name,job/name)
            n.write_json(job/'job.json',prior)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path)
    import json
    print(json.dumps(rebuild(parser.parse_args().job),ensure_ascii=False,indent=2),flush=True)
