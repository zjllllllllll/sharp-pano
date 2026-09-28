"""Versioned iteration workspaces; public steps 10..13 reuse stages 6..9."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import shutil
import time
import uuid
import numpy as np
from PIL import Image
from novel_pipeline import read,write_json,digest,public_state
from novel_geometry import FACE_NAMES,clean_mask,MASK_CLEANUP,merge_faces,six_faces
from novel_depth_geometry import merge_depth_faces

OBSERVATION_VERSION='accepted-patch-observation-v2-no-old-space-veto'


def trusted_patch_pixels(merged,alpha,patch_alpha,target,hole,extrapolated,support=None):
    return ((True if support is None else support)&hole&~extrapolated&np.isfinite(target)&(target>1e-3)&np.isfinite(merged)&(merged>1e-3)&
        (alpha>=.7)&(patch_alpha>=.7)&(np.abs(merged-target)<=np.maximum(.02,.08*target)))


def compose_observation_depth(rendered,valid,reason,surface):
    """An accepted patch supplies its observed depth, even if old geometry differs."""
    old_known=(reason==0)&np.isfinite(surface)
    depth=np.where(valid,rendered,np.where(old_known,surface,0)).astype(np.float32)
    return depth,old_known|valid,old_known


def ensure_workspace(parent,round_index=2):
    from iteration_stop import MAX_ROUNDS
    if not 2<=round_index<=MAX_ROUNDS:raise ValueError('最多修补 5 轮')
    state=read(parent/'job.json')
    previous=state['stages'].get(str(5+4*(round_index-1)))
    if not previous or previous['acceptance']['status']!='accepted':raise ValueError('上一轮高斯补丁必须通过验收后才能继续')
    evaluation=state.get('iteration_evaluation',{})
    if (evaluation.get('after_round')==round_index-1 and evaluation.get('source_sha256')==previous['sha256']
            and evaluation.get('stop')):raise ValueError('已满足迭代停止规则')
    previous_job=parent.parent/previous.get('execution_job',parent.name)
    existing=state.get('iteration_jobs',{}).get(str(round_index))
    if existing:
        child=parent.parent/existing;child_state=read(child/'job.json')
        if child_state['iteration_parent']!=parent.name:raise ValueError('Iteration owner mismatch')
        if child_state['previous_patch_revision']!=previous['patch_revision']:raise ValueError('Accepted previous round changed')
        return child
    source=previous_job/'novel'/previous['revision']/previous['download']
    if digest(source)!=previous['sha256']:raise ValueError('Previous merged Gaussian changed')
    child=parent.parent/uuid.uuid4().hex;child.mkdir()
    shutil.copy2(source,child/'iteration-base.ply')
    child_state=dict(id=child.name,created=time.time(),iteration_parent=parent.name,iteration_index=round_index,
        camera_selection_policy='max-mask',rgb_repair_model=state.get('rgb_repair_model','gpt-image-2'),
        image_provider=state.get('image_provider'),
        busy=False,running_stage=None,error=None,message=f'准备第 {round_index} 轮累计空间',stages=deepcopy({str(i):state['stages'][str(i)] for i in range(1,6)}),
        iteration_source=dict(file='iteration-base.ply',sha256=previous['sha256'],count=previous['count'],variant='previous-merged'),
        previous_patch_revision=previous['patch_revision'])
    write_json(child/'job.json',child_state)
    state.setdefault('iteration_jobs',{})[str(round_index)]=child.name
    write_json(parent/'job.json',state)
    return child


def accepted_observation(parent,previous,initial,out):
    """Only validated surface rays add knowledge; extrapolated targets add none."""
    from multi_view_volume import RGBDVolume,erp_rays
    state=read(parent/'job.json');folder=parent/'novel'/previous['revision'];patch=folder/previous['path']
    depth=state['stages']['8'];depth_folder=folder/depth['path']
    position=np.asarray(previous['position_world_m'],np.float32)
    faces=[]
    for name in FACE_NAMES:
        merged=np.load(patch/f'merged-{name}-z.npy');alpha=np.load(patch/f'merged-{name}-alpha.npy')
        patch_alpha=np.load(patch/f'patch-{name}-alpha.npy')
        target=np.load(patch/name/'depth-projected.npy')
        extrapolated=np.asarray(Image.open(patch/name/'extrapolated.png'))>0
        hole=np.asarray(Image.open(patch/name/'mask.png'))>0
        support=None
        if previous.get('quality_hash'):
            support=np.asarray(Image.open(patch/name/'support.png'))>0
        valid=trusted_patch_pixels(merged,alpha,patch_alpha,target,hole,extrapolated,support)
        faces.append(np.where(valid,merged,0).astype(np.float32))
    rendered,valid=merge_depth_faces(faces)
    # Do not promote zero/invalid LingBot pixels, even when SHARP extrapolated.
    model=np.load(depth_folder/'panorama-completed.npy')
    valid &= np.isfinite(model)&(model>1e-3)
    rays=erp_rays(1024,512).reshape(-1,3)
    reasons=[];surfaces=[]
    for start in range(0,len(rays),65536):
        _,reason,surface,_,_=initial.query(position,rays[start:start+65536])
        reasons.append(reason);surfaces.append(surface)
    reason=np.concatenate(reasons).reshape(512,1024)
    surface=np.concatenate(surfaces).reshape(512,1024)
    # The patch has already passed its RGB-D/coverage checks. Initial geometry
    # supplies inherited context; it must not veto or overwrite a valid patch.
    depth_new,accepted,old_known=compose_observation_depth(rendered,valid,reason,surface)
    newly_accepted=valid&~old_known
    rgb=np.asarray(Image.open(patch/'merged-panorama.png').convert('RGB'))
    np.save(out/'accepted-depth.npy',depth_new)
    np.save(out/'accepted-valid.npy',accepted)
    Image.fromarray(newly_accepted.astype(np.uint8)*255).save(out/'newly-trusted.png')
    record=dict(version=OBSERVATION_VERSION,camera_id=previous['camera_id'],position_world_m=position.tolist(),
        depth_sha256=digest(out/'accepted-depth.npy'),valid_sha256=digest(out/'accepted-valid.npy'),
        inherited_pixels=int((old_known&~valid).sum()),newly_trusted_pixels=int(newly_accepted.sum()),
        accepted_patch_pixels=int(valid.sum()),updated_known_pixels=int((old_known&valid).sum()),
        rule='accepted merged render, alpha>=0.70, patch coverage, depth agreement; exclude extrapolated/invalid rays; no old-space conflict veto',
        source_patch_sha256=previous['patch_sha256'],source_depth_sha256=depth['files']['panorama-completed.npy'])
    write_json(out/'accepted-observation.json',record)
    return RGBDVolume(depth_new,rgb,accepted),record


def prepare_iteration(job):
    from iteration_stop import evaluate
    from multi_view_volume import VERSION
    state=read(job/'job.json');parent=job.parent/state['iteration_parent'];parent_state=read(parent/'job.json')
    round_index=state['iteration_index'];previous=parent_state['stages'][str(5+4*(round_index-1))]
    if previous['patch_revision']!=state['previous_patch_revision']:raise ValueError('Accepted previous round changed')
    source=state['iteration_source']
    if digest(job/source['file'])!=source['sha256']:raise ValueError('Iteration base changed')
    assessment=evaluate(parent,round_index-1)
    if assessment['stop']:raise ValueError('已满足迭代停止规则')
    evaluated=parent/assessment['path']
    fingerprint=dict(version=VERSION,parent=parent.name,previous=previous['sha256'],
        evaluation_revision=assessment['revision'],code=digest(Path(__file__)))
    revision=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()[:16]
    folder=job/'novel'/revision;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'pipeline.json').exists():return public_state(job,folder)
    info=read(evaluated/'sample-cameras.json')
    write_json(folder/'sample-cameras.json',info)
    shutil.copytree(evaluated/'geometry-masks',folder/'geometry-masks',dirs_exist_ok=True)
    original=parent/'novel'/parent_state['stages']['6']['revision']
    for name in ('cameras.png','camera-map-base.png'):shutil.copy2(original/name,folder/name)
    write_json(folder/'selection.json',dict(selected_ids=assessment['selected_ids'],failed_ids=assessment['failed_ids'],
        current=None,completed_ids=[],previews={}))
    write_json(folder/'pipeline.json',dict(fingerprint=fingerprint,revision=revision,source=source,
        mask_source=str(evaluated/'sampling'),selection_policy='max-mask',mask_version=VERSION,render_resolution=[1024,512],
        face_size=512,fov_degrees=100,candidate_count=len(info['cameras']),mask_cleanup=MASK_CLEANUP,created=time.time(),
        iteration_index=round_index,previous_job=previous.get('execution_job',parent.name),
        accepted_observations=assessment['accepted_observations'],stop_evaluation=assessment['path']))
    return public_state(job,folder)
