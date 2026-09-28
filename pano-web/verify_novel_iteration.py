"""Audit round isolation, cumulative masks and accepted observation provenance."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from novel_pipeline import read,digest,write_json
from novel_geometry import clean_mask,next_camera,FACE_NAMES


def verify(parent):
    state=read(parent/'job.json')
    snapshot=parent/'iteration-2-before.json'
    if snapshot.exists():
        before=read(snapshot)
        for name,sha in before['files'].items():assert digest(parent/name)==sha,name
        for i in range(1,10):assert state['stages'][str(i)]==before['stages'][str(i)]
    child=parent.parent/state['iteration_jobs']['2'];current=read(child/'job.json')
    assert current['iteration_parent']==parent.name
    # Production saves the inherited stages in the child at workspace creation.
    # A legacy manual snapshot is optional; never fabricate historical evidence.
    for i in range(1,6):assert state['stages'][str(i)]==current['stages'][str(i)]
    for i in range(2,10):assert state['stages'][str(i)]==read(parent/f'stage-{i}.json')
    previous_files_checked=0
    for i in (8,9):
        result=state['stages'][str(i)]
        artifacts=parent/'novel'/result['revision']/result['path']
        for name,sha in result['files'].items():
            assert digest(artifacts/name)==sha,name
            previous_files_checked+=1
    for stage in range(10,14):
        if str(stage) not in state['stages']:continue
        published=dict(state['stages'][str(stage)])
        assert published.pop('execution_job')==child.name
        if 'iteration_index' not in current['stages'][str(stage-4)]:published.pop('iteration_index')
        assert published==current['stages'][str(stage-4)]
    source=state['stages']['9'];folder=child/'novel'/current['stages']['6']['revision']
    assert current['previous_patch_revision']==source['patch_revision']
    assert current['iteration_source']['sha256']==source['sha256']
    assert current['iteration_source']['count']==source['count']
    assert digest(child/'iteration-base.ply')==source['sha256']
    old=parent/'novel'/source['revision'];old_manifest=read(old/'pipeline.json');manifest=read(folder/'pipeline.json')
    assert manifest['source']['count']==source['count']
    cameras=read(folder/'sample-cameras.json')['cameras'];old_cameras=read(old/'sample-cameras.json')['cameras']
    assert [c['id'] for c in cameras]==[c['id'] for c in old_cameras]
    decreased=[];removed=0
    for camera,previous in zip(cameras,old_cameras):
        assert {k:v for k,v in camera.items() if k!='mask_stats'}=={k:v for k,v in previous.items() if k!='mask_stats'}
        baseline=np.asarray(Image.open(Path(old_manifest['mask_source'])/camera['views']['mask']))>0
        raw=np.asarray(Image.open(Path(manifest['mask_source'])/camera['views']['mask']))>0
        cleaned=np.asarray(Image.open(folder/'geometry-masks'/f'{camera["id"]}.png'))>0
        assert not np.any(raw&~baseline),'Previously known geometry became unknown'
        np.testing.assert_array_equal(cleaned,clean_mask(raw,periodic=True))
        assert camera['mask_stats']['mask_fraction']==float(cleaned.mean())
        removed+=int((baseline&~raw).sum())
        if np.any(baseline&~raw):decreased.append(camera['id'])
    selection=read(folder/'selection.json');old_selected=read(old/'selection.json')['selected_ids']
    recommendation,_=next_camera(cameras,old_selected,'max-mask')
    if selection['current']:assert selection['current']['camera_id']==recommendation
    record=read(folder/'observations/accepted-observation.json')
    assert record['camera_id']==source['camera_id']
    assert record['source_patch_sha256']==source['patch_sha256']
    assert digest(folder/'observations/accepted-depth.npy')==record['depth_sha256']
    assert digest(folder/'observations/accepted-valid.npy')==record['valid_sha256']
    trusted=np.asarray(Image.open(folder/'observations/newly-trusted.png'))>0
    depth=state['stages']['8'];model=np.load(old/depth['arrays']['completed'])
    assert not np.any(trusted&(~np.isfinite(model)|(model<=1e-3)))
    # Check every saved input used to decide which previous patch rays to accept.
    patch=old/source['path']
    for face in FACE_NAMES:
        names=[f'merged-{face}-z.npy',f'merged-{face}-alpha.npy',f'patch-{face}-alpha.npy',
            f'{face}/depth-projected.npy',f'{face}/extrapolated.png',f'{face}/mask.png']
        for name in names:assert digest(patch/name)==source['files'][name],name
    assert digest(old/depth['arrays']['completed'])==record['source_depth_sha256']
    return dict(passed=True,pre_iteration_snapshot_available=snapshot.exists(),
        first_nine_stages_unchanged=True if snapshot.exists() else None,
        inherited_first_five_stages_unchanged=True,saved_stage_records_match=True,
        previous_artifact_hashes_checked=previous_files_checked,previous_gaussians_unchanged=True,
        iteration_job=child.name,candidates=len(cameras),accepted_cameras=old_selected,
        selected_max_mask_camera=recommendation,geometry_masks_monotone=True,
        newly_known_pixels_over_all_candidates=removed,candidates_with_reduced_unknown=decreased,
        newly_trusted_observation_pixels=int(trusted.sum()),completed_steps=list(state['stages']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);args=p.parse_args()
    report=verify(args.job);write_json(args.job/'iteration-2-verification.json',report)
    import json
    print(json.dumps(report,ensure_ascii=False,indent=2))
