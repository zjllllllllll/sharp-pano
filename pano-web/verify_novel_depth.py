"""Read-only audit of a completed stage 8, independently checking saved arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from novel_pipeline import read,digest
from novel_geometry import FACE_NAMES,camera,six_faces
from novel_depth_geometry import merge_depth_faces


def accepted_projection(out,capture,result,repair,models,requested,panorama):
    """Rebuild the quality-filtered RGB-D bundle from its recorded face inputs."""
    from quality_fusion import merge_accepted,compose_accepted
    from face_quality import POLICY_HASH,joint
    bundle=result['accepted_bundle']
    assert read(out/'accepted-bundle.json')==bundle
    assert bundle['policy_hash']==result['quality_hash']==POLICY_HASH
    rgb_quality={q['face']:q for q in repair['quality_faces']}
    depth_quality={q['face']:q for q in result['quality_faces']}
    decisions={q['face']:q for q in bundle['faces']}
    assert bundle['guidance']=='fused_panorama_reprojected'
    assert bundle['depth_fusion']=='valid_weighted_no_cross_face_rejection'
    projected=six_faces(panorama,requested)
    valids=[];rgbs=[]
    for i,name in enumerate(FACE_NAMES):
        decision=joint(rgb_quality[name],depth_quality[name])
        assert decisions[name]==dict(face=name,**decision)
        model_valid=np.asarray(Image.open(out/f'{name}-model-valid.png'))>0
        valid=model_valid&np.isfinite(models[i])&(models[i]>.001)
        accepted=valid if decision['accepted'] else np.zeros_like(valid)
        np.testing.assert_array_equal(np.asarray(Image.open(out/f'{name}-accepted-valid.png'))>0,accepted)
        valids.append(valid if decision['accepted'] else None)
        rgbs.append(projected[i]['rgb'])
    fused=merge_accepted(rgbs,models,valids)
    original=np.asarray(Image.open(capture/'raw.png').convert('RGB'))
    rgb,effective,unresolved=compose_accepted(original,fused['rgb'],requested,fused['support'])
    for name,array in [('accepted-rgb',rgb),('accepted-support',fused['support']),
                       ('effective-mask',effective),('unresolved-mask',unresolved),
                       ('accepted-provenance',fused['provenance']),('depth-conflict',fused['conflict'])]:
        expected=array.astype(np.uint8)*255 if array.dtype==bool else array
        np.testing.assert_array_equal(np.asarray(Image.open(out/f'{name}.png')),expected)
    assert bundle['effective_pixels']==int(effective.sum())
    assert bundle['unresolved_pixels']==int(unresolved.sum())
    return fused['depth'],fused['support']


def verify(job,result=None):
    state=read(job/'job.json');result=result or state['stages']['8'];repair=state['stages']['7']
    folder=job/'novel'/result['revision'];out=folder/result['path']
    manifest=read(folder/'pipeline.json')
    assert result['camera_id']==repair['camera_id']==repair['capture']['camera_id']
    assert result['revision']==repair['revision']
    assert result['source']==manifest['source']==repair['capture']['source']
    assert digest(job/result['source']['file'])==result['source']['sha256']
    assert digest(folder/repair['image'])==repair['image_sha256']
    assert digest(folder/repair['capture']['path']/'mask.png')==repair['capture']['mask_sha256']
    assert all(digest(out/name)==sha for name,sha in result['files'].items())
    cameras=read(folder/'sample-cameras.json')['cameras']
    point=next(c for c in cameras if c['id']==result['camera_id'])
    assert point['position_world_m']==result['position_world_m']
    rgb=np.asarray(Image.open(folder/repair['image']).convert('RGB'))
    original_mask=np.asarray(Image.open(folder/repair['capture']['path']/'mask.png'))>0
    projected=six_faces(rgb,original_mask)
    with np.load(out/'depths.npz',allow_pickle=False) as data:
        raw=data['rendered_radial_m'];final=data['completed_radial_m'];hole=data['depth_hole_mask']
        assert raw.dtype==final.dtype==np.float32 and raw.shape==(512,1024)
        assert hole.dtype==np.bool_
        np.testing.assert_array_equal(data['original_mask'],original_mask)
        np.testing.assert_array_equal(hole,original_mask|~np.isfinite(raw)|(raw<=1e-3))
        np.testing.assert_array_equal(data['position_world_m'],result['position_world_m'])
        direct=result.get('panorama_source')=='lingbot_raw_faces'
        if direct:
            models=[]
            for i,name in enumerate(FACE_NAMES):
                record=next(r for r in result['face_results'] if r['face']==name)
                suffix='rendered' if record.get('skipped') else 'lingbot'
                model=np.load(out/f'{name}-{suffix}-z.npy',allow_pickle=False)
                np.testing.assert_array_equal(data['face_lingbot_z_m'][i],model)
                models.append(model)
            if result.get('accepted_bundle'):
                expected,valid=accepted_projection(out,folder/repair['capture']['path'],result,repair,models,original_mask,rgb)
            else:expected,valid=merge_depth_faces(models)
            np.testing.assert_array_equal(final,expected)
            np.testing.assert_array_equal(data['completed_valid'],valid)
            assert result['panorama_compositing']=='none'
            assert result['invalid_completed_pixels']==int((~valid).sum())
            assert result['invalid_completed_hole_pixels']==int((~valid&hole).sum())
            assert result['outside_changed_pixels']==int(np.count_nonzero(raw[~hole]!=final[~hole]))
            assert np.all(final[~valid]==0)
        else:
            assert raw[~hole].tobytes()==final[~hole].tobytes()
            valid=np.ones_like(hole)
        assert np.all(data['input_radial_m'][hole]==0)
        assert np.isfinite(final).all() and np.all(final[valid]>1e-3)
        for name,key in [('rendered','rendered_radial_m'),('input','input_radial_m'),('completed','completed_radial_m')]:
            np.testing.assert_array_equal(np.load(out/f'panorama-{name}.npy',allow_pickle=False),data[key])
        for i,name in enumerate(FACE_NAMES):
            a=data['face_rendered_z_m'][i];b=data['face_completed_z_m'][i]
            face_hole=np.asarray(Image.open(out/f'{name}-hole.png'))>0
            if result.get('accepted_bundle'):
                capture=folder/repair['capture']['path']
                face_mask=np.asarray(Image.open(capture/f'{name}-mask.png'))>0
                np.testing.assert_array_equal(np.asarray(Image.open(out/f'{name}-rgb.png')),
                                              projected[i]['rgb'])
            else:face_mask=projected[i]['mask']
            np.testing.assert_array_equal(face_hole,face_mask|~np.isfinite(a)|(a<=1e-3))
            expected_input=a.copy();expected_input[face_hole]=0
            np.testing.assert_array_equal(np.load(out/f'{name}-input-z.npy'),expected_input)
            assert a[~face_hole].tobytes()==b[~face_hole].tobytes()
            if result.get('accepted_bundle'):
                model_valid=np.asarray(Image.open(out/f'{name}-model-valid.png'))>0
                usable=model_valid&np.isfinite(models[i])&(models[i]>.001)
                expected_face=a.copy();expected_face[face_hole]=np.where(usable,models[i],0)[face_hole]
                np.testing.assert_array_equal(b,expected_face)
                assert np.isfinite(b).all() and np.all(b>=0)
            else:assert np.isfinite(b).all() and np.all(b>1e-3)
            meta=read(out/f'{name}-camera.json');c2w=np.asarray(meta['c2w'])
            np.testing.assert_allclose(c2w[:3,3],result['position_world_m'])
            np.testing.assert_allclose(c2w[:3,:3],np.asarray(camera(i)['c2w'])[:3,:3])
            np.testing.assert_allclose(np.asarray(meta['w2c'])@c2w,np.eye(4),atol=1e-10)
        return dict(passed=True,camera=result['camera_id'],position_world_m=result['position_world_m'],
            source_sha256_unchanged=True,rgb_and_mask_unchanged=True,artifact_hashes_checked=len(result['files']),
            outside_changed_pixels=int(np.count_nonzero(raw[~hole]!=final[~hole])),
            depth_hole_pixels=int(hole.sum()),panorama_source=result.get('panorama_source','legacy_mask_composite'),
            direct_model_projection_exact=direct,accepted_bundle_verified=bool(result.get('accepted_bundle')),valid_depth_positive_finite=True,invalid_completed_pixels=int((~valid).sum()),
            radial_min_max_m=[float(final[valid].min()),float(final[valid].max())],seconds=result['seconds'],warnings=result['warnings'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path)
    print(json.dumps(verify(parser.parse_args().job),ensure_ascii=False,indent=2))
