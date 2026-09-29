"""Read-only independent pixel/provenance audit (no model calls)."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from novel_pipeline import read,digest,write_json
from novel_geometry import FACE_NAMES,next_camera,outward_blend_weights


def verify(job):
    result=read(job/'stage-7.json');stage6=read(job/'stage-6.json')
    folder=job/'novel'/result['revision'];manifest=read(folder/'pipeline.json')
    info=read(folder/'sample-cameras.json');preview=result['capture'];capture=folder/preview['path']
    original=np.asarray(Image.open(capture/'raw.png').convert('RGB'))
    mask=np.asarray(Image.open(capture/'mask.png'))>0
    final=np.asarray(Image.open(folder/result['image']).convert('RGB'))
    camera=next(c for c in info['cameras'] if c['id']==result['camera_id'])
    mask_source=Path(manifest['mask_source'])/camera['views']['mask']
    # PNG encoders/compression settings can differ; compare actual Mask pixels.
    if preview.get('capture_version'):
        from gpt_green_repair import validate_capture
        validate_capture(folder,preview)
        filename='geometry-mask-raw.png' if preview.get('mask_cleanup') else 'geometry-mask.png'
        assert np.array_equal(np.asarray(Image.open(mask_source)),np.asarray(Image.open(capture/filename)))
    else:assert np.array_equal(np.asarray(Image.open(mask_source)),np.asarray(Image.open(capture/'mask.png')))
    assert digest(capture/'mask.png')==preview['mask_sha256']
    assert digest(job/manifest['source']['file'])==manifest['source']['sha256']
    assert manifest['source']==preview['source']==stage6['source']
    outer=result.get('composite',{}).get('direction')=='outward'
    protected=outward_blend_weights(mask,4,True)==0 if outer else ~mask
    assert np.array_equal(original[protected],final[protected])
    assert digest(folder/result['image'])==result['image_sha256']
    faces=[]
    for name in FACE_NAMES:
        base=np.asarray(Image.open(capture/f'{name}.png').convert('RGB'))
        fm=np.asarray(Image.open(capture/f'{name}-mask.png'))>0
        repaired=np.asarray(Image.open(capture/f'repair/{name}-repaired.png').convert('RGB'))
        protected_face=outward_blend_weights(fm,4)==0 if outer else ~fm
        assert np.array_equal(base[protected_face],repaired[protected_face])
        record=read(capture/f'repair/{name}.json')
        assert digest(capture/f'repair/{name}-repaired.png')==record['sha256']
        faces.append(dict(face=name,mask_pixels=int(fm.sum()),outside_protected_changed_pixels=0,model=record['model'],skipped=record['skipped']))
    selected=read(folder/'selection.json')['selected_ids']
    recommended,frontier=next_camera(info['cameras'],selected,manifest.get('selection_policy','four-neighbor'))
    return dict(passed=True,camera=result['camera_id'],count=len(info['cameras']),source=manifest['source'],
        source_ply_unchanged=True,validated_mask_identical=not bool(preview.get('capture_version')),
        geometry_mask_verified=bool(preview.get('capture_version')),final_outside_protected_changed_pixels=0,
        changed_inside_mask_pixels=int(np.any(original!=final,axis=-1)[mask].sum()),faces=faces,
        selected_ids=selected,next_frontier=frontier,next_recommended_after_future_gaussian_update=recommended,
        seconds=result.get('seconds'),no_gaussian_update=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('--report',type=Path)
    a=p.parse_args();result=verify(a.job)
    if a.report:write_json(a.report,result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2))
