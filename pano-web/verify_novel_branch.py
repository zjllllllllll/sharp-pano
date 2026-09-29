"""Audit branch isolation, candidate choice and input provenance without billing."""
from pathlib import Path
import argparse
from novel_pipeline import read,digest,write_json
from gpt_green_repair import validate_capture
from novel_geometry import clean_mask,next_camera,MASK_CLEANUP
import numpy as np
from PIL import Image


def verify_candidates(folder,original_folder):
    info=read(folder/'sample-cameras.json');original=read(original_folder/'sample-cameras.json')
    cleanup=info.get('mask_cleanup')
    if not cleanup:
        assert info==original
        return info['cameras']
    assert cleanup==MASK_CLEANUP
    excluded={'cameras','all_cameras','mask_cleanup'}
    assert {k:v for k,v in info.items() if k not in excluded}=={k:v for k,v in original.items() if k not in excluded}
    raw_folder=Path(read(folder/'pipeline.json')['mask_source'])
    changed_stats={'mask_fraction','raw_mask_fraction','mask_cleanup'}
    for group in ('cameras','all_cameras'):
        assert len(info[group])==len(original[group])
        for current,previous in zip(info[group],original[group]):
            assert {k:v for k,v in current.items() if k!='mask_stats'}=={k:v for k,v in previous.items() if k!='mask_stats'}
            assert {k:v for k,v in current['mask_stats'].items() if k not in changed_stats}=={k:v for k,v in previous['mask_stats'].items() if k not in changed_stats}
            raw=np.asarray(Image.open(raw_folder/current['views']['mask']))>0
            mask=np.asarray(Image.open(folder/'geometry-masks'/f'{current["id"]}.png'))>0
            np.testing.assert_array_equal(mask,clean_mask(raw,periodic=True))
            assert current['mask_stats']['raw_mask_fraction']==float(raw.mean())
            assert current['mask_stats']['mask_fraction']==float(mask.mean())
            assert current['mask_stats']['mask_cleanup']==cleanup
    return info['cameras']

def verify(job):
    branch=read(job/'branch.json');state=read(job/'job.json');source=job.parent/branch['source_job']
    original=read(source/'job.json')
    assert digest(source/'job.json')==branch['source_state_sha256'],'Original state changed'
    for i in range(1,6):assert state['stages'][str(i)]==original['stages'][str(i)]
    for item in branch['copied_files']:
        a=source/item['file'];b=job/item['file']
        assert a.stat().st_size==b.stat().st_size==item['bytes']
        assert a.stat().st_ino!=b.stat().st_ino,'Shared writable file'
        assert digest(a)==digest(b),'Copied asset differs'
    folder=job/'novel'/state['stages']['6']['revision']
    original_folder=source/'novel'/original['stages']['6']['revision']
    cameras=verify_candidates(folder,original_folder)
    expected=sorted((c for c in cameras if c['id']!='C000'),key=lambda c:(-c['mask_stats']['mask_fraction'],int(c['id'][1:])))[0]
    selection=read(folder/'selection.json');current=selection['current']
    assert current['camera_id']==expected['id']==state['stages']['6']['highlighted']
    # Once confirmed, recommended names the next unselected candidate.
    recommended,frontier=next_camera(cameras,selection['selected_ids'],'max-mask')
    assert state['stages']['6']['recommended']==recommended
    assert state['stages']['6']['frontier']==frontier
    validate_capture(folder,current['preview'])
    return dict(passed=True,original_job_unchanged=True,first_five_stages_identical=True,
        independently_copied_assets=len(branch['copied_files']),candidate_pool_identical=True,
        candidate_count=len(cameras),selected_global_max=expected['id'],
        geometry_mask_fraction=expected['mask_stats']['mask_fraction'],
        repair_mask_fraction=current['preview']['mask_fraction'],
        capture_verified=True,completed_stages=list(state['stages']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path)
    job=parser.parse_args().job;result=verify(job);write_json(job/'branch-verification.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2))
