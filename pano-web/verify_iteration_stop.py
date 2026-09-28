"""Read-only audit of the pixel stop report and multi-round provenance."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from novel_pipeline import read,write_json,digest
from novel_geometry import clean_mask
from iteration_stop import POLICY,decide,remaining_rows


def verify(parent):
    state=read(parent/'job.json');report=state['iteration_evaluation'];count=report['after_round']
    assert report['policy']==POLICY
    folder=parent/report['path'];assert read(folder/'report.json')==report
    inputs=read(folder/'inputs.json')
    for name,sha in inputs['code'].items():assert digest(Path(__file__).with_name(name))==sha,name
    for name,sha in inputs['inputs'].items():
        assert (digest(parent/name) if (parent/name).exists() else None)==sha,name
    selected={'C000'};failed=set();previous=None;observations=['C000']
    for i in range(1,count+1):
        result=state['stages'][str(5+4*i)];execution=parent.parent/result.get('execution_job',parent.name)
        current=read(execution/'job.json');scene=execution/'novel'/result['revision']
        assert result['acceptance']['status']=='accepted'
        assert digest(scene/result['download'])==result['sha256']
        selection=read(scene/'selection.json');selected.update(selection['selected_ids']);selected.add(result['camera_id']);failed.update(selection.get('failed_ids',[]))
        observations.append(result['camera_id'])
        record=report['accepted_observations'][i]
        assert record['camera_id']==result['camera_id'];assert record['source_patch_sha256']==result['patch_sha256']
        for name,key in [('accepted-depth.npy','depth_sha256'),('accepted-valid.npy','valid_sha256')]:
            assert digest(folder/'observations'/str(i)/name)==record[key]
        if i>1:
            assert current['iteration_source']['sha256']==previous['sha256']
            assert digest(execution/'iteration-base.ply')==previous['sha256']
            for offset in range(4):
                public=dict(state['stages'][str(6+4*(i-1)+offset)])
                assert public.pop('execution_job')==execution.name
                local=current['stages'][str(6+offset)]
                if 'iteration_index' not in local:public.pop('iteration_index')
                assert public==local
            manifest=read(scene/'pipeline.json')
            assert [r['camera_id'] for r in manifest['accepted_observations']]==observations[:-1]
        previous=result
    assert report['source_sha256']==previous['sha256']
    assert report['selected_ids']==sorted(selected);assert report['failed_ids']==sorted(failed)
    info=read(folder/'sample-cameras.json');checks=0
    for camera in info['cameras']:
        raw=np.asarray(Image.open(folder/'sampling'/camera['views']['mask']))>0
        geometry=np.asarray(Image.open(folder/'geometry-masks'/f'{camera["id"]}.png'))>0
        np.testing.assert_array_equal(geometry,clean_mask(raw,periodic=True))
        assert camera['mask_stats']['mask_fraction']==float(geometry.mean())
        assert camera['mask_stats']['raw_mask_fraction']==float(raw.mean());checks+=1
    rows=sorted(remaining_rows(info['cameras'],selected|failed),key=lambda row:(-row['pixel_percent'],row['camera']))
    assert rows==report['rows']
    decision=decide([row['pixel_percent'] for row in rows],count)
    for key,value in decision.items():assert report[key]==value,key
    assert state['iteration_status']==('stopped' if decision['stop'] else 'ready')
    assert len(state['stages'])==5+4*count
    assert not state.get('busy') and not state.get('error')
    return dict(passed=True,after_round=count,mask_files_checked=checks,statistics=report['statistics'],
        stop=report['stop'],reason=report['reason'],observations=observations,selected_ids=report['selected_ids'],
        next_round_absent=str(count+1) not in state.get('iteration_jobs',{}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    result=verify(a.job)
    if a.output:write_json(a.output,result)
    print(result)
