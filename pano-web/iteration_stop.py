"""Evaluate remaining cameras after each merged repair, using geometry pixels."""
from copy import deepcopy
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import numpy as np
from PIL import Image
from novel_pipeline import read,write_json,digest
from novel_geometry import clean_mask,MASK_CLEANUP

MAX_ROUNDS=5
POLICY=dict(version='geometry-pixel-stop-v1',area='cleaned_geometry_pixels',
    strict_max_percent=5.0,relaxed_max_percent=7.0,std_percentage_points=1.5,
    std_ddof=0,max_rounds=MAX_ROUNDS,scope='remaining_unprocessed_candidates')


def decide(percentages,completed_rounds):
    values=np.asarray(percentages,dtype=np.float64)
    if values.ndim!=1 or not np.all(np.isfinite(values)) or np.any((values<0)|(values>100)):
        raise ValueError('Expected finite pixel percentages in [0, 100]')
    stats=dict(count=len(values),max_percent=float(values.max()) if len(values) else None,
        min_percent=float(values.min()) if len(values) else None,
        mean_percent=float(values.mean()) if len(values) else None,
        std_percentage_points=float(values.std(ddof=0)) if len(values) else None)
    if completed_rounds>=MAX_ROUNDS:reason='max_rounds'
    elif not len(values):reason='no_remaining_candidates'
    elif stats['max_percent']<=POLICY['strict_max_percent']:reason='strict_max'
    elif (stats['max_percent']<=POLICY['relaxed_max_percent'] and
          stats['std_percentage_points']<=POLICY['std_percentage_points']):reason='relaxed_max_and_std'
    else:reason='continue'
    return dict(stop=reason!='continue',reason=reason,statistics=stats)


def remaining_rows(cameras,excluded):
    excluded=set(excluded)
    return [dict(camera=c['id'],pixel_percent=float(c['mask_stats']['mask_fraction'])*100,
        raw_pixel_percent=float(c['mask_stats']['raw_mask_fraction'])*100) for c in cameras if c['id'] not in excluded]


def evaluate(parent,after_round=None):
    from novel_iteration import accepted_observation,OBSERVATION_VERSION
    from multi_view_volume import RGBDVolume,PlacedVolume,AccumulatedVolume,VERSION
    state=read(parent/'job.json')
    completed=[i for i in range(1,MAX_ROUNDS+1) if str(5+4*i) in state['stages']]
    after_round=after_round if after_round is not None else max(completed,default=0)
    if after_round<1 or completed[:after_round]!=list(range(1,after_round+1)):
        raise ValueError('Evaluate only completed consecutive repair rounds')
    rounds=[];selected={'C000'};failed=set()
    for i in range(1,after_round+1):
        result=state['stages'][str(5+4*i)]
        if result['acceptance']['status']!='accepted':raise ValueError('Previous patch is not accepted')
        job=parent.parent/result.get('execution_job',parent.name)
        folder=job/'novel'/result['revision'];selection=read(folder/'selection.json')
        selected.update(selection['selected_ids']);selected.add(result['camera_id'])
        failed.update(selection.get('failed_ids',[]))
        if digest(folder/result['download'])!=result['sha256']:raise ValueError('Merged scene changed')
        rounds.append((job,result))
    original=parent/'novel'/state['stages']['6']['revision']
    info=deepcopy(read(original/'sample-cameras.json'))
    fingerprint=dict(policy=POLICY,after_round=after_round,selected_ids=sorted(selected),failed_ids=sorted(failed),
        inputs={name:digest(parent/name) if (parent/name).exists() else None for name in ('input.png','depth.npy','depth-valid.npy')},
        candidate_pool_sha256=digest(original/'sample-cameras.json'),mask_cleanup=MASK_CLEANUP,
        volume_version=VERSION,observation_version=OBSERVATION_VERSION,
        rounds=[dict(job=job.name,sha256=r['sha256'],patch_revision=r['patch_revision'],patch_sha256=r['patch_sha256'],
            depth=read(job/'job.json')['stages']['8']['files']['panorama-completed.npy']) for job,r in rounds],
        code={name:digest(Path(__file__).with_name(name)) for name in ('iteration_stop.py','novel_iteration.py','multi_view_volume.py','novel_geometry.py')})
    revision=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()[:16]
    relative=f'iteration-evaluations/{revision}';out=parent/relative
    if (out/'report.json').exists():return read(out/'report.json')
    out.mkdir(parents=True,exist_ok=True);write_json(out/'inputs.json',fingerprint)
    started=time.perf_counter()
    print(f'第 {after_round} 轮完成：重算累计几何 Mask；停止统计只使用剩余候选的像素占比。',flush=True)
    initial=RGBDVolume.from_job(parent);volumes=[PlacedVolume(initial,[0,0,0])]
    records=[dict(camera_id='C000',position_world_m=[0,0,0])]
    for i,(job,result) in enumerate(rounds,1):
        observed=out/'observations'/str(i);observed.mkdir(parents=True,exist_ok=True)
        added,record=accepted_observation(job,result,initial,observed)
        volumes.append(PlacedVolume(added,record['position_world_m']));records.append(record)
    cumulative=AccumulatedVolume(volumes)
    sampling=out/'sampling';geometry=out/'geometry-masks';geometry.mkdir(exist_ok=True)
    baseline_root=Path(read(original/'pipeline.json')['mask_source']);updated={}
    for index,c in enumerate(info['cameras'],1):
        mask=cumulative.mask(c['position_world_m']);cleaned=clean_mask(mask,periodic=True)
        target=sampling/c['views']['mask'];target.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(mask.astype(np.uint8)*255).save(target)
        Image.fromarray(cleaned.astype(np.uint8)*255).save(geometry/f'{c["id"]}.png')
        baseline=np.asarray(Image.open(baseline_root/c['views']['mask']))>0
        c['mask_stats'].update(raw_mask_fraction=float(mask.mean()),mask_fraction=float(cleaned.mean()),mask_cleanup=MASK_CLEANUP,
            previous_raw_mask_fraction=float(baseline.mean()),newly_known_pixels=int((baseline&~mask).sum()))
        updated[c['id']]=c['mask_stats']
        print(f'{index}/{len(info["cameras"])} {c["id"]}: 几何 Mask 像素占比 {cleaned.mean()*100:.6f}%'+
            ('（已处理，排除停止统计）' if c['id'] in selected else ''),flush=True)
    for c in info.get('all_cameras',[]):
        if c['id'] in updated:c['mask_stats']=deepcopy(updated[c['id']])
    info['mask_cleanup']=MASK_CLEANUP;write_json(out/'sample-cameras.json',info)
    rows=sorted(remaining_rows(info['cameras'],selected|failed),key=lambda row:(-row['pixel_percent'],row['camera']))
    decision=decide([row['pixel_percent'] for row in rows],after_round)
    report=dict(**decision,policy=POLICY,after_round=after_round,source_sha256=rounds[-1][1]['sha256'],
        revision=revision,path=relative,selected_ids=sorted(selected),failed_ids=sorted(failed),rows=rows,
        next_camera=rows[0]['camera'] if rows and not decision['stop'] else None,
        accepted_observations=records,seconds=time.perf_counter()-started)
    with (out/'candidates.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['camera','pixel_percent','raw_pixel_percent']);writer.writeheader();writer.writerows(rows)
    write_json(out/'report.json',report)
    print(json.dumps(dict(after_round=after_round,**decision),ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);args=parser.parse_args()
    result=evaluate(args.job)
    write_json(args.job/'iteration-evaluation.json',result)
