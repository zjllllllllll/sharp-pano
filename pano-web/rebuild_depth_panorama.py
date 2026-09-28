"""Reproject saved LingBot returns, preserving the previous stage-8 artifacts.

No rendering, inference, or external service requests. The local server must be
idle; all source hashes and the new projection are checked before publication.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import urlopen

import numpy as np
from PIL import Image

from novel_depth import VERSION,DEPTH_DESCRIPTION,write_depth_outputs,depth_warnings
from novel_geometry import FACE_NAMES
from novel_pipeline import read,digest
from rerun_novel import write
from verify_novel_depth import verify


def rebuild(job,port=8765):
    start=time.perf_counter()
    job=job.resolve();state=read(job/'job.json')
    def require_idle():
        with urlopen(f'http://127.0.0.1:{port}/api/state',timeout=10) as response:
            online=json.load(response)
        if online['gpu_busy'] or state['busy']:raise ValueError('A task is running')
        if read(job/'job.json')!=state:raise ValueError('Job changed; refusing to replace newer results')
    require_idle()
    old=state['stages']['8']
    if old.get('panorama_source')=='lingbot_raw_faces':return verify(job)
    if any(int(stage)>8 for stage in state['stages']):raise ValueError('Later stages depend on the current depth')
    verify(job)
    folder=job/'novel'/old['revision'];previous=folder/old['path']
    signature=dict(version=VERSION,mode='reproject_saved_lingbot_faces',source_depth_revision=old['depth_revision'],
        source_result_sha256=digest(previous/'result.json'),inference_inputs=read(previous/'inputs.json'),
        code_sha256=digest(Path(__file__).with_name('novel_depth.py')),
        geometry_sha256=digest(Path(__file__).with_name('novel_depth_geometry.py')))
    revision=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    relative=f"{state['stages']['7']['capture']['path']}/depth/{revision}"
    out=folder/relative
    if out.exists():raise FileExistsError(f'Rebuild output already exists: {out}')
    shutil.copytree(previous,out)
    write(out/'inputs.json',signature)
    raw=[np.load(out/f'{n}-rendered-z.npy',allow_pickle=False) for n in FACE_NAMES]
    composites=[np.load(out/f'{n}-completed-z.npy',allow_pickle=False) for n in FACE_NAMES]
    models=[]
    for i,name in enumerate(FACE_NAMES):
        record=next(r for r in old['face_results'] if r['face']==name)
        models.append(raw[i].copy() if record.get('skipped') else np.load(out/f'{name}-lingbot-z.npy',allow_pickle=False))
    mask=np.asarray(Image.open(folder/state['stages']['7']['capture']['path']/'mask.png'))>0
    metrics=write_depth_outputs(out,raw,composites,models,mask,old['position_world_m'])
    result=deepcopy(old)
    result.update(metrics)
    result.update(depth_revision=revision,path=relative,image=f'{relative}/panorama-completed.png',
        download=f'{relative}/depths.npz',
        views={name:f'{relative}/panorama-{name}.png' for name in ('rendered','input','completed','hole')},
        arrays={name:f'{relative}/panorama-{name}.npy' for name in ('rendered','input','completed')},
        files={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='result.json'},
        description=DEPTH_DESCRIPTION,warnings=depth_warnings(metrics,old['face_results']),
        rebuilt_from=dict(depth_revision=old['depth_revision'],path=old['path'],inference_reused=True),
        inference_seconds_reused=old.get('seconds'),seconds=time.perf_counter()-start)
    write(out/'result.json',result)
    audit=verify(job,result)
    require_idle()
    backup=job/'reruns'/('depth-direct-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    backup.mkdir(parents=True)
    shutil.copy2(job/'job.json',backup/'job.json')
    shutil.copy2(job/'stage-8.json',backup/'stage-8.json')
    updated=deepcopy(state);updated['stages']['8']=result
    write(job/'stage-8.json',result)
    write(job/'job.json',updated)
    write(job/'direct-panorama-verification.json',dict(audit,backup=str(backup),depth_revision=revision))
    return dict(audit,depth_revision=revision,backup=str(backup))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('job',type=Path)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    print(json.dumps(rebuild(args.job,args.port),ensure_ascii=False,indent=2))
