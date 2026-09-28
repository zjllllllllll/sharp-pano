"""Prepare a preserved-copy stage-7 rerun, then use the live server's worker.

Administrative CLI: only run when a rerun has been explicitly requested.
Uses stdlib only; all model execution stays in the server's verified launchers.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from urllib.request import Request,urlopen
import uuid


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def api(base,path,token=None):
    headers={'Content-Type':'application/json'}
    if token:headers['X-Local-Token']=token
    request=Request(base+path,data=b'{}' if token else None,headers=headers)
    with urlopen(request,timeout=30) as response:
        return json.load(response)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--port',type=int,default=8770)
    args=parser.parse_args()
    if not re.fullmatch('[a-f0-9]{32}',args.job):raise ValueError('Invalid job')
    base=f'http://127.0.0.1:{args.port}'
    job=Path(__file__).resolve().parent/'runs'/args.job
    online=api(base,'/api/state')
    state=read(job/'job.json')
    if online['gpu_busy'] or state['busy']:raise ValueError('A task is already running')
    if api(base,f'/api/jobs/{args.job}')!=state:raise ValueError('Server and disk state disagree')
    if '7' not in state['stages']:raise ValueError('No completed RGB repair to rerun')
    published=state['stages']['7'];preview=published['capture']
    revision=published['revision']
    if not re.fullmatch('[a-f0-9]{16}',revision):raise ValueError('Invalid revision')
    folder=job/'novel'/revision
    old=(folder/preview['path']).resolve()
    if not old.is_relative_to(folder.resolve()):raise ValueError('Invalid capture path')
    selection_path=folder/'selection.json';selection=read(selection_path)
    if selection.get('current',{}).get('camera_id')!=published['camera_id']:
        raise ValueError('Confirmed camera changed; refusing to replace another camera')
    if sha(old/'raw.png')!=preview['raw_sha256'] or sha(old/'mask.png')!=preview['mask_sha256']:
        raise ValueError('Original capture hashes changed')
    original_state_hash=sha(job/'job.json');original_selection_hash=sha(selection_path)
    backup=job/'reruns'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    backup.mkdir(parents=True)
    for name in ('job.json','stage-7.json','stage-8.json','stage-9.json','stage-7.log','stage-8.log','stage-9.log'):
        if (job/name).is_file():shutil.copy2(job/name,backup/name)
    shutil.copy2(selection_path,backup/'selection.json')
    # Fresh URLs avoid stale browser images; all old RGB and depth files remain.
    new=old.parent/uuid.uuid4().hex[:16];new.mkdir()
    names=['raw.png','mask.png','masked.png']
    names += [f'{face}{suffix}.png' for face in preview['faces'] for suffix in ('','-mask','-masked')]
    for name in names:shutil.copy2(old/name,new/name)
    replacement=deepcopy(preview)
    replacement.update(path=new.relative_to(folder).as_posix(),rerun_of=preview['path'],rerun_created=time.time())
    write(new/'capture.json',replacement)
    rerun=dict(job=args.job,camera_id=published['camera_id'],old_capture=preview['path'],
        new_capture=replacement['path'],backup=str(backup),created=time.time())
    write(backup/'rerun.json',rerun)
    if (sha(job/'job.json')!=original_state_hash or sha(selection_path)!=original_selection_hash or
            api(base,'/api/state')['gpu_busy']):
        raise ValueError('State changed during preparation; live job was left untouched')
    selection['current']['preview']=replacement
    selection['previews'][published['camera_id']]=replacement
    selection['completed_ids']=[cid for cid in selection['completed_ids'] if cid!=published['camera_id']]
    state['stages'].pop('7',None);state['stages'].pop('8',None);state['stages'].pop('9',None)
    state.update(busy=False,running_stage=None,error=None,message='旧结果已保留，准备使用单图 1024 输入重新补洞')
    write(selection_path,selection);write(job/'job.json',state)
    # A single POST only: do not retry an uncertain paid request automatically.
    response=api(base,f'/api/jobs/{args.job}/next',token=online['token'])
    print(json.dumps({**rerun,'running_stage':response['running_stage']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
