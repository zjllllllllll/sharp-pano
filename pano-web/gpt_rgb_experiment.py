"""Compare GPT Image 2 vs cached Seedream on identical faces/prompt. No depth.

Use the user's existing company gateway implementation, credentials and OSS
transport. Run from the project root through run_cuda_python.cmd.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
import uuid
import numpy as np
from PIL import Image
from novel_pipeline import read,write_json,digest,FACE_NAMES
from novel_composite import COMPOSITE,compose_faces

from project_paths import PROVIDERS as REFERENCE
MODEL='gpt-image-2'


def clone_capture(folder,preview):
    result=deepcopy(preview)
    capture=folder/preview['path'];new=capture.parent/uuid.uuid4().hex[:16];new.mkdir()
    for name in ['raw.png','mask.png','masked.png']+[f'{f}{s}.png' for f in FACE_NAMES for s in ('','-mask','-masked')]:
        shutil.copy2(capture/name,new/name)
    result.update(path=new.relative_to(folder).as_posix(),reused_capture=preview['path'])
    write_json(new/'capture.json',result)
    (new/'repair').mkdir()
    return result


def assemble(folder,preview,source_result,model,request,rows,seconds,quality=None):
    capture=folder/preview['path'];out=capture/'repair';models=[]
    for name in FACE_NAMES:
        path=out/f'{name}-model.png'
        if any(row['face']==name and row.get('skipped') for row in rows):path=capture/f'{name}.png'
        with Image.open(path if path.exists() else capture/f'{name}.png') as image:
            models.append(np.asarray(image.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)))
    faces,stats=compose_faces(capture,out,models,quality)
    for row,values in zip(rows,faces,strict=True):
        row.update(values,composite=COMPOSITE,sha256=digest(out/f'{row["face"]}-repaired.png'))
        write_json(out/f'{row["face"]}.json',row)
    result=dict(kind='repair',revision=source_result['revision'],camera_id=source_result['camera_id'],
        capture=preview,image=f'{preview["path"]}/repair/panorama.png',image_sha256=digest(out/'panorama.png'),
        faces=rows,request=request,composite=COMPOSITE,seconds=seconds,**stats,
        external_provider='Company API · GPT Image 2' if model==MODEL else 'Volcengine Ark Seedream (cached API images)',
        description=f'{model} 六面 RGB 补洞；Mask 内完全使用新图，向外 4 像素羽化，过渡带外逐字节保留原图。仅 RGB，未运行新深度。')
    if quality is not None:
        from face_quality import VERSION,POLICY_HASH
        result.update(quality_version=VERSION,quality_hash=POLICY_HASH,quality_faces=quality,
            quality_phase='rgb_provisional',unresolved_mask=f'{preview["path"]}/repair/unresolved-mask.png')
    write_json(out/'repair.json',result)
    return result


def run(job):
    from image_provider import client,selected_provider,preflight
    provider=selected_provider();image_api=client(provider)
    repair_image,decode_image,ImageDownloadError=image_api.repair_image,image_api.decode_image,image_api.ImageDownloadError
    state=read(job/'job.json')
    if state['busy']:raise ValueError('Wait for the active job')
    baseline=state['stages']['7'];folder=job/'novel'/baseline['revision'];old=folder/baseline['capture']['path']
    if baseline['request']['image_count']!=1:raise ValueError('Expected single-image baseline')
    key=preflight(provider)
    status_path=job/'gpt-rgb-experiment.json'
    if status_path.exists():
        status=read(status_path)
        if status['baseline_image_sha256']!=baseline['image_sha256']:raise ValueError('Experiment baseline changed')
        preview=status['gpt_capture'];seed_preview=status['seedream_capture']
    else:
        preview=clone_capture(folder,baseline['capture']);seed_preview=clone_capture(folder,baseline['capture'])
        status=dict(baseline_image_sha256=baseline['image_sha256'],gpt_capture=preview,
            seedream_capture=seed_preview,created=time.time(),status='running',completed=[])
        write_json(status_path,status)
    capture=folder/preview['path'];out=capture/'repair'
    prompt=baseline['request']['prompt'];rows=[];started=time.perf_counter()
    for prior in baseline['faces']:
        name=prior['face'];record=out/f'{name}.json';model_path=out/f'{name}-model.png'
        if prior['skipped']:
            rows.append(dict(face=name,model=MODEL,skipped=True,seconds=0,mask_pixels=0));continue
        input_path=old/'repair'/f'{name}-api-input.png'
        expected=digest(input_path)
        cached=read(record) if record.exists() else {}
        if (cached.get('api_input_sha256')==expected and cached.get('prompt')==prompt and model_path.exists()
                and cached.get('model_sha256')==digest(model_path)):
            rows.append(cached);print(f'{name}: cached GPT output',flush=True);continue
        request_dir=out/f'{name}-company';request_dir.mkdir(exist_ok=True)
        # A marker without saved output means an earlier POST may have been billed.
        # Never silently repeat it when this experiment is resumed.
        marker=request_dir/'submitted.json'
        if marker.exists():raise RuntimeError(f'{name}: previous request has uncertain outcome; no automatic paid retry')
        print(f'GPT {name}: one unchanged 1024x1024 input; same prompt; output 2048x2048',flush=True)
        write_json(marker,dict(started=time.time(),input_sha256=expected))
        begin=time.perf_counter();cache={}
        try:
            try:repair_image(input_path,request_dir,MODEL,prompt,key,result_cache=cache)
            except ImageDownloadError:
                # Retry only GET of the already generated image, never POST.
                if 'result' not in cache:raise
                repair_image(input_path,request_dir,MODEL,prompt,key,result_cache=cache)
        except Exception:
            status.update(status='failed',failed_face=name);write_json(status_path,status);raise
        raw=(request_dir/'api_original.image').read_bytes();model_image=decode_image(raw).convert('RGB')
        model_image.save(model_path);shutil.copy2(input_path,out/f'{name}-api-input.png')
        row=dict(face=name,model=MODEL,skipped=False,seconds=time.perf_counter()-begin,
            mask_pixels=prior['mask_pixels'],api_input_sha256=expected,model_sha256=digest(model_path),
            prompt=prompt,input_size=[1024,1024],output_size=list(model_image.size))
        rows.append(row);write_json(record,row)
        status['completed']=[r['face'] for r in rows if not r['skipped']];write_json(status_path,status)
        print(f'{name}: saved GPT output ({row["seconds"]:.1f}s)',flush=True)
    request=dict(model=MODEL,prompt=prompt,image_count=1,input_size=[1024,1024],output_size='2048x2048',
        version='company-gpt-image-2-comparison-v1')
    result=assemble(folder,preview,baseline,MODEL,request,rows,time.perf_counter()-started)
    seed_out=folder/seed_preview['path']/'repair'
    for row in baseline['faces']:
        if not row['skipped']:shutil.copy2(old/'repair'/f'{row["face"]}-model.png',seed_out/f'{row["face"]}-model.png')
    seed=assemble(folder,seed_preview,baseline,baseline['request']['model'],baseline['request'],deepcopy(baseline['faces']),0.)
    status.update(status='complete',gpt_result=f'{preview["path"]}/repair/repair.json',
        seedream_result=f'{seed_preview["path"]}/repair/repair.json',seconds=time.perf_counter()-started)
    write_json(status_path,status)
    print(json.dumps(dict(status='complete',gpt_image=result['image'],seedream_image=seed['image'],depth_run=False)),flush=True)


def publish(job,port=8770):
    """Publish only after all five outputs exist; preserve the old depth on disk."""
    from rerun_novel import api,write
    state=read(job/'job.json');status=read(job/'gpt-rgb-experiment.json')
    if state['busy'] or api(f'http://127.0.0.1:{port}','/api/state')['gpu_busy']:
        raise ValueError('A pipeline step is running')
    baseline=state['stages']['7']
    if status['status']!='complete' or baseline['image_sha256']!=status['baseline_image_sha256']:
        raise ValueError('Experiment is incomplete or its baseline changed')
    before=digest(job/'job.json');folder=job/'novel'/baseline['revision']
    result=read(folder/status['gpt_result']);seed=read(folder/status['seedream_result'])
    for candidate in (result,seed):
        if digest(folder/candidate['image'])!=candidate['image_sha256']:raise ValueError('Result image changed')
        if candidate['outside_feather_changed_pixels'] or candidate['inside_generated_mismatch_pixels']:
            raise ValueError('Blend validation failed')
    result['comparison_variants']=[
        dict(id='gpt',label='GPT Image 2 · 外羽化',path=status['gpt_result'],image=result['image']),
        dict(id='seedream-outer',label='Seedream · 外羽化',path=status['seedream_result'],image=seed['image']),
        dict(id='seedream-inner',label='Seedream · 旧内羽化',path=f'{baseline["capture"]["path"]}/repair/repair.json',image=baseline['image'])]
    result['depth_not_run']=True
    backup=job/'reruns'/('gpt-rgb-'+time.strftime('%Y%m%d-%H%M%S'));backup.mkdir(parents=True)
    for name in ('job.json','stage-7.json','stage-8.json','stage-9.json','stage-7.log','stage-8.log','stage-9.log'):
        if (job/name).exists():shutil.copy2(job/name,backup/name)
    selection_path=folder/'selection.json';selection=read(selection_path)
    shutil.copy2(selection_path,backup/'selection.json')
    if digest(job/'job.json')!=before:raise ValueError('Job changed during publication')
    selection['current']['preview']=result['capture']
    selection['previews'][result['camera_id']]=result['capture']
    write(selection_path,selection)
    write(folder/status['gpt_result'],result)
    write(job/'stage-7.json',result)
    state['stages'].pop('9',None)
    old_depth=state['stages'].pop('8',None)
    if old_depth:state['archived_depth_before_rgb_test']=dict(path=old_depth['path'],backup=str(backup/'stage-8.json'))
    state['stages']['7']=result
    state.update(busy=False,running_stage=None,error=None,message='GPT RGB 对比已完成；外羽化；未运行新深度，旧深度已保留')
    write(job/'job.json',state)
    status.update(published=True,backup=str(backup));write(job/'gpt-rgb-experiment.json',status)
    print(json.dumps(dict(published=True,stage=7,variants=3,depth_run=False,backup=str(backup))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path)
    parser.add_argument('--publish',action='store_true');parser.add_argument('--port',type=int,default=8770)
    args=parser.parse_args()
    if args.publish:publish(args.job,args.port)
    else:run(args.job)
