"""Regenerate the confirmed camera with green geometry-only masks.

Runs only on explicit user request, using the explicitly configured GPT provider.
Old captures/depth stay intact. Successful faces resume without another POST.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import re
import shutil
import sys
import time
import numpy as np
from PIL import Image
from novel_pipeline import (read,digest,FACE_NAMES,CAPTURE_VERSION,MASK_COLOR,
                            REPAIR_PROMPT,repair_input)
from novel_geometry import repair_masks,repair_faces,MASK_CLEANUP
from gpt_rgb_experiment import assemble,REFERENCE,MODEL
from rerun_novel import write,api
from face_quality_io import assess_rgb
from face_requests import ordered_face_map

VERSION='gpt-green-geometry-open3-smooth1-v5-provider'


def validate_capture(folder,preview):
    if preview.get('capture_version')!=CAPTURE_VERSION:raise ValueError('Fresh geometry-only capture required')
    capture=folder/preview['path']
    if not capture.resolve().is_relative_to(folder.resolve()):raise ValueError('Invalid capture path')
    for name,expected in [('raw.png',preview['raw_sha256']),('mask.png',preview['mask_sha256'])]:
        if digest(capture/name)!=expected:raise ValueError('Capture changed: '+name)
    if preview.get('mask_cleanup')!=MASK_CLEANUP:
        raise ValueError('Capture mask settings differ from production settings')
    geometry=np.asarray(Image.open(capture/'geometry-mask-raw.png'))>0
    if digest(capture/'geometry-mask-raw.png')!=preview['raw_geometry_sha256']:raise ValueError('Raw geometry mask changed')
    for face in FACE_NAMES:
        path=capture/f'{face}-render-alpha.png'
        if digest(path)!=preview['alpha_faces'][face]['sha256']:raise ValueError('Alpha changed: '+face)
    expected_masks=repair_masks(geometry)
    for key,filename in [('geometry','geometry-mask.png')]:
        if not np.array_equal(np.asarray(Image.open(capture/filename))>0,expected_masks[key]):raise ValueError('Mask cleanup mismatch: '+filename)
    mask=np.asarray(Image.open(capture/'mask.png'))>0
    if not np.array_equal(mask,expected_masks['combined']):raise ValueError('Repair mask differs from cleaned geometry')
    raw=np.asarray(Image.open(capture/'raw.png').convert('RGB'))
    masked=np.asarray(Image.open(capture/'masked.png').convert('RGB'))
    if not np.all(masked[mask]==MASK_COLOR) or not np.array_equal(masked[~mask],raw[~mask]):
        raise ValueError('Invalid green preview')
    for face in repair_faces(raw,mask):
        name=face['name'];expected=face['mask']
        if not np.array_equal(np.asarray(Image.open(capture/f'{name}-mask.png'))>0,expected):
            raise ValueError('Face mask cleanup mismatch: '+name)
    return capture


def generate(job,preview,baseline,repair_image,decode_image,key,download_error,preflight=None,*,provider='company'):
    """Dependency injection keeps paid-request/resume behavior testable offline."""
    folder=job/'novel'/baseline['revision'];capture=validate_capture(folder,preview)
    out=capture/'repair';out.mkdir(exist_ok=True);started=time.perf_counter()
    expected_size=(2048,2048) if provider=='company' else None
    request=dict(version=VERSION,model=MODEL,provider=provider,prompt=REPAIR_PROMPT,image_count=1,
        input_size=[1024,1024],output_size='2048x2048' if expected_size else 'provider-native-square',mask_color=list(MASK_COLOR),
        mask_source=preview['mask_source'],mask_cleanup=MASK_CLEANUP)
    write(out/'request.json',request)
    def repair_face(face):
        base=np.asarray(Image.open(capture/f'{face}.png').convert('RGB'))
        mask=np.asarray(Image.open(capture/f'{face}-mask.png'))>0
        if not mask.any():
            return dict(face=face,model=MODEL,skipped=True,mask_pixels=0,seconds=0),assess_rgb(capture,face,out)
        input_path=out/f'{face}-api-input.png';model_path=out/f'{face}-model.png';receipt=out/f'{face}.json'
        Image.fromarray(repair_input(base,mask)).save(input_path)
        expected=digest(input_path);cached=read(receipt) if receipt.exists() else {}
        if (cached.get('api_input_sha256')==expected and cached.get('prompt')==REPAIR_PROMPT
                and cached.get('request_version')==VERSION and model_path.exists()
                and cached.get('provider')==provider
                and cached.get('model_sha256')==digest(model_path)):
            quality=assess_rgb(capture,face,out)
            print(f'{face}: verified cached GPT result',flush=True)
            return cached,quality
        request_dir=out/f'{face}-{provider}';request_dir.mkdir(exist_ok=True)
        marker=request_dir/'submitted.json'
        if marker.exists():raise RuntimeError(f'{face}: previous POST outcome uncertain; no automatic paid retry')
        # Cached returns need no provider credentials. Validate a live provider
        # before recording any possibly billable submission.
        face_key=preflight() if preflight is not None else key
        write(marker,dict(started=time.time(),input_sha256=expected,request_version=VERSION))
        begin=time.perf_counter();cache={}
        print(f'GPT/{provider} {face}: green mask {mask.mean():.2%}; input 1024x1024',flush=True)
        try:
            repair_image(input_path,request_dir,MODEL,REPAIR_PROMPT,face_key,result_cache=cache)
        except download_error:
            if 'result' not in cache:raise
            # Only redownload the existing response. Never repeat a paid POST.
            repair_image(input_path,request_dir,MODEL,REPAIR_PROMPT,face_key,result_cache=cache)
        model=decode_image((request_dir/'api_original.image').read_bytes()).convert('RGB')
        if expected_size is not None and model.size!=expected_size:
            raise ValueError(f'{face}: unexpected company GPT output size {model.size}')
        if model.width!=model.height or model.width<512:
            raise ValueError(f'{face}: expected a square GPT image at least 512 pixels, got {model.size}')
        model.save(model_path)
        row=dict(face=face,model=MODEL,skipped=False,mask_pixels=int(mask.sum()),
            seconds=time.perf_counter()-begin,api_input_sha256=expected,model_sha256=digest(model_path),
            prompt=REPAIR_PROMPT,request_version=VERSION,provider=provider,input_size=[1024,1024],output_size=list(model.size))
        write(receipt,row)
        quality=assess_rgb(capture,face,out)
        print(f'{face}: saved and hashed ({row["seconds"]:.1f}s)',flush=True)
        return row,quality
    repaired=ordered_face_map(repair_face,FACE_NAMES)
    rows=[row for row,quality in repaired];qualities=[quality for row,quality in repaired]
    result=assemble(folder,preview,baseline,MODEL,request,rows,time.perf_counter()-started,qualities)
    result['external_provider']='Company API · GPT Image 2' if provider=='company' else 'GrsAI · GPT Image 2'
    result['description']='GPT Image 2 绿幕补洞：仅几何未知区域；Mask 做 3×3 开运算、小于 16 像素连通块清理，以及高斯 σ=1 px 羽化后取 50% 边缘。洞内完整生成，RGB 外侧 4 像素羽化。尚未运行新深度。'
    if baseline.get('image'):
        result['comparison_note']='当前为绿幕 + 几何 Mask 重生成。历史版本使用旧黑色 Mask，修补范围和提示词不同；仅用于效果回看。切换历史版本不会改变后续输入。'
        result['comparison_variants']=[dict(id='gpt',label='GPT · 绿幕 + 几何',
            path=f'{preview["path"]}/repair/repair.json',image=result['image']),
            dict(id='gpt-black',label='历史 GPT · 黑色 Mask',
            path=f'{baseline["capture"]["path"]}/repair/repair.json',image=baseline['image'])]
        for row in baseline.get('comparison_variants',[]):
            if row['id']!='gpt':result['comparison_variants'].append({**row,'label':'历史 '+row['label']})
    result['depth_not_run']=True
    write(out/'repair.json',result)
    return result


def repair_fresh(job):
    """Stage-7 worker entry; publication/locking belongs to the local server."""
    from novel_pipeline import begin_repair
    current=begin_repair(job);preview=current['preview']
    if digest(job/preview['source']['file'])!=preview['source']['sha256']:
        raise ValueError('Frozen PLY changed')
    validate_capture(job/'novel'/preview['revision'],preview)
    from image_provider import selected_provider,client,preflight
    provider=read(job/'job.json').get('image_provider') or selected_provider();image_api=client(provider)
    source=dict(revision=preview['revision'],camera_id=current['camera_id'])
    return generate(job,preview,source,image_api.repair_image,image_api.decode_image,None,image_api.ImageDownloadError,
                    lambda:preflight(provider),provider=provider)


def run(job,port):
    from image_provider import selected_provider,client,preflight
    provider=read(job/'job.json').get('image_provider') or selected_provider();image_api=client(provider)
    repair_image,decode_image,ImageDownloadError=image_api.repair_image,image_api.decode_image,image_api.ImageDownloadError
    online=api(f'http://127.0.0.1:{port}','/api/state');state=read(job/'job.json')
    if state['busy'] or online['gpu_busy']:raise ValueError('A job is running')
    baseline=deepcopy(state['stages']['7']);folder=job/'novel'/baseline['revision']
    selection=read(folder/'selection.json');preview=selection['previews'][baseline['camera_id']]
    if preview['path']==baseline['capture']['path']:raise ValueError('Recapture the same camera before regenerating')
    if preview['source']!=baseline['capture']['source']:raise ValueError('Gaussian source changed')
    if digest(job/preview['source']['file'])!=preview['source']['sha256']:raise ValueError('Frozen PLY changed')
    validate_capture(folder,preview)
    key=preflight(provider)
    backup=job/'reruns'/('gpt-green-'+time.strftime('%Y%m%d-%H%M%S'));backup.mkdir(parents=True)
    for name in ('job.json','stage-7.json','stage-8.json','stage-9.json'):
        if (job/name).exists():shutil.copy2(job/name,backup/name)
    shutil.copy2(folder/'selection.json',backup/'selection.json')
    state.update(busy=True,running_stage=7,rgb_repair_run=preview['path'],error=None,
        message='正在用绿幕 + 几何 Mask 重新生成 GPT 六面图片')
    write(job/'job.json',state)
    try:
        result=generate(job,preview,baseline,repair_image,decode_image,key,ImageDownloadError,provider=provider)
        current=read(job/'job.json')
        if current['stages']!=state['stages'] or current.get('rgb_repair_run')!=preview['path']:
            raise ValueError('Job changed while generating; results saved but not published')
        selection=read(folder/'selection.json')
        if selection['current']['camera_id']!=baseline['camera_id']:raise ValueError('Confirmed camera changed')
        selection['current']['preview']=preview;selection['previews'][baseline['camera_id']]=preview
        write(folder/'selection.json',selection)
        current['stages'].pop('9',None)
        old_depth=current['stages'].pop('8',None)
        if old_depth:current['archived_depth_before_rgb_test']=dict(path=old_depth['path'],backup=str(backup/'stage-8.json'))
        current['stages']['7']=result
        current.update(busy=False,running_stage=None,error=None,message='GPT 绿幕 + 几何 六面补洞已完成；新深度尚未生成')
        current.pop('rgb_repair_run',None)
        write(job/'stage-7.json',result);write(job/'job.json',current)
        write(backup/'publication.json',dict(capture=preview['path'],image=result['image'],image_sha256=result['image_sha256']))
        print(f'Published {result["image"]}; old RGB/depth preserved at {backup}',flush=True)
    except Exception as exc:
        current=read(job/'job.json')
        if current.get('rgb_repair_run')==preview['path']:
            current.update(busy=False,running_stage=None,error='GPT 绿幕重生成未完成；已保存成功面，原结果保留',message='重生成失败')
            current.pop('rgb_repair_run',None);write(job/'job.json',current)
        message=re.sub(r'https?://\S+','[remote URL]',str(exc))
        raise RuntimeError(message[:800]) from None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();run(args.job,args.port)
