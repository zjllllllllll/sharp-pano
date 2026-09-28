"""Strict stage-8 RGB-D gate. GPU imports remain inside the worker entry."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import inspect
import shutil
import numpy as np
from PIL import Image
from face_quality import POLICY_HASH,VERSION as QUALITY_VERSION,record,joint
from face_quality_io import assess_depth,verify_report,image,read,write,digest
from novel_geometry import FACE_NAMES,camera,six_faces
from quality_fusion import merge_accepted,compose_accepted


def complete_depth(job):
    from novel_depth import REFERENCE,LINGBOT,CHECKPOINT,render_faces,write_depth_outputs,depth_warnings
    from novel_pipeline import work_folder
    sys.path.insert(0,str(REFERENCE));sys.path.insert(0,str(LINGBOT))
    state=read(job/'job.json');repair=state['stages'].get('7')
    if not repair or repair.get('quality_hash')!=POLICY_HASH:
        raise ValueError('本次深度需要已完成当前质量检查的 RGB；历史结果仅可查看或离线诊断')
    folder=work_folder(job);manifest=read(folder/'pipeline.json');preview=repair['capture'];capture=folder/preview['path']
    source=manifest['source'];ply=job/source['file']
    if (repair['revision']!=manifest['revision'] or source!=preview['source'] or digest(ply)!=source['sha256']
        or digest(capture/'mask.png')!=preview['mask_sha256'] or digest(folder/repair['image'])!=repair['image_sha256']):
        raise ValueError('Frozen RGB/Mask/Gaussian input changed')
    qualities={q['face']:q for q in repair['quality_faces']}
    if set(qualities)!=set(FACE_NAMES):raise ValueError('Missing RGB face assessments')
    for q in qualities.values():verify_report(capture,q)
    position=next(c for c in read(folder/'sample-cameras.json')['cameras'] if c['id']==repair['camera_id'])['position_world_m']
    if not CHECKPOINT.is_file():raise FileNotFoundError('参考项目缺少 LingBot 权重')
    panorama_rgb=image(folder/repair['image'],'RGB')
    projected=six_faces(panorama_rgb,image(capture/'mask.png')>0)
    guidance={face['name']:hashlib.sha256(face['rgb'].tobytes()).hexdigest() for face in projected}
    signature=dict(version='lingbot-fused-panorama-guidance-v2',policy_hash=POLICY_HASH,source=source,
        rgb_sha256=repair['image_sha256'],mask_sha256=preview['mask_sha256'],guidance=guidance,
        rgb_quality=repair['quality_faces'],position_world_m=position,checkpoint_sha256=digest(CHECKPOINT),
        adapter={n:digest(Path(__file__).with_name(n)) for n in ('quality_depth.py','quality_fusion.py','face_quality.py','face_quality_io.py','novel_depth.py','novel_geometry.py')},
        model_sha256=digest(LINGBOT/'mdm/model/v2.py'))
    revision=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    relative=f'{preview["path"]}/depth/{revision}';out=folder/relative;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        cached=read(out/'result.json')
        if all((out/n).is_file() and digest(out/n)==h for n,h in cached['files'].items()):return cached
    write(out/'inputs.json',signature)
    # Expensive raw outputs are keyed independently of assessment thresholds.
    render_key=hashlib.sha256(json.dumps(dict(source=source,position=position,
        code=inspect.getsource(render_faces)),sort_keys=True).encode()).hexdigest()[:16]
    render_cache=capture/'depth-render'/render_key;render_cache.mkdir(parents=True,exist_ok=True)
    render_files=[render_cache/f'{n}-{s}' for n in FACE_NAMES for s in ('rendered-z.npy','alpha.npy','camera.json')]
    render_receipt=read(render_cache/'receipt.json') if (render_cache/'receipt.json').exists() else {}
    if not all(p.is_file() and render_receipt.get(p.name)==digest(p) for p in render_files):
        render_faces(ply,position,render_cache);render_receipt={p.name:digest(p) for p in render_files};write(render_cache/'receipt.json',render_receipt)
    for p in render_files:shutil.copy2(p,out/p.name)
    raw=[np.load(out/f'{n}-rendered-z.npy',allow_pickle=False) for n in FACE_NAMES]
    import torch
    from mdm.model.v2 import MDMModel
    model=None;device=torch.device('cuda');records=[];completed=[];lingbot=[];valids=[];rgb_faces=[];pairs=[];depth_quality=[]
    requested=image(capture/'mask.png')>0
    try:
        for i,name in enumerate(FACE_NAMES):
            qrgb=qualities[name];rgb=projected[i]['rgb']
            source_z=raw[i];hole=(image(capture/f'{name}-mask.png')>0)|~np.isfinite(source_z)|(source_z<=.001)
            depth_input=source_z.copy();depth_input[hole]=0
            Image.fromarray(rgb).save(out/f'{name}-rgb.png');Image.fromarray(hole.astype(np.uint8)*255).save(out/f'{name}-hole.png')
            np.save(out/f'{name}-input-z.npy',depth_input)
            begin=time.perf_counter();skip_rgb=qrgb['status'] not in ('passed','not_needed')
            if skip_rgb:
                pred=np.zeros_like(source_z);model_valid=np.zeros(hole.shape,bool)
                qdepth=record('depth','rejected',['upstream_rejected'],face=name)
            elif not hole.any():
                pred=source_z.copy();model_valid=np.isfinite(pred)&(pred>.001)
                qdepth=record('depth','not_needed',face=name)
            else:
                inference_key=hashlib.sha256(json.dumps(dict(version='fused-panorama-lingbot-v2',face=name,
                    rendered=render_receipt[f'{name}-rendered-z.npy'],rgb=guidance[name],input_depth=digest(out/f'{name}-input-z.npy'),
                    checkpoint=signature['checkpoint_sha256'],model=signature['model_sha256'],resolution_level=9,fp16=True,
                    K=camera(i,size=512,fov=100)['K_pixel_edge']),sort_keys=True).encode()).hexdigest()[:16]
                raw_cache=capture/'depth-inference'/inference_key;raw_cache.mkdir(parents=True,exist_ok=True)
                receipt_path=raw_cache/'receipt.json';receipt=read(receipt_path) if receipt_path.exists() else {}
                raw_files=[raw_cache/f'{name}-lingbot-z.npy',raw_cache/f'{name}-model-valid.png']
                if not all(p.is_file() and receipt.get(p.name)==digest(p) for p in raw_files):
                    if model is None:model=MDMModel.from_pretrained(str(CHECKPOINT)).eval().to(device)
                    print(f'LingBot {name}：使用融合 RGB 全景重新截取的本面，待补 {hole.mean():.2%}',flush=True)
                    k=np.asarray(camera(i,size=512,fov=100)['K_pixel_edge'],np.float32);k[:2]/=512
                    with torch.inference_mode():
                        output=model.infer(torch.tensor(rgb.astype(np.float32)/255,device=device).permute(2,0,1)[None],
                            depth_in=torch.tensor(depth_input,device=device),intrinsics=torch.tensor(k,device=device)[None],
                            resolution_level=9,apply_mask=True,use_fp16=True)
                    pred=output['depth'].squeeze().detach().cpu().numpy().astype(np.float32)
                    model_valid=output['mask'].squeeze().detach().cpu().numpy().astype(bool);del output
                    if pred.shape!=(512,512) or model_valid.shape!=(512,512):raise ValueError('LingBot grid mismatch')
                    np.save(raw_files[0],pred);Image.fromarray(model_valid.astype(np.uint8)*255).save(raw_files[1])
                    write(receipt_path,{p.name:digest(p) for p in raw_files})
                else:
                    pred=np.load(raw_files[0],allow_pickle=False);model_valid=image(raw_files[1])>0
                for p in raw_files:shutil.copy2(p,out/p.name)
                qdepth=assess_depth(capture,out,name,out,position)
            # Keep an explicit raw/empty output even for skipped faces, for UI and provenance.
            if skip_rgb or not hole.any():
                np.save(out/f'{name}-lingbot-z.npy',pred);Image.fromarray(model_valid.astype(np.uint8)*255).save(out/f'{name}-model-valid.png')
                write(out/f'{name}-depth-quality.json',qdepth)
            decision=joint(qrgb,qdepth);valid=model_valid&np.isfinite(pred)&(pred>.001)
            accepted=decision['accepted'];valids.append(valid if accepted else None)
            # Comparison only: no calibration/extrapolation can affect the acceptance gate.
            comparison=source_z.copy();comparison[hole]=np.where(valid,pred,0)[hole]
            np.save(out/f'{name}-completed-z.npy',comparison)
            item=dict(face=name,skipped=not hole.any(),quality_skipped=skip_rgb,seconds=time.perf_counter()-begin,
                hole_pixels=int(hole.sum()),outside_changed_pixels=0,sha256=digest(out/f'{name}-completed-z.npy'),quality=qdepth,joint=decision,
                comparison_only=True,comparison_method='raw_inside_mask')
            write(out/f'{name}-result.json',item);records.append(item);completed.append(comparison);lingbot.append(pred)
            rgb_faces.append(rgb)
            pairs.append(dict(face=name,**decision));depth_quality.append(qdepth)
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
    fused=merge_accepted(rgb_faces,lingbot,valids)
    accepted_rgb,effective,unresolved=compose_accepted(image(capture/'raw.png','RGB'),fused['rgb'],requested,fused['support'])
    for name,array in [('accepted-rgb',accepted_rgb),('accepted-support',fused['support']),('effective-mask',effective),
                       ('unresolved-mask',unresolved),('accepted-provenance',fused['provenance']),('depth-conflict',fused['conflict'])]:
        Image.fromarray(array.astype(np.uint8)*255 if array.dtype==bool else array).save(out/f'{name}.png')
    for i,name in enumerate(FACE_NAMES):
        Image.fromarray((np.zeros((512,512),bool) if valids[i] is None else valids[i]).astype(np.uint8)*255).save(out/f'{name}-accepted-valid.png')
    metrics=write_depth_outputs(out,raw,completed,lingbot,requested,position,accepted_fusion=fused)
    bundle=dict(version=QUALITY_VERSION,policy_hash=POLICY_HASH,revision=revision,camera_id=repair['camera_id'],source=source,
        rgb_sha256=repair['image_sha256'],mask_sha256=preview['mask_sha256'],guidance='fused_panorama_reprojected',depth_fusion='valid_weighted_no_cross_face_rejection',faces=pairs,
        rgb=f'{relative}/accepted-rgb.png',depth=f'{relative}/panorama-completed.npy',support=f'{relative}/accepted-support.png',
        effective_mask=f'{relative}/effective-mask.png',unresolved_mask=f'{relative}/unresolved-mask.png',
        provenance=f'{relative}/accepted-provenance.png',effective_pixels=int(effective.sum()),unresolved_pixels=int(unresolved.sum()))
    write(out/'accepted-bundle.json',bundle)
    files={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='result.json'}
    result=dict(kind='depth-completion',revision=repair['revision'],depth_revision=revision,camera_id=repair['camera_id'],
        path=relative,source=source,position_world_m=position,rgb=repair['image'],image=f'{relative}/panorama-completed.png',
        download=f'{relative}/depths.npz',files=files,views={n:f'{relative}/panorama-{n}.png' for n in ('rendered','input','completed','hole')},
        arrays={n:f'{relative}/panorama-{n}.npy' for n in ('rendered','input','completed')},face_results=records,
        quality_version=QUALITY_VERSION,quality_hash=POLICY_HASH,quality_faces=depth_quality,accepted_bundle=bundle,**metrics,
        semantics='ERP radial distance in metres; six perspective faces are camera-Z in metres',warnings=depth_warnings(metrics,records),
        model='LingBot-Depth',description='先融合 RGB 全景，再截取六面引导 LingBot；有效深度直接加权拼成全景，不做跨面深度拒绝。')
    write(out/'result.json',result);return result
