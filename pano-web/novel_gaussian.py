"""Stage 9: reference depth-anchored SHARP -> six patches -> append-only PLY.

GPU execution must use the reference run_cuda_python.cmd launcher. Inputs are
the current stage-7 composited RGB and stage-8 directly projected LingBot range.
"""
from pathlib import Path
import gc
import hashlib
import json
import sys
import time
import numpy as np
from PIL import Image
from novel_pipeline import read,write_json,digest,work_folder
from novel_geometry import FACE_NAMES,merge_faces
from novel_gaussian_geometry import REFERENCE,BLEND_REFERENCE_PIXELS,BLEND_REFERENCE_SHORT_SIDE,project_inputs,select_patch
from novel_gaussian_filter import capture_initializer,build_point_filter,PARAMETERS as POINT_PARAMETERS,NORMAL_PARAMETERS

VERSION='sharp-hard-original-grid-depth-v6-no-quality-gates'
from project_paths import MODELS, SHARP
CHECKPOINT=MODELS/'sharp_2572gikvuh.pt'
REFERENCE_FILES=('run_sharp_hard_depth.py','sharp_runtime.py','merge_hard_patch.py','gaussian_patch_io.py','fuse_and_validate_depth.py')


def prepare_signature(job):
    state=read(job/'job.json');rgb=state['stages'].get('7');depth=state['stages'].get('8')
    if not rgb or not depth:raise ValueError('请先完成第七步 RGB 和第八步深度')
    if depth.get('panorama_source')!='lingbot_raw_faces':raise ValueError('第九步要求第八步 LingBot 直接全景')
    folder=work_folder(job);manifest=read(folder/'pipeline.json')
    if rgb['revision']!=depth['revision'] or rgb['revision']!=manifest['revision']:raise ValueError('Scene revisions differ')
    if rgb['camera_id']!=depth['camera_id'] or depth['rgb']!=rgb['image']:raise ValueError('RGB/depth camera mismatch')
    if rgb['capture']['source']!=depth['source'] or depth['source']!=manifest['source']:raise ValueError('Frozen scene mismatch')
    files=dict(base=job/depth['source']['file'],rgb=folder/rgb['image'],
        depth=folder/depth['arrays']['completed'],mask=folder/rgb['capture']['path']/'mask.png')
    expected=dict(base=depth['source']['sha256'],rgb=rgb['image_sha256'],
        depth=depth['files']['panorama-completed.npy'],mask=rgb['capture']['mask_sha256'])
    from face_quality import POLICY_HASH
    bundle=depth.get('accepted_bundle')
    if not bundle or bundle.get('policy_hash')!=POLICY_HASH:raise ValueError('需要当前质量策略的联合 RGB-D 结果包；历史结果不可静默进入新补丁')
    if bundle['camera_id']!=rgb['camera_id'] or bundle['source']!=depth['source'] or bundle['rgb_sha256']!=rgb['image_sha256']:
        raise ValueError('Accepted bundle provenance mismatch')
    if (bundle['revision']!=depth['depth_revision'] or bundle['mask_sha256']!=rgb['capture']['mask_sha256']
        or read(folder/depth['path']/'accepted-bundle.json')!=bundle):raise ValueError('Accepted bundle publication mismatch')
    for name in ('accepted-bundle.json','accepted-rgb.png','accepted-support.png','effective-mask.png'):
        if digest(folder/depth['path']/name)!=depth['files'][name]:raise ValueError('Accepted bundle changed')
    files.update(rgb=folder/bundle['rgb'],mask=folder/bundle['effective_mask'],support=folder/bundle['support'])
    expected.update(rgb=depth['files']['accepted-rgb.png'],mask=depth['files']['effective-mask.png'],support=depth['files']['accepted-support.png'])
    for key,path in files.items():
        if digest(path)!=expected[key]:raise ValueError(f'{key} input hash mismatch')
    position=next(c for c in read(folder/'sample-cameras.json')['cameras'] if c['id']==rgb['camera_id'])['position_world_m']
    if position!=depth['position_world_m']:raise ValueError('Confirmed camera position changed')
    signature=dict(version=VERSION,inputs=expected,camera_id=rgb['camera_id'],position_world_m=position,
        quality_hash=POLICY_HASH,bundle_sha256=depth['files']['accepted-bundle.json'],
        depth_revision=depth['depth_revision'],checkpoint_sha256=digest(CHECKPOINT),
        reference={name:digest(REFERENCE/name) for name in REFERENCE_FILES},
        adapter={name:digest(Path(__file__).with_name(name)) for name in ('novel_gaussian.py','novel_gaussian_geometry.py',
            'novel_gaussian_filter.py','point_normals.py','point_distances.py','point_mask.py','quality_fusion.py')},
        settings=dict(depth_geometry='original_grid_camera_z',sharp_quality_gate=False,patch_quality_gate=False,coverage_policy='diagnostic_only',face_size=512,fov=100,internal_size=1536,layer0_grid=768,covariance_mode='surface_jacobian',surface_tangent_sigma=1.5,
            blend_reference_pixels=BLEND_REFERENCE_PIXELS,blend_reference_short_side=BLEND_REFERENCE_SHORT_SIDE,blend_ring_count=6,core_opacity_mode='full',
            point_filter=POINT_PARAMETERS,normal_estimation=NORMAL_PARAMETERS))
    return state,folder,files,signature


def file_hashes(out):
    return {p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and p.name not in ('result.json','cache.json','inference-cache.json')}


def verify_cached(out,record):
    return record.get('files') and all((out/name).is_file() and digest(out/name)==sha for name,sha in record['files'].items())


def run_reference_sharp(face):
    import run_sharp_hard_depth
    args=['run_sharp_hard_depth.py','--image',str(face/'rgb.png'),'--depth',str(face/'depth-input.npy'),
        '--hole-mask',str(face/'mask.png'),'--camera-json',str(face/'camera.json'),
        '--checkpoint',str(CHECKPOINT),'--covariance-mode','surface_jacobian','--surface-tangent-sigma','1.5','--output-dir',str(face/'sharp')]
    previous=sys.argv;loader=run_sharp_hard_depth.load_predictor;hooks=[];captures=[]
    cam=read(face/'camera.json')
    def load_with_capture(*args,**kwargs):
        model=loader(*args,**kwargs)
        def capture(module,inputs,output):
            capture_initializer(output,cam,face);captures.append(True)
        hooks.append(model.init_model.register_forward_hook(capture))
        return model
    try:
        run_sharp_hard_depth.load_predictor=load_with_capture
        sys.argv=args
        if np.asarray(Image.open(face/'mask.png')).any():
            run_sharp_hard_depth.main()
        else:
            # Empty-hole faces still belong to the full KNN reference cloud.
            # Capture their initializer without inventing a validation hole.
            device=run_sharp_hard_depth.torch.device('cuda')
            run_sharp_hard_depth.patch_gpu_svd();model=load_with_capture(CHECKPOINT,device)
            run_sharp_hard_depth.hard_predict(model,np.asarray(Image.open(face/'rgb.png').convert('RGB')),
                np.load(face/'depth-input.npy'),float(cam['intrinsic_K_opencv'][0][0]),device,'surface_jacobian',1.5,.05,.35,4.)
        if len(captures)!=1:raise ValueError('Expected exactly one initializer capture')
    finally:
        for hook in hooks:hook.remove()
        run_sharp_hard_depth.load_predictor=loader;sys.argv=previous


def repair_gaussians(job):
    start=time.perf_counter();job=job.resolve()
    state,folder,inputs,signature=prepare_signature(job)
    revision=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    relative=f"{state['stages']['7']['capture']['path']}/gaussians/{revision}"
    out=folder/relative;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        result=read(out/'result.json')
        if verify_cached(out,result):print('复用已校验的第九步结果',flush=True);return result
    write_json(out/'inputs.json',signature)
    # Keep the authoritative reference sources ahead of stale editable installs.
    sys.path.insert(0,str(REFERENCE));sys.path.insert(0,str(SHARP/'src'))
    import torch
    import ctypes
    import torch.utils.cpp_extension as cpp_extension
    if sys.platform=='win32':
        codepage=ctypes.windll.kernel32.GetConsoleOutputCP() or ctypes.windll.kernel32.GetACP()
        cpp_extension.SUBPROCESS_DECODE_ARGS=(f'cp{codepage}',)
    from gsplat.cuda._backend import _C
    if _C is None:raise RuntimeError('Reference CUDA backend is unavailable')
    from plyfile import PlyData,PlyElement
    from gaussian_patch_io import append_patch_to_base
    torch.set_num_threads(8);torch.manual_seed(42)
    rgb=np.asarray(Image.open(inputs['rgb']).convert('RGB'));radial=np.load(inputs['depth'],allow_pickle=False)
    mask=np.asarray(Image.open(inputs['mask']).convert('L'))>0
    support=np.asarray(Image.open(inputs['support']))>0
    bundle=state['stages']['8']['accepted_bundle'];depth_folder=folder/state['stages']['8']['path']
    accepted={r['face'] for r in bundle['faces'] if r['accepted']}
    valid_faces=[]
    for name in FACE_NAMES:
        path=depth_folder/f'{name}-accepted-valid.png'
        if digest(path)!=state['stages']['8']['files'][path.name]:raise ValueError('Accepted face validity changed')
        valid_faces.append(np.asarray(Image.open(path))>0 if name in accepted else None)
    from quality_retry import QualityNoContribution
    if not mask.any():raise QualityNoContribution('没有联合通过检查的待补像素')
    base=PlyData.read(inputs['base'],mmap='r')['vertex'].data
    patches=[];records=[]
    for i,name in enumerate(FACE_NAMES):
        face=out/name;face.mkdir(exist_ok=True)
        cache=read(face/'inference-cache.json') if (face/'inference-cache.json').exists() else None
        if cache and verify_cached(face,cache):
            print(f'{name}: 复用已校验 SHARP 与原始初始化点',flush=True);continue
        print(f'{i+1}/6 {name}：从当前 RGB / LingBot 全景投影同相机输入',flush=True)
        projected=project_inputs(rgb,radial,mask,i,signature['position_world_m'],support=support,face_valid=valid_faces[i],face_enabled=name in accepted)
        for key in ('rgb','mask','extrapolated','support'):
            array=projected[key];Image.fromarray(array.astype(np.uint8)*255 if array.dtype==bool else array).save(face/f'{key}.png')
        for key in ('depth_projected','depth_input'):np.save(face/(key.replace('_','-')+'.npy'),projected[key])
        write_json(face/'camera.json',projected['camera']);write_json(face/'projection.json',projected['diagnostics'])
        if name not in accepted or not projected['support'].any():
            accepted.discard(name);continue
        print(f'{name}：深度输入 → SHARP 生成/调整 → 沿学习射线单次回拉到原始网格目标深度 → 参考表面协方差',flush=True)
        run_reference_sharp(face)
        write_json(face/'inference-cache.json',dict(files=file_hashes(face)))
        gc.collect();torch.cuda.empty_cache()
    point_filter=build_point_filter(out,lambda message:print(message,flush=True),accepted_faces=accepted)
    for i,name in enumerate(FACE_NAMES):
        face=out/name;hole=np.asarray(Image.open(face/'mask.png'))>0
        cam=read(face/'camera.json');projection=read(face/'projection.json')
        if not hole.any():
            empty=np.empty(0,dtype=base.dtype);PlyData([PlyElement.describe(empty,'vertex')],text=False).write(face/'patch-world.ply')
            record=dict(face=name,skipped=True,count=0,projection=projection)
        else:
            sharp=face/'sharp';acceptance=read(sharp/'sharp_hard_acceptance.json')
            vertices=PlyData.read(sharp/'sharp_hard_layer0_camera.ply')['vertex'].data
            layer_z=np.load(sharp/'sharp_hard_layer0_z.npy',allow_pickle=False)
            tail,transform,ids,labels,weights,selection=select_patch(vertices,layer_z,hole,i,cam,np.load(face/'point-keep.npy'),
                support=np.asarray(Image.open(face/'support.png'))>0,valid_faces=valid_faces)
            print(f"{name}：补丁 {selection['before_point_filter']:,} → {len(tail):,}，剔除 {selection['point_filter_removed']:,}",flush=True)
            np.save(face/'selected-rows.npy',ids);np.save(face/'selected-weights.npy',weights)
            Image.fromarray(labels).save(face/'ring-labels.png')
            PlyData([PlyElement.describe(tail,'vertex')],text=False,byte_order='<').write(face/'patch-camera.ply')
            append_patch_to_base(np.empty(0,dtype=base.dtype),tail,transform,face/'patch-world.ply')
            record=dict(face=name,skipped=False,count=len(tail),projection=projection,
                selection=selection,sharp_acceptance=acceptance)
            del vertices,tail,transform
        records.append(record);patches.append(PlyData.read(face/'patch-world.ply')['vertex'].data.copy())
        gc.collect();torch.cuda.empty_cache()
    point_filter.update(patch_before=sum(r.get('selection',{}).get('before_point_filter',0) for r in records),
        patch_kept=sum(r['count'] for r in records),patch_removed=sum(r.get('selection',{}).get('point_filter_removed',0) for r in records))
    write_json(out/'point-filter.json',point_filter)
    print('合并六面补丁；逐字节保留原始高斯前缀',flush=True)
    patch=np.concatenate(patches);del patches
    if not len(patch):raise QualityNoContribution('严格点剔除后没有新增高斯')
    PlyData([PlyElement.describe(patch,'vertex')],text=False,byte_order='<').write(out/'patch.ply')
    combined=np.empty(len(base)+len(patch),dtype=base.dtype);combined[:len(base)]=base;combined[len(base):]=patch
    PlyData([PlyElement.describe(combined,'vertex')],text=False,byte_order='<').write(out/'merged.ply')
    count=len(combined);del combined,patch
    print('实际渲染原始场景、补丁和合并场景，记录深度与覆盖率诊断',flush=True)
    render_report=render_diagnostics(out,inputs['base'],records)
    warnings=[]
    report=dict(status='accepted',quality_gate_enabled=False,base_prefix_exact=None,source_sha256_unchanged=digest(inputs['base'])==signature['inputs']['base'],
        base_count=len(base),added_count=count-len(base),final_count=count,face_results=records,render=render_report,point_filter=point_filter)
    if not report['source_sha256_unchanged']:raise AssertionError('Frozen base changed during generation')
    write_json(out/'acceptance.json',report)
    # Inputs must still refer to this exact published RGB-D pair before returning.
    current=read(job/'job.json')
    if current['stages']['7']!=state['stages']['7'] or current['stages']['8']!=state['stages']['8']:raise ValueError('RGB-D changed during generation')
    files=file_hashes(out)
    result=dict(kind='gaussian-repair',revision=state['stages']['8']['revision'],patch_revision=revision,
        quality_hash=signature['quality_hash'],accepted_bundle_sha256=signature['bundle_sha256'],
        depth_revision=signature['depth_revision'],camera_id=signature['camera_id'],position_world_m=signature['position_world_m'],
        path=relative,source=state['stages']['8']['source'],inputs=signature['inputs'],
        preview=f'{relative}/merged.ply',download=f'{relative}/merged.ply',patch=f'{relative}/patch.ply',
        image=f'{relative}/merged-panorama.png',before_image=f'{relative}/base-panorama.png',
        patch_image=f'{relative}/patch-panorama.png',count=count,base_count=len(base),added_count=count-len(base),
        sha256=files['merged.ply'],patch_sha256=files['patch.ply'],files=files,acceptance=report,warnings=warnings,seconds=time.perf_counter()-start,
        description='第七步融合 RGB + 第八步 LingBot 直接深度 → 参考 SHARP 深度注入、网络调整、原始网格目标深度单次回拉与协方差更新 → 按初始化身份应用 82° / K=64 / 15 cm 点剔除 → Mask 及六环补丁 → 追加到原始高斯。')
    write_json(out/'result.json',result);return result


def render_diagnostics(out,base_path,records):
    import torch
    from sharp.utils.gaussians import load_ply
    from merge_hard_patch import render_gaussians,depth_metrics,rgb_metrics
    device=torch.device('cuda');report={};images={}
    for kind,path in [('base',base_path),('patch',out/'patch.ply'),('merged',out/'merged.ply')]:
        gaussians,_=load_ply(path);gaussians=gaussians.to(device);rows=[];views=[]
        for name,record in zip(FACE_NAMES,records):
            face=out/name;cam=read(face/'camera.json')
            rgb,z,alpha=render_gaussians(gaussians,np.asarray(cam['w2c']),np.asarray(cam['intrinsic_K_opencv']),512,512,1.,device)
            views.append(rgb);Image.fromarray(rgb).save(out/f'{kind}-{name}.png')
            np.save(out/f'{kind}-{name}-z.npy',z);np.save(out/f'{kind}-{name}-alpha.npy',alpha)
            hole=np.asarray(Image.open(face/'mask.png'))>0;target=np.load(face/'depth-input.npy')
            if hole.any():
                metrics=depth_metrics(z,alpha,target,hole,allow_empty=True)
                if kind=='patch':metrics['rgb']=rgb_metrics(rgb,alpha,np.asarray(Image.open(face/'rgb.png')),hole)
                rows.append(dict(face=name,**metrics))
        images[kind]=merge_faces(views);Image.fromarray(images[kind]).save(out/f'{kind}-panorama.png')
        report[kind]=rows;del gaussians;gc.collect();torch.cuda.empty_cache()
    report.update(status='diagnostic_only',quality_gate_enabled=False,coverage_policy='diagnostic_only')
    write_json(out/'render-acceptance.json',report)
    return report
