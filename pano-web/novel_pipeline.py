"""Stages 6/7 state and artifacts; no mutations to the original five stages."""
from pathlib import Path
from copy import deepcopy
import base64
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import uuid
import numpy as np
from PIL import Image
from novel_geometry import next_camera,merge_faces,six_faces,FACE_NAMES,MASK_COLOR,CAPTURE_VERSION,merge_alpha_holes,ALPHA_THRESHOLD,ALPHA_THRESHOLD_U8
from novel_geometry import clean_mask,repair_masks,repair_faces,MASK_CLEANUP
from novel_composite import COMPOSITE,compose_faces

APP=Path(__file__).resolve().parent
ROOT=APP.parent
EXPERIMENT=ROOT/'room-outline-experiment'
from project_paths import PROVIDERS as REFERENCE, CPU_LAUNCHER as LAUNCHER
sys.path.insert(0,str(EXPERIMENT))
from batch import digest,write_json
from candidate_selection import VERSION as SELECTION_VERSION

VERSION='novel-gaussian-mask-v1'
CAM_ID=re.compile(r'^C\d{3,6}$')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def source_ply(job,variant):
    result=read(job/'stage-5.json')
    # Latest separately-built densification metadata is authoritative too.
    dense_path=job/'gaussian-dense-5.json'
    density=read(dense_path) if dense_path.exists() else result.get('densification')
    if variant=='dense':
        if not density or density.get('version')!=3:raise ValueError('请先生成调整后高斯的加密版本')
        result=density['dense']
    else:
        if variant not in ('masked','unmasked'):raise ValueError('Invalid Gaussian variant')
        variants=density['baseline_variants'] if density and density.get('version')==3 else result.get('variants',{})
        if variant not in variants:raise ValueError('Missing requested Gaussian variant')
        result=variants[variant]
    name=result['preview']
    if Path(name).name!=name or not name.endswith('.ply'):raise ValueError('Invalid Gaussian source')
    path=job/name
    if not path.is_file():raise FileNotFoundError('Missing final Gaussian PLY')
    return dict(file=name,sha256=digest(path),count=result['count'],variant=variant)


def prepare(job):
    state=read(job/'job.json');variant=state.get('novel_scene_variant','masked')
    policy=state.get('camera_selection_policy','four-neighbor')
    if policy not in ('four-neighbor','max-mask'):raise ValueError('Unknown camera selection policy')
    source=source_ply(job,variant)
    hashes={name:digest(job/name) for name in ('input.png','depth.npy','cloud-full.bin')}
    valid_name='depth-valid.npy'
    hashes[valid_name]=digest(job/valid_name) if (job/valid_name).is_file() else None
    fingerprint=dict(version=VERSION,source=source,inputs=hashes,selection=SELECTION_VERSION,
        mask_code=digest(EXPERIMENT/'occlusion.py'),geometry_code=digest(APP/'novel_geometry.py'),selection_policy=policy)
    revision=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()[:16]
    folder=job/'novel'/revision;folder.mkdir(parents=True,exist_ok=True)
    manifest_path=folder/'pipeline.json'
    if manifest_path.exists():return public_state(job,folder)
    # Reuse only when RGB, numeric depth AND actual initial cloud match exactly.
    mask_source=None
    cache=EXPERIMENT/'runs/20260922-adaptive-sampling'
    if (cache/'inputs.json').exists():
        for row in read(cache/'inputs.json'):
            if row['rgb_sha256']!=hashes['input.png']:continue
            candidate=cache/row['stem']
            valid_hash=digest(candidate/valid_name) if (candidate/valid_name).is_file() else None
            if valid_hash==hashes[valid_name] and all(digest(candidate/name)==hashes[name] for name in ('depth.npy','cloud-full.bin')):
                info=read(candidate/'sample-cameras.json')
                if info.get('selection_version')==SELECTION_VERSION and info.get('selection_complete'):
                    mask_source=candidate;break
    if mask_source is None:
        print('当前场景无完全匹配的候选缓存，按固定规则计算轮廓和未知空间 Mask。',flush=True)
        mask_source=folder/'sampling';mask_source.mkdir(exist_ok=True)
        for name in ('input.png','depth.npy','cloud-full.bin'):
            target=mask_source/name
            if not target.exists():os.link(job/name,target)
        if hashes[valid_name] and not (mask_source/valid_name).exists():os.link(job/valid_name,mask_source/valid_name)
        import outline
        from cameras import process
        if not (mask_source/'outline.json').exists():outline.process(mask_source)
        info=process(mask_source)
    else:print(f'精确复用候选与 Mask：{mask_source.name}；高斯截图仍在点击时重新渲染。',flush=True)
    if info.get('sampling_error') or info.get('selection_error'):
        raise ValueError(info.get('sampling_error') or info['selection_error'])
    info=deepcopy(info)
    # Keep the established candidate pool/poses; rank its cleaned geometry masks.
    geometry_dir=folder/'geometry-masks';geometry_dir.mkdir(exist_ok=True)
    cleaned={}
    for c in info.get('all_cameras',info['cameras']):
        raw=np.asarray(Image.open(mask_source/c['views']['mask']))>0
        mask=clean_mask(raw,periodic=True)
        Image.fromarray(mask.astype(np.uint8)*255).save(geometry_dir/f'{c["id"]}.png')
        cleaned[c['id']]=dict(raw_mask_fraction=float(raw.mean()),mask_fraction=float(mask.mean()),mask_cleanup=MASK_CLEANUP)
        c['mask_stats'].update(cleaned[c['id']])
    for c in info['cameras']:c['mask_stats'].update(cleaned[c['id']])
    info['mask_cleanup']=MASK_CLEANUP
    write_json(folder/'sample-cameras.json',info)
    for name in ('cameras.png','camera-map-base.png'):
        shutil.copy2(mask_source/name,folder/name)
    manifest=dict(fingerprint=fingerprint,revision=revision,source=source,mask_source=str(mask_source),selection_policy=policy,
        mask_version=info['render_version'],render_resolution=[1024,512],face_size=512,fov_degrees=100,
        candidate_count=len(info['cameras']),mask_cleanup=MASK_CLEANUP,created=time.time())
    write_json(manifest_path,manifest)
    write_json(folder/'selection.json',dict(selected_ids=['C000'],current=None,completed_ids=[],previews={}))
    return public_state(job,folder)


def work_folder(job):
    state=read(job/'job.json')
    stage=state['stages'].get('6')
    if not stage:raise ValueError('请先生成候选相机')
    revision=stage.get('revision','')
    if not re.fullmatch('[a-f0-9]{16}',revision):raise ValueError('Invalid scene revision')
    return job/'novel'/revision


def public_state(job,folder=None):
    folder=folder or work_folder(job);manifest=read(folder/'pipeline.json');selection=read(folder/'selection.json')
    info=read(folder/'sample-cameras.json')
    candidates=[{k:c[k] for k in ('id','grid','position_world_m','mask_stats','coincides_with_original')} for c in info['cameras']]
    policy=manifest.get('selection_policy','four-neighbor')
    excluded=list(dict.fromkeys(selection['selected_ids']+selection.get('failed_ids',[])))
    recommended,frontier=next_camera(candidates,excluded,policy)
    return dict(kind='cameras',revision=manifest['revision'],source=manifest['source'],cameras=candidates,
        count=len(candidates),selected_ids=selection['selected_ids'],current=selection['current'],
        highlighted=selection['current']['camera_id'] if selection['current'] else recommended,
        recommended=recommended,frontier=frontier,previews=selection['previews'],
        mask_version=manifest['mask_version'],mask_threshold=.15,connectivity=None if policy=='max-mask' else 4,
        selection_policy=policy,rgb_repair_model=read(job/'job.json').get('rgb_repair_model','seedream'),
        repair_running=read(job/'job.json').get('running_stage')==7,
        iteration_index=manifest.get('iteration_index',1),accepted_observations=manifest.get('accepted_observations',[]),
        inward_offset_m=info['inward_offset_m'],neighbor_count=info['neighbor_count'],
        face_size=512,fov_degrees=100,download=manifest['source']['file'],
        description=('同一最终高斯场景中的固定候选相机；在全部未选候选中选择 Mask 面积最大者。' if policy=='max-mask' else '同一最终高斯场景中的固定候选相机；全部已选点的四邻域中优先选择 Mask 面积最大者。')+'点击仅预览，不改变已选集合。')


def _png(data):
    if not isinstance(data,str) or len(data)>8*1024*1024 or not data.startswith('data:image/png;base64,'):
        raise ValueError('Expected a bounded PNG data URI')
    raw=base64.b64decode(data.split(',',1)[1],validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format!='PNG' or image.size!=(512,512):raise ValueError('Expected 512x512 Gaussian capture')
        return np.asarray(image.convert('RGB')).copy()


def capture(job,camera_id,payload,*,folder=None):
    folder=folder or work_folder(job);manifest=read(folder/'pipeline.json');info=read(folder/'sample-cameras.json')
    if not isinstance(payload,dict):raise ValueError('Expected capture object')
    if not CAM_ID.fullmatch(camera_id) or camera_id not in {c['id'] for c in info['cameras']}:
        raise ValueError('只能预览当前候选相机')
    if payload.get('revision')!=manifest['revision'] or payload.get('source_sha256')!=manifest['source']['sha256']:
        raise ValueError('高斯场景版本已变更，请重新进入候选步骤')
    if digest(job/manifest['source']['file'])!=manifest['source']['sha256']:
        raise ValueError('高斯文件已改变，旧候选场景不可继续截图')
    if len(payload.get('faces',[]))!=6:raise ValueError('需要六个高斯渲染面')
    if payload.get('capture_version')!=CAPTURE_VERSION or len(payload.get('alpha_faces',[]))!=6:
        raise ValueError('需要保留真实 alpha 的新版六面截图，请刷新页面重新渲染')
    frames=[_png(data) for data in payload['faces']]
    alpha_rgb=[_png(data) for data in payload['alpha_faces']]
    if any(not np.all(a==a[...,:1]) for a in alpha_rgb):raise ValueError('Alpha PNG must be grayscale')
    alphas=[a[...,0] for a in alpha_rgb]
    panorama=merge_faces(frames)
    mask_source=Path(manifest['mask_source'])
    c=next(c for c in info['cameras'] if c['id']==camera_id)
    with Image.open(mask_source/c['views']['mask']) as image:geometry_mask=np.asarray(image)>0
    if geometry_mask.shape!=panorama.shape[:2]:raise ValueError('Validated Mask dimensions disagree')
    raw_geometry=geometry_mask.copy();masks=repair_masks(raw_geometry,alphas)
    geometry_mask=masks['geometry'];alpha_mask=masks['alpha'];mask=masks['combined']
    capture_id=uuid.uuid4().hex[:16]
    relative=f'previews/{camera_id}/{capture_id}'
    output=folder/relative;output.mkdir(parents=True)
    Image.fromarray(panorama).save(output/'raw.png')
    Image.fromarray(mask.astype(np.uint8)*255).save(output/'mask.png')
    Image.fromarray(geometry_mask.astype(np.uint8)*255).save(output/'geometry-mask.png')
    Image.fromarray(alpha_mask.astype(np.uint8)*255).save(output/'alpha-mask.png')
    Image.fromarray(raw_geometry.astype(np.uint8)*255).save(output/'geometry-mask-raw.png')
    Image.fromarray(masks['raw_alpha'].astype(np.uint8)*255).save(output/'alpha-mask-raw.png')
    masked=panorama.copy();masked[mask]=MASK_COLOR;Image.fromarray(masked).save(output/'masked.png')
    faces=repair_faces(panorama,mask,alphas)
    for index,face in enumerate(faces):
        Image.fromarray(alphas[index]).save(output/f'{face["name"]}-render-alpha.png')
        Image.fromarray(frames[index]).save(output/f'{face["name"]}-render-rgb.png')
        Image.fromarray(face['rgb']).save(output/f'{face["name"]}.png')
        Image.fromarray(face['mask'].astype(np.uint8)*255).save(output/f'{face["name"]}-mask.png')
        view=face['rgb'].copy();view[face['mask']]=MASK_COLOR
        Image.fromarray(view).save(output/f'{face["name"]}-masked.png')
    result=dict(camera_id=camera_id,revision=manifest['revision'],path=relative,created=time.time(),
        source=manifest['source'],renderer='Spark: actual visible final Gaussian scene; six calibrated FOV100 renders',
        capture_version=CAPTURE_VERSION,mask_color=list(MASK_COLOR),
        mask_source=manifest['mask_version']+' UNION actual Spark render alpha < 0.70; open3, area16, Gaussian sigma1 and 50-percent contour on component masks, union and final faces',
        mask_cleanup=MASK_CLEANUP,raw_geometry_sha256=digest(output/'geometry-mask-raw.png'),
        before_cleanup_pixels=int((raw_geometry|masks['raw_alpha']).sum()),
        cleanup_removed_pixels=int(((raw_geometry|masks['raw_alpha'])&~mask).sum()),
        cleanup_added_pixels=int((mask&~(raw_geometry|masks['raw_alpha'])).sum()),
        shared_smoothing_pixels=masks['shared_smoothing_pixels'],
        alpha_threshold=ALPHA_THRESHOLD,alpha_threshold_u8_exclusive=ALPHA_THRESHOLD_U8,geometry_mask_pixels=int(geometry_mask.sum()),
        alpha_mask_pixels=int(alpha_mask.sum()),alpha_added_pixels=int((alpha_mask&~geometry_mask).sum()),
        alpha_faces={name:dict(sha256=digest(output/f'{name}-render-alpha.png'),
            transparent_pixels=int((a<ALPHA_THRESHOLD_U8).sum()),zero_pixels=int((a==0).sum()),
            quantiles=np.quantile(a,[0,.01,.5,.99,1]).tolist()) for name,a in zip(FACE_NAMES,alphas,strict=True)},
        mask_sha256=digest(output/'mask.png'),
        raw_sha256=digest(output/'raw.png'),mask_fraction=float(mask.mean()),size=[1024,512],faces=list(FACE_NAMES),fov_degrees=100)
    write_json(output/'capture.json',result)
    selection=read(folder/'selection.json');selection['previews'][camera_id]=result
    write_json(folder/'selection.json',selection)
    return result


def begin_repair(job):
    folder=work_folder(job);state=public_state(job,folder);selection=read(folder/'selection.json')
    if selection['current']:
        current=selection['current']
        if current['preview'].get('capture_version')!=CAPTURE_VERSION:
            fresh=selection['previews'].get(current['camera_id'])
            if not fresh or fresh.get('capture_version')!=CAPTURE_VERSION:
                raise ValueError('请重新渲染已确认相机，旧截图缺少真实 alpha')
            current['preview']=fresh;write_json(folder/'selection.json',selection)
        return current
    cid=state['recommended']
    if cid is None:raise ValueError('符合选择规则的候选已遍历完，没有下一台相机')
    preview=selection['previews'].get(cid)
    if not preview:raise ValueError('请先点击高亮候选相机，完成实际高斯全景渲染')
    selection['current']=dict(camera_id=cid,preview=preview,confirmed=time.time())
    # Confirmation is an attempt, not a trusted observation. Only an accepted
    # Gaussian patch can add this camera to selected_ids/completed_ids.
    write_json(folder/'selection.json',selection)
    return selection['current']


REPAIR_VERSION='single-image-1024-green-alpha070-open3-smooth1-v4'
REPAIR_INPUT_SIZE=(1024,1024)
REPAIR_PROMPT='''执行受约束的 3DGS 图像补洞修复。图中覆盖在缺损位置的纯亮绿色绿幕（RGB 0,255,0，#00FF00）是明确的待修补 Mask，包含未知空间遮挡及 alpha 透明、半透明缺口。绿幕是人工标记，不是场景里的物体、材质、灯光或反射。必须完整替换所有绿幕区域，只修补绿幕及其极小边缘衔接范围；不要根据黑色或白色自行扩大修补范围。真实黑色物体、深色材质和阴影必须保留。不得将绿幕颜色扩散到周围或在结果中残留绿边、绿色反光。
只修改确认存在的破洞及其极小边缘衔接范围，其他区域保持原始构图、机位、透视、空间结构、物品、材质、色彩、灯光、反射及装修或施工状态。不得把真实黑色物体、深色材质、阴影、正常开口、白墙或高光误判为破洞。
修复必须依据破洞四周真实可见的边缘连接、表面走向、颜色、材质、纹理、重复排列、透视、遮挡和物理连接进行连续性补全。与破洞边界直接相交的已有表面、结构、图案或物体，应按照原有方向、尺度和透视向内部延伸；若破洞跨越多个表面，应分别延伸并正确衔接。只有已有物体轮廓明确进入破洞时，才补全该物体缺失的部分；否则只延续周围占主导的背景表面。
禁止根据场景常识在破洞内新增独立物体、人物、家具、装饰、房间、门窗、镜面、开口或复杂图案。证据不足时采用最近邻的已确认表面、底色和结构进行最简单补全。不得把破洞的原始颜色、形状或外轮廓解释为需要保留的场景内容。
最终所有绿幕标记的破洞必须被完整填补，并与周围场景无缝融合；几何、透视、材质、光照和遮挡关系必须符合现实。不得残留绿幕、绿边、空洞、异常透明、模糊、团块、涂抹、融化、重影、断裂、重复或生成接缝。直接输出修复后的同一场景图像，不输出说明。'''


def repair_input(base,mask):
    # Upscale RGB before applying the binary mask, so hole colors do not bleed
    # into the context. This changes API input size, not the capture resolution.
    rgb=np.asarray(Image.fromarray(base).resize(REPAIR_INPUT_SIZE,Image.Resampling.LANCZOS)).copy()
    hole=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(REPAIR_INPUT_SIZE,Image.Resampling.NEAREST))>0
    rgb[hole]=MASK_COLOR
    return rgb


def repair(job):
    # Executed through the reference checkout's verified launcher; no key is
    # copied into this repository, logs, browser or persisted request payloads.
    sys.path.insert(0,str(REFERENCE))
    from click_gaussian.image_repair import RepairConfig,request_repair,fetch_image,encode_data_uri,minimum_seedream_output_size
    folder=work_folder(job);selection=read(folder/'selection.json');current=selection['current']
    if not current:raise ValueError('No confirmed next camera')
    preview=current['preview'];capture_dir=folder/preview['path']
    if preview.get('capture_version')!=CAPTURE_VERSION:raise ValueError('请重新渲染，当前旧截图缺少真实 alpha')
    if digest(capture_dir/'raw.png')!=preview['raw_sha256'] or digest(capture_dir/'mask.png')!=preview['mask_sha256']:
        raise ValueError('Capture changed after camera confirmation')
    out=capture_dir/'repair';out.mkdir(exist_ok=True)
    config=RepairConfig(prompt=REPAIR_PROMPT,output_size=minimum_seedream_output_size(*REPAIR_INPUT_SIZE))
    request_spec=dict(version=REPAIR_VERSION,model=config.model,prompt=REPAIR_PROMPT,
        image_count=1,input_size=list(REPAIR_INPUT_SIZE),output_size=config.output_size,
        alpha_threshold=ALPHA_THRESHOLD,mask_cleanup=MASK_CLEANUP)
    from face_quality_io import assess_rgb
    from face_quality import VERSION as QUALITY_VERSION,POLICY_HASH
    records=[];models=[];qualities=[]
    for name in FACE_NAMES:
        base=np.asarray(Image.open(capture_dir/f'{name}.png').convert('RGB'))
        fm=np.asarray(Image.open(capture_dir/f'{name}-mask.png'))>0
        fingerprint=hashlib.sha256(json.dumps(dict(request=request_spec,
            rgb_sha256=digest(capture_dir/f'{name}.png'),mask_sha256=digest(capture_dir/f'{name}-mask.png')),
            ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()
        receipt=out/f'{name}.json';result_path=out/f'{name}-repaired.png';model_path=out/f'{name}-model.png'
        cached=read(receipt) if receipt.exists() else {}
        same_request=cached.get('request_fingerprint')==fingerprint
        model_valid=model_path.exists() and (cached.get('model_sha256')==digest(model_path) or
            ('model_sha256' not in cached and result_path.exists() and cached.get('sha256')==digest(result_path)))
        if same_request and (not fm.any() or model_valid):
            record=cached
            generated=np.asarray(Image.open(model_path).convert('RGB').resize((base.shape[1],base.shape[0]),Image.Resampling.LANCZOS)) if fm.any() else base.copy()
            print(f'{name}: 复用模型返回图，按外羽化重新合成（不重复请求 API）',flush=True)
        else:
            started=time.perf_counter()
            if fm.any():
                damaged=repair_input(base,fm)
                Image.fromarray(damaged).save(out/f'{name}-api-input.png')
                print(f'Seedream 补洞 {name}，单张输入 1024x1024，Mask={fm.mean():.2%}',flush=True)
                marker=out/f'{name}-submitted.json'
                if marker.exists():raise RuntimeError('previous POST outcome uncertain; no automatic paid retry')
                write_json(marker,dict(request_fingerprint=fingerprint,started=time.time()))
                try:
                    urls,_=request_repair([encode_data_uri(damaged.astype(np.float32)/255)],config=config)
                    generated=fetch_image(urls[0])
                except Exception as exc:
                    # Never expose signed response URLs or request headers in tracebacks.
                    message=re.sub(r'https?://\S+','[remote URL]',str(exc))
                    raise RuntimeError(f'{name}: Seedream 请求或下载失败：{message[:700]}') from None
                h,w=generated.shape[:2]
                if abs(w/h-1)>.02:raise ValueError(f'{name}: 模型返回画幅不匹配')
                generated=np.rint(generated*255).clip(0,255).astype(np.uint8)
                Image.fromarray(generated).save(model_path)
                generated=np.asarray(Image.fromarray(generated).resize((base.shape[1],base.shape[0]),Image.Resampling.LANCZOS))
            else:generated=base.copy()
            record=dict(face=name,model=config.model,skipped=not bool(fm.any()),seconds=time.perf_counter()-started,
                request_fingerprint=fingerprint,request=request_spec,
                api_input_sha256=digest(out/f'{name}-api-input.png') if fm.any() else None,
                mask_pixels=int(fm.sum()))
        record['model_sha256']=digest(model_path) if fm.any() else None
        write_json(receipt,record)
        models.append(generated);records.append(record)
        qualities.append(assess_rgb(capture_dir,name,out))
    face_stats,stats=compose_faces(capture_dir,out,models,qualities)
    for record,values in zip(records,face_stats,strict=True):
        record.update(values,composite=COMPOSITE,sha256=digest(out/f'{record["face"]}-repaired.png'))
        write_json(out/f'{record["face"]}.json',record)
    result=dict(kind='repair',revision=preview['revision'],camera_id=current['camera_id'],capture=preview,
        quality_version=QUALITY_VERSION,quality_hash=POLICY_HASH,quality_faces=qualities,quality_phase='rgb_provisional',
        unresolved_mask=f'{preview["path"]}/repair/unresolved-mask.png',
        image=f'{preview["path"]}/repair/panorama.png',faces=records,composite=COMPOSITE,**stats,
        image_sha256=digest(out/'panorama.png'),external_provider='Volcengine Ark Seedream',request=request_spec,
        description='候选相机实际高斯全景 → FOV100 六面 Seedream 补洞 → 原始模型返回图重叠融合 → 向外 4 像素羽化。Mask 内完全使用新图，过渡带外逐字节保留原图；尚未回写高斯。')
    write_json(out/'repair.json',result)
    write_json(folder/'selection.json',selection)
    return result


def artifact(job,relative):
    parts=relative.split('/')
    if not parts or not re.fullmatch('[a-f0-9]{16}',parts[0]):raise FileNotFoundError('Invalid revision')
    allowed=(len(parts)==2 and parts[1] in ('cameras.png','camera-map-base.png','sample-cameras.json'))
    if len(parts) in (5,6) and parts[1]=='previews' and CAM_ID.fullmatch(parts[2]) and re.fullmatch('[a-f0-9]{16}',parts[3]):
        names={'raw.png','masked.png','mask.png','geometry-mask.png','alpha-mask.png','geometry-mask-raw.png','alpha-mask-raw.png','capture.json'}|{f'{f}{s}.png' for f in FACE_NAMES for s in ('','-mask','-masked','-render-alpha','-render-rgb')}
        if len(parts)==5:allowed=parts[-1] in names
        elif parts[4]=='repair':
            names={'panorama.png','repair.json','fused-before-composite.png','rgb-support.png','rgb-provenance.png','effective-mask.png','unresolved-mask.png'}
            names|={f'{f}{s}.png' for f in FACE_NAMES for s in ('-model','-repaired')}
            names|={f'{f}-rgb-{s}.png' for f in FACE_NAMES for s in ('anchors','metric_support','difference','bad')}
            names|={f'{f}-rgb-quality.json' for f in FACE_NAMES}
            allowed=parts[-1] in names
    if (len(parts)==7 and parts[1]=='previews' and CAM_ID.fullmatch(parts[2]) and
        re.fullmatch('[a-f0-9]{16}',parts[3]) and parts[4]=='depth' and re.fullmatch('[a-f0-9]{16}',parts[5])):
        names={'depths.npz','result.json','panorama-hole.png'}
        names|={'accepted-bundle.json','accepted-rgb.png','accepted-support.png','effective-mask.png','unresolved-mask.png','accepted-provenance.png','depth-conflict.png'}
        names|={f'{f}-depth-{s}.png' for f in FACE_NAMES for s in ('anchors','metric_support','difference','bad')}
        names|={f'{f}-{s}' for f in FACE_NAMES for s in ('depth-quality.json','accepted-valid.png')}
        names|={f'panorama-{stage}.{suffix}' for stage in ('rendered','input','completed') for suffix in ('npy','png')}
        names|={f'{face}-{suffix}' for face in FACE_NAMES for suffix in
            ('rendered-z.npy','completed-z.npy','input-z.npy','lingbot-z.npy','alpha.npy','camera.json','result.json',
             'rendered.png','completed.png','rgb.png','hole.png','model-valid.png')}
        allowed=parts[-1] in names
    if (len(parts) in (7,8,9) and parts[1]=='previews' and CAM_ID.fullmatch(parts[2]) and
        re.fullmatch('[a-f0-9]{16}',parts[3]) and parts[4]=='gaussians' and re.fullmatch('[a-f0-9]{16}',parts[5])):
        if len(parts)==7:
            names={'merged.ply','patch.ply','result.json','acceptance.json','render-acceptance.json'}
            names|={f'{kind}-panorama.png' for kind in ('base','patch','merged')}
            names|={f'{kind}-{face}{suffix}' for kind in ('base','patch','merged') for face in FACE_NAMES for suffix in ('.png','-z.npy','-alpha.npy')}
            allowed=parts[-1] in names
        elif parts[6] in FACE_NAMES:
            if len(parts)==8:
                allowed=parts[-1] in {'rgb.png','mask.png','support.png','extrapolated.png','depth-projected.npy','depth-input.npy','camera.json','projection.json',
                    'selected-rows.npy','selected-weights.npy','ring-labels.png','patch-camera.ply','patch-world.ply'}
            elif parts[7]=='sharp':
                allowed=parts[-1] in {'sharp_hard_acceptance.json','sharp_hard_acceptance_preview.png','sharp_hard_render_rgb.png',
                    'sharp_hard_render_depth.png','sharp_hard_render_depth.npy','sharp_hard_render_alpha.npy','sharp_hard_layer0_z.npy'}
    if not allowed:raise FileNotFoundError('Invalid novel-view artifact')
    base=(job/'novel').resolve();path=(base/relative).resolve()
    if not path.is_relative_to(base):raise FileNotFoundError('Invalid artifact path')
    return path
