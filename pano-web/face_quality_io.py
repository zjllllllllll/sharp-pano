"""Local inputs, provenance checks and inspection images for quality gates."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from face_quality import evaluate_rgb,evaluate_depth,record,POLICY_HASH,VERSION
from novel_geometry import FACE_NAMES,camera,view_rays,sample_erp


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def image(path,mode=None):
    with Image.open(path) as im:return np.asarray(im.convert(mode) if mode else im).copy()


def geometry_face(capture,name):
    raw=image(capture/'geometry-mask-raw.png')>0
    _,rays=view_rays(camera(FACE_NAMES.index(name),size=512,fov=100))
    return sample_erp(raw.astype(np.float32),rays)>0


def inputs(capture,name):
    capture=Path(capture)
    return (image(capture/f'{name}-mask.png')>0,geometry_face(capture,name),
            image(capture/f'{name}-render-alpha.png'))


def save_maps(out,prefix,maps):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);files={}
    for name,array in maps.items():
        if name=='difference':
            t=np.clip(array/.20,0,1)
            array=np.stack([255*t,160*t*(1-t),255*(1-t)],axis=-1).astype(np.uint8)
            array[~maps['anchors']]=[30,34,38]
        else:array=np.asarray(array,np.uint8)*255
        filename=f'{prefix}-{name}.png';Image.fromarray(array).save(out/filename);files[name]=filename
    return files


def assess_rgb(capture,name,out=None):
    capture=Path(capture);model=capture/'repair'/f'{name}-model.png'
    paths=[capture/f'{name}.png',capture/f'{name}-mask.png',capture/'geometry-mask-raw.png',
           capture/f'{name}-render-alpha.png']
    try:
        mask,geometry,alpha=inputs(capture,name)
        if not mask.any():r,maps=record('rgb','not_needed'),{}
        else:
            paths.append(model)
            with Image.open(model) as im:
                if im.width!=im.height or im.width<512:raise ValueError('Invalid RGB shape')
                generated=np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.LANCZOS))
            r,maps=evaluate_rgb(image(paths[0],'RGB'),generated,mask,geometry,alpha)
        r['input_hashes']={str(p.relative_to(capture)):digest(p) for p in paths}
    except FileNotFoundError:r,maps=record('rgb','rejected',['missing_input']),{}
    except (ValueError,OSError):r,maps=record('rgb','rejected',['invalid_input']),{}
    r.update(face=name)
    if out is not None:
        r['maps']=save_maps(out,name+'-rgb',maps);write(Path(out)/f'{name}-rgb-quality.json',r)
    return r


def assess_depth(capture,depth,name,out=None,position=None):
    capture,depth=Path(capture),Path(depth)
    paths={s:depth/f'{name}-{s}' for s in ('rendered-z.npy','lingbot-z.npy','model-valid.png','alpha.npy','camera.json','hole.png')}
    try:
        mask,geometry,_=inputs(capture,name)
        source=np.load(paths['rendered-z.npy'],allow_pickle=False)
        # Depth also masks invalid source rays; respect the actual model input.
        mask |= image(paths['hole.png'])>0
        cam=read(paths['camera.json']);expected=camera(FACE_NAMES.index(name),size=512,fov=100)
        pose=np.asarray(cam['c2w'])
        if (cam.get('semantics')!='camera_z_m' or not np.allclose(cam['intrinsic_K_opencv'],expected['K_pixel_edge'])
            or not np.allclose(pose[:3,:3],np.asarray(expected['c2w'])[:3,:3])
            or (position is not None and not np.allclose(pose[:3,3],position))):raise ValueError('Camera provenance mismatch')
        if not mask.any():r,maps=record('depth','not_needed'),{}
        else:
            r,maps=evaluate_depth(source,np.load(paths['lingbot-z.npy'],allow_pickle=False),
                image(paths['model-valid.png'])>0,mask,geometry,np.load(paths['alpha.npy'],allow_pickle=False))
        r['input_hashes']={p.name:digest(p) for p in paths.values() if p.exists()}
        r['input_hashes'].update({str(p.relative_to(capture)):digest(p) for p in
            (capture/f'{name}-mask.png',capture/'geometry-mask-raw.png')})
    except FileNotFoundError:r,maps=record('depth','rejected',['missing_input']),{}
    except (ValueError,KeyError,OSError):r,maps=record('depth','rejected',['invalid_input']),{}
    r.update(face=name)
    if out is not None:
        r['maps']=save_maps(out,name+'-depth',maps);write(Path(out)/f'{name}-depth-quality.json',r)
    return r


def verify_report(capture,report):
    if report.get('policy_hash')!=POLICY_HASH:raise ValueError('Quality policy changed; reassess RGB first')
    for name,expected in report.get('input_hashes',{}).items():
        p=(Path(capture)/name).resolve()
        if not p.is_relative_to(Path(capture).resolve()) or digest(p)!=expected:raise ValueError('Quality input changed')
    if not report.get('input_hashes'):raise ValueError('Missing quality provenance')


def assessment_revision(rows):
    return hashlib.sha256(json.dumps(dict(policy=POLICY_HASH,rows=rows),sort_keys=True).encode()).hexdigest()[:16]
