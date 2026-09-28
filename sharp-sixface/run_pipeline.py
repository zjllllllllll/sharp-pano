"""Five panoramas, native SHARP depth input, calibrated six-view fusion."""
from pathlib import Path
import argparse
import gc
import json
import os
import time

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','garbage_collection_threshold:0.75')
import numpy as np
from PIL import Image
from plyfile import PlyData

from geometry import camera, split_face, sample_erp
from gaussian_ops import (digest,summary,ndc_to_camera,world_transform,
                          to_vertices,save_vertices,DTYPE)

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'sharp-sixface-results'


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def prepare(stem,output):
    src_rgb=ROOT/'pano-tests'/f'{stem}.png'
    src_depth=ROOT/'depthmaster-results'/f'{stem}.range_m.npy'
    sources={'rgb':str(src_rgb),'rgb_sha256':digest(src_rgb),
             'radial_depth':str(src_depth),'depth_sha256':digest(src_depth)}
    directory=output/'inputs'/stem
    manifest=directory/'cameras.json'
    if manifest.exists():
        record=json.loads(manifest.read_text(encoding='utf-8'))
        if record['sources']!=sources: raise RuntimeError('Source image/depth changed')
        for cam in record['cameras']:
            for key in ['rgb','camera_z','range']:
                if digest(Path(cam['files'][key]))!=cam['hashes'][key]:
                    raise RuntimeError(f'Prepared input changed: {key}')
        return record
    directory.mkdir(parents=True,exist_ok=True)
    rgb=np.asarray(Image.open(src_rgb).convert('RGB'))
    radial=np.load(src_depth)
    if rgb.shape[:2]!=radial.shape or rgb.shape[1]!=rgb.shape[0]*2:
        raise ValueError('Expected registered full ERP RGB/range')
    if not np.all(np.isfinite(radial)&(radial>0)):
        raise ValueError('Input range must be dense, positive and finite')
    cams=[]
    for i in range(6):
        cam=camera(i); name=cam['face']
        paths={k:directory/f'{name}.{suffix}' for k,suffix in
               [('rgb','rgb.png'),('camera_z','camera_z.npy'),('range','range.npy')]}
        if any(p.exists() for p in paths.values()):
            raise FileExistsError('Incomplete prepared input: inspect before replacing')
        image,z,ranges=split_face(rgb,radial,cam)
        Image.fromarray(image).save(paths['rgb'])
        np.save(paths['camera_z'],z); np.save(paths['range'],ranges)
        cam['files']={k:str(v) for k,v in paths.items()}
        cam['hashes']={k:digest(v) for k,v in paths.items()}
        cam['camera_z_quantiles_m']=summary(z)
        write_json(directory/f'{name}.camera.json',cam)
        cams.append(cam)
        print('FACE_PREPARED',stem,name,flush=True)
        del image,z,ranges; gc.collect()
    record={'sources':sources,'source_size_wh':[rgb.shape[1],rgb.shape[0]],
            'depth_convention':'radial distance -> perspective camera-Z (not colored depth PNG)',
            'cameras':cams,'known_relative_poses':True,'all_camera_centers':[0,0,0]}
    write_json(manifest,record)
    return record


def depth_diagnostics(scene,radial,cam,layer_size):
    means=scene['means'][:layer_size].astype(np.float64)
    world=means@np.asarray(cam['c2w'])[:3,:3].T
    radius=np.linalg.norm(world,axis=-1)
    target=sample_erp(radial,world)
    absolute=np.abs(radius-target)
    return {'scope':'layer0 centers sampled at final learned ray; before overlap feather',
            'radial_abs_error_m':summary(absolute),
            'radial_relative_error':summary(absolute/target),
            'fraction_rel_error_below_1pct':float(np.mean(absolute/target<.01)),
            'camera_z_positive_fraction':float(np.mean(means[:,2]>0))}


def merge_scene(stem,output,cams):
    mode='shallow'
    target=output/mode/f'{stem}.ply'
    report_path=output/'reports'/stem/f'{mode}.merged.json'
    if target.exists():
        report=json.loads(report_path.read_text(encoding='utf-8'))
        if digest(target)!=report['ply']['sha256']: raise RuntimeError('Merged PLY changed')
        return report
    paths=[output/'work'/stem/f"{cam['face']}.{mode}.ply" for cam in cams]
    parts=[PlyData.read(p,mmap='r')['vertex'].data for p in paths]
    counts=[len(x) for x in parts]
    vertices=np.empty(sum(counts),dtype=DTYPE)
    offset=0
    for part in parts:
        vertices[offset:offset+len(part)]=part
        offset+=len(part)
    target.parent.mkdir(parents=True,exist_ok=True)
    info=save_vertices(target,vertices)
    del parts,vertices; gc.collect()
    report={'mode':mode,'ply':info,'face_counts':dict(zip([c['face'] for c in cams],counts)),
            'merge':'exact calibrated c2w on means AND covariance rotations; normalized angular optical-thickness feather',
            'pose_fit':'none; exact common camera origin','icp':False,
            'cameras':str(output/'inputs'/stem/'cameras.json'),
            'layer_policy':'both predicted layers',
            'face_reports':[str(output/'reports'/stem/f"{c['face']}.{mode}.json") for c in cams]}
    write_json(report_path,report)
    print('MERGED_VERIFIED',mode,stem,info['gaussians'],str(target),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',nargs='+',default=['0.1','1.1','2.1','3.1','4.1'])
    p.add_argument('--faces',nargs='+',default=None)
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args(); output=args.output_dir.resolve()
    if any(x not in ['0.1','1.1','2.1','3.1','4.1'] for x in args.images): raise ValueError('Unexpected scene')
    output.mkdir(parents=True,exist_ok=True)
    records={s:prepare(s,output) for s in args.images}
    if args.prepare_only: return
    from sharp_model import load_model,infer_shallow
    import torch
    torch.set_num_threads(8); torch.manual_seed(42)
    model,info=load_model()
    write_json(output/'reports'/'model.json',info)
    print('SHARP_STRICT_LOAD_OK',json.dumps(info),flush=True)
    for stem,record in records.items():
        radial=np.load(record['sources']['radial_depth'])
        for index,cam in enumerate(record['cameras']):
            face=cam['face']
            if args.faces and face not in args.faces: continue
            mode='shallow'
            dest=output/'work'/stem/f'{face}.{mode}.ply'
            report_path=output/'reports'/stem/f'{face}.{mode}.json'
            if dest.exists():
                report=json.loads(report_path.read_text(encoding='utf-8'))
                if digest(dest)!=report['ply']['sha256']: raise RuntimeError('Face PLY changed')
                continue
            rgb=np.asarray(Image.open(cam['files']['rgb']).convert('RGB'))
            depth=np.load(cam['files']['camera_z'])
            print('FACE_INFERENCE_START',stem,face,mode,flush=True)
            scene,model_info=infer_shallow(model,rgb,depth,cam)
            started=time.perf_counter()
            layer_size=int(np.prod(model_info['grid_hw']))
            scene,unproject=ndc_to_camera(scene,cam)
            diagnostics=depth_diagnostics(scene,radial,cam,layer_size)
            scene=world_transform(scene,cam)
            vertices,merge_info=to_vertices(scene,index,cam['hfov_deg'])
            dest.parent.mkdir(parents=True,exist_ok=True)
            ply=save_vertices(dest,vertices)
            report={'scene':stem,'face':face,'mode':mode,'model':model_info,'unprojection':unproject,
                    'depth_diagnostics':diagnostics,'overlap':merge_info,'ply':ply,
                    'postprocess_seconds':time.perf_counter()-started}
            write_json(report_path,report)
            print('FACE_VERIFIED',stem,face,mode,'gaussians',len(vertices),
                  'depth_rel_med',diagnostics['radial_relative_error']['median'],flush=True)
            del scene,vertices; gc.collect()
            del rgb,depth; gc.collect(); torch.cuda.empty_cache()
        if all((output/'work'/stem/f"{c['face']}.shallow.ply").exists() for c in record['cameras']):
            merge_scene(stem,output,record['cameras'])
        for key in ('rgb','radial_depth'):
            hashkey='rgb_sha256' if key=='rgb' else 'depth_sha256'
            if digest(Path(record['sources'][key]))!=record['sources'][hashkey]: raise RuntimeError('Input changed')
        del radial; gc.collect()
    print('REQUESTED_BATCH_FINISHED',args.images,flush=True)


if __name__=='__main__': main()
