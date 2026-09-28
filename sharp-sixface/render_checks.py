"""Actual gsplat renders: source, seams, translated and close views of shallow-depth outputs.

Run with the existing verified goris CUDA launcher. No environment installation.
Source-view RGB/depth metrics are input-consistency checks, not ground truth.
"""
from pathlib import Path
import argparse
import gc
import json
import math
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
import torch
from gsplat.rendering import rasterization
from geometry import camera,view_rays,sample_erp
from gaussian_ops import summary,digest

ROOT=Path(__file__).resolve().parents[1]


def gpu_scene(path):
    v=PlyData.read(path,mmap='r')['vertex'].data
    def columns(names):
        return torch.from_numpy(np.column_stack([v[k] for k in names]).copy()).cuda().float()
    means=columns(['x','y','z']); scales=columns(['scale_0','scale_1','scale_2']).exp()
    quats=columns(['rot_0','rot_1','rot_2','rot_3'])
    alpha=columns(['opacity'])[:,0].sigmoid()
    srgb=columns(['f_dc_0','f_dc_1','f_dc_2'])*math.sqrt(1/(4*math.pi))+.5
    linear=torch.where(srgb<=.04045,srgb/12.92,((srgb+.055).clamp_min(0)/1.055).pow(2.4))
    return dict(means=means,quats=quats,scales=scales,opacities=alpha,colors=linear)


def look_rotation(forward):
    f=np.asarray(forward,dtype=np.float64); f/=np.linalg.norm(f)
    r=np.cross([0,1,0],f); r/=np.linalg.norm(r)
    d=np.cross(f,r)
    return np.stack([r,d,f],axis=1)


def views(radial,size):
    result=[]
    for i in range(6):
        c=camera(i,size=size)
        result.append(dict(name=c['face'],c2w=np.asarray(c['c2w']),fov=100.,reference=True,group='origin'))
    for name,direction in [('seam_front_right',(1,0,1)),('seam_front_up',(0,-1,1))]:
        c=np.eye(4); c[:3,:3]=look_rotation(direction)
        result.append(dict(name=name,c2w=c,fov=90.,reference=True,group='seam'))
    nearest=float(np.quantile(radial,.01))
    side=np.eye(4); side[0,3]=min(.20,nearest*.2)
    result.append(dict(name='translated_right',c2w=side,fov=90.,reference=False,group='novel'))
    center_range=float(sample_erp(radial,np.array([[0.,0.,1.]]))[0])
    close=np.eye(4); close[2,3]=.65*center_range
    result.append(dict(name='close_front_surface',c2w=close,fov=70.,reference=False,group='novel'))
    return result


def target_at_view(rgb,radial,view,size):
    f=size/(2*math.tan(math.radians(view['fov']/2)))
    a=(np.arange(size)+.5-size/2)/f
    x,y=np.meshgrid(a,a)
    local=np.stack([x,y,np.ones_like(x)],axis=-1)
    local/=np.linalg.norm(local,axis=-1,keepdims=True)
    directions=local@view['c2w'][:3,:3].T
    image=np.clip(np.rint(sample_erp(rgb,directions)),0,255).astype(np.uint8)
    depth=sample_erp(radial,directions)*local[...,2]
    return image,depth


@torch.inference_mode()
def render(scene,view,size):
    f=size/(2*math.tan(math.radians(view['fov']/2)))
    k=torch.tensor([[[f,0,size/2],[0,f,size/2],[0,0,1]]],device='cuda',dtype=torch.float32)
    w2c=torch.from_numpy(np.linalg.inv(view['c2w']).astype(np.float32))[None].cuda()
    color,alpha,_=rasterization(**scene,viewmats=w2c,Ks=k,width=size,height=size,
                               near_plane=.01,far_plane=1e4,packed=False,render_mode='RGB+ED',rasterize_mode='classic')
    if not torch.isfinite(color).all() or not torch.isfinite(alpha).all(): raise RuntimeError('Nonfinite gsplat render')
    rgb=color[0,...,:3]
    srgb=torch.where(rgb<=.0031308,rgb*12.92,1.055*rgb.clamp_min(.0031308).pow(1/2.4)-.055)
    image=(srgb.clamp(0,1)*255).round().byte().cpu().numpy()
    return image,color[0,...,3].cpu().numpy(),alpha[0,...,0].cpu().numpy()


def save_comparison_sheets(directory,stem,report):
    size=report['face_size']
    shots=[{'name':name,**value} for name,value in report['views'].items()]
    for group in ('origin','seam','novel'):
        selected=[s for s in shots if s['group']==group]
        canvas=Image.new('RGB',(2*size,len(selected)*(size+28)), '#181b20')
        draw=ImageDraw.Draw(canvas)
        for row,shot in enumerate(selected):
            for col,mode in enumerate(('input','shallow')):
                x=col*size; y=row*(size+28)
                draw.text((x+8,y+7),f"{stem} / {shot['name']} / {mode}",fill='white')
                source=directory/f"{shot['name']}.{mode}.png"
                if source.exists():
                    with Image.open(source) as im: canvas.paste(im,(x,y+28))
                else: draw.text((x+15,y+70),'Novel viewpoint: no ground-truth RGB',fill='#aab3bd')
        canvas.save(directory/f'{group}_comparison.jpg',quality=94)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',nargs='+',default=['0.1','1.1','2.1','3.1','4.1'])
    p.add_argument('--results',type=Path,default=ROOT/'sharp-sixface-results')
    p.add_argument('--size',type=int,default=512)
    args=p.parse_args(); size=args.size
    torch.set_num_threads(4)
    for stem in args.images:
        directory=args.results/'validation'/stem
        directory.mkdir(parents=True,exist_ok=True)
        report_path=directory/'renders.json'
        if report_path.exists():
            old=json.loads(report_path.read_text())
            if all(digest(args.results/m/f'{stem}.ply')==old['ply_sha256'][m] for m in ('shallow',)):
                if size != old['face_size']: raise ValueError('Existing render size differs')
                save_comparison_sheets(directory,stem,old)
                print('RENDER_ALREADY_VERIFIED',stem,flush=True); continue
            raise RuntimeError('Outputs changed after rendering')
        rgb=np.asarray(Image.open(ROOT/'pano-tests'/f'{stem}.png').convert('RGB'))
        radial=np.load(ROOT/'depthmaster-results'/f'{stem}.range_m.npy')
        shots=views(radial,size)
        report={'scene':stem,'face_size':size,'ply_sha256':{},'views':{},
                'metrics':'origin/seam consistency with supplied RGB and depth, NOT geometric ground truth',
                'renderer':'gsplat 1.5.3 RGB+ED classic; linear RGB then sRGB; black background'}
        targets={}
        for shot in shots:
            if shot['reference']:
                image,depth=target_at_view(rgb,radial,shot,size)
                targets[shot['name']]=(image,depth)
                Image.fromarray(image).save(directory/f"{shot['name']}.input.png")
            report['views'][shot['name']]={'c2w':shot['c2w'].tolist(),'w2c':np.linalg.inv(shot['c2w']).tolist(),
                                         'fov':shot['fov'],'group':shot['group'],'has_reference':shot['reference']}
        for mode in ('shallow',):
            path=args.results/mode/f'{stem}.ply'
            report['ply_sha256'][mode]=digest(path)
            scene=gpu_scene(path)
            for shot in shots:
                name=shot['name']; image,depth,alpha=render(scene,shot,size)
                Image.fromarray(image).save(directory/f'{name}.{mode}.png')
                np.save(directory/f'{name}.{mode}.depth.npy',depth)
                data={'finite':True,'alpha_mean':float(alpha.mean()),'coverage_alpha_gt_half':float((alpha>.5).mean())}
                if shot['reference']:
                    reference,target=targets[name]
                    valid=(alpha>.5)&(depth>0)&np.isfinite(depth)
                    data['radial_or_z_relative_error']=summary(np.abs(depth[valid]-target[valid])/target[valid])
                    data['camera_z_abs_error_m']=summary(np.abs(depth[valid]-target[valid]))
                    mse=np.mean(((image.astype(np.float32)-reference)/255)**2)
                    data['rgb_psnr_db']=float(-10*np.log10(max(float(mse),1e-12)))
                report['views'][name][mode]=data
                print('RENDER_OK',stem,mode,name,'coverage',round(data['coverage_alpha_gt_half'],5),flush=True)
            del scene; gc.collect(); torch.cuda.empty_cache()
        save_comparison_sheets(directory,stem,report)
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('SCENE_RENDER_VERIFIED',stem,flush=True)


if __name__=='__main__': main()
