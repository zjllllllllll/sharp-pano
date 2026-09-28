"""Stage 8: render the frozen Gaussian scene at the confirmed camera, then LingBot.

Run only through the project-root run_cuda_python.cmd.
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
from novel_geometry import six_faces,FACE_NAMES,camera,MASK_COLOR
from novel_depth_geometry import merge_depth_faces,fuse_face

from project_paths import REPAIR as REFERENCE, CUDA_LAUNCHER as LAUNCHER, LINGBOT, MODELS
CHECKPOINT=MODELS/'lingbot-depth.pt'
VERSION='lingbot-gaussian-depth-direct-panorama-v3'


def write_depth_outputs(out,raw,completed,lingbot,mask,position,accepted_fusion=None):
    """Project model returns directly; original depth/Mask are comparison data only.

    The per-face calibrated composites remain available for inspection, but are
    never inputs to the completed panorama. Uncovered model pixels remain zero.
    """
    mask=np.asarray(mask,bool)
    height,width=mask.shape
    raw_pano,raw_valid=merge_depth_faces(raw,width,height)
    if accepted_fusion is None:final,valid=merge_depth_faces(lingbot,width,height)
    else:final,valid=accepted_fusion['depth'],accepted_fusion['support']
    if accepted_fusion is None and not valid.any():raise ValueError('LingBot faces contain no valid panorama depth')
    hole=mask|~raw_valid
    observed=raw_pano.copy();observed[hole]=0
    arrays=[('panorama-rendered',raw_pano),('panorama-input',observed),('panorama-completed',final)]
    reference=final[valid] if valid.any() else raw_pano[raw_valid]
    lo,hi=[float(v) for v in np.quantile(reference,[.01,.99])] if reference.size else (0.,1.)
    for name,array in arrays:
        np.save(out/f'{name}.npy',array)
        Image.fromarray(preview_depth(array,lo,hi)).save(out/f'{name}.png')
    Image.fromarray(hole.astype(np.uint8)*255).save(out/'panorama-hole.png')
    for i,name in enumerate(FACE_NAMES):
        for suffix,depth in [('rendered',raw[i]),('completed',completed[i])]:
            Image.fromarray(preview_depth(depth,lo,hi)).save(out/f'{name}-{suffix}.png')
    np.savez_compressed(out/'depths.npz',rendered_radial_m=raw_pano,input_radial_m=observed,completed_radial_m=final,
        original_mask=mask,depth_hole_mask=hole,observed_valid=raw_valid,completed_valid=valid,
        face_rendered_z_m=np.stack(raw),face_completed_z_m=np.stack(completed),face_lingbot_z_m=np.stack(lingbot),
        position_world_m=position,face_names=np.asarray(FACE_NAMES))
    return dict(panorama_source='lingbot_raw_faces',panorama_compositing='none',preview_range_m=[lo,hi],
        original_mask_pixels=int(mask.sum()),depth_hole_pixels=int(hole.sum()),
        additional_invalid_panorama_pixels=int((hole&~mask).sum()),
        outside_changed_pixels=int(np.count_nonzero(raw_pano[~hole]!=final[~hole])),
        invalid_completed_pixels=int((~valid).sum()),invalid_completed_hole_pixels=int((hole&~valid).sum()),
        panorama_valid_coverage=float(valid.mean()),hole_coverage=float(valid[hole].mean()) if hole.any() else 1.)


def depth_warnings(metrics,records):
    warnings=[]
    if metrics['invalid_completed_pixels']:
        warnings.append(f"LingBot 原始返回融合后仍有 {metrics['invalid_completed_pixels']} 个全景像素无有效深度"
                        '（NPY 为 0，预览为绿）；保留模型结果，不用原图补回。')
    warnings.extend(f"六面对照合成 {r['face']}：{r['extrapolated_hole_pixels']} 个模型无效像素使用参考算法外推（不用于全景）"
                    for r in records if r.get('extrapolated_hole_pixels',0))
    warnings.extend(f"六面对照合成 {r['face']}：局部校准不可靠，使用未缩放的模型深度制作对照；全景仍直接使用 LingBot 原始返回"
                    for r in records if r.get('comparison_only_uncalibrated'))
    return warnings


DEPTH_DESCRIPTION='同一高斯、同一新相机渲染深度 → 六面 LingBot RGB-D 原始返回 → 直接投影融合为全景径向深度。全景不按 Mask 拼回原深度；六面局部合成仅作对照。'


def render_faces(ply,position,out):
    import torch
    if sys.platform=='win32':
        # Background launchers may use the Windows default console code page
        # rather than UTF-8. Localized MSVC emits that encoding.
        # Correct decoding in this worker only; leave installed packages/cache intact.
        import ctypes
        import torch.utils.cpp_extension as cpp_extension
        codepage=ctypes.windll.kernel32.GetConsoleOutputCP() or ctypes.windll.kernel32.GetACP()
        cpp_extension.SUBPROCESS_DECODE_ARGS=(f'cp{codepage}',)
        print(f'MSVC diagnostics code page: {codepage}',flush=True)
    from gsplat.cuda._backend import _C
    if _C is None:raise RuntimeError('gsplat CUDA backend unavailable; use the reference launcher')
    from gsplat import rasterization
    from plyfile import PlyData
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for Gaussian depth rendering')
    vertices=PlyData.read(str(ply),mmap='r')['vertex'].data
    def columns(names):return torch.from_numpy(np.column_stack([vertices[n] for n in names]).astype(np.float32)).cuda()
    means=columns(['x','y','z']);quats=columns([f'rot_{i}' for i in range(4)])
    scales=columns([f'scale_{i}' for i in range(3)]).exp()
    opacity=torch.from_numpy(np.asarray(vertices['opacity']).astype(np.float32).copy()).cuda().sigmoid()
    colors=torch.zeros((len(vertices),3),dtype=torch.float32,device='cuda')
    count=len(vertices);del vertices
    results=[]
    try:
        with torch.inference_mode():
            for i,name in enumerate(FACE_NAMES):
                print(f'渲染 {name} 相机 Z 深度：{count:,} 个原始高斯',flush=True)
                cam=camera(i,size=512,fov=100);c2w=np.asarray(cam['c2w']);c2w[:3,3]=position
                w2c=np.linalg.inv(c2w)
                # Same depth definition as the reference GSplatRenderer:
                # accumulated camera-Z divided by accumulated alpha.
                outputs=rasterization(means=means,quats=quats,scales=scales,opacities=opacity,colors=colors,
                    viewmats=torch.tensor(w2c,dtype=torch.float32,device='cuda')[None],
                    Ks=torch.tensor(cam['K_pixel_edge'],dtype=torch.float32,device='cuda')[None],
                    width=512,height=512,render_mode='D',rasterize_mode='classic',packed=False,eps2d=0.)
                pixels,alpha=outputs[:2]
                z=(pixels[0,...,0]/alpha[0,...,0].clamp_min(1e-8)).cpu().numpy().astype(np.float32)
                a=alpha[0,...,0].cpu().numpy().astype(np.float32)
                z[~np.isfinite(z)|(z<=1e-3)|(a<=1e-6)]=0
                np.save(out/f'{name}-rendered-z.npy',z);np.save(out/f'{name}-alpha.npy',a)
                write_json(out/f'{name}-camera.json',dict(c2w=c2w.tolist(),w2c=w2c.tolist(),
                    intrinsic_K_opencv=cam['K_pixel_edge'],image=dict(width=512,height=512),fov=100,semantics='camera_z_m'))
                results.append(z);del outputs,pixels,alpha
    finally:
        del means,quats,scales,opacity,colors
        gc.collect();torch.cuda.empty_cache()
    return results


def preview_depth(depth,lo,hi):
    import cv2
    valid=np.isfinite(depth)&(depth>1e-3)
    scaled=np.clip((np.where(valid,depth,lo)-lo)/max(hi-lo,1e-6),0,1)
    u8=np.rint(scaled*255).astype(np.uint8)
    color=cv2.cvtColor(cv2.applyColorMap(u8,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB);color[~valid]=MASK_COLOR
    return color


def complete_depth(job):
    from quality_depth import complete_depth as gated_complete_depth
    return gated_complete_depth(job)
