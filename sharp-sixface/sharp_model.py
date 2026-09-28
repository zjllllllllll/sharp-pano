"""Original SHARP checkpoint and its native external-depth input."""
from pathlib import Path
import gc
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/ml-sharp'/'src'))
from sharp.models import create_predictor, PredictorParams
from gaussian_ops import digest


def load_model():
    checkpoint=ROOT/'models'/'sharp_2572gikvuh.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    # Loading the public state is strict, including the learned alignment UNet.
    try:
        with torch.device('meta'): model=create_predictor(PredictorParams())
        model.load_state_dict(state,strict=True,assign=True)
    except (NotImplementedError,RuntimeError,TypeError):
        model=create_predictor(PredictorParams())
        model.load_state_dict(state,strict=True)
    info={'checkpoint':str(checkpoint),'sha256':digest(checkpoint),'strict_keys':len(state),
          'source':str(ROOT/'third_party/ml-sharp'/'src'),'precision':'fp32','torch':torch.__version__,
          'alignment_keys':sum(k.startswith('depth_alignment.') for k in state)}
    del state; gc.collect()
    if info['alignment_keys']==0: raise RuntimeError('Native depth-alignment weights missing')
    return model.eval().cuda(),info


def cpu_scene(g):
    names=('means','scales','quats','colors','alpha')
    result={name:t.detach().float().cpu().numpy()[0].copy() for name,t in zip(names,g)}
    for k,v in result.items():
        if not np.isfinite(v).all(): raise RuntimeError(f'Nonfinite SHARP {k}')
    return result


@torch.inference_mode()
def infer_shallow(model, rgb, depth_z, cam):
    n=cam['size_wh'][0]
    if n!=1536 or rgb.shape!=(n,n,3) or depth_z.shape!=(n,n):
        raise ValueError('Use the original SHARP 1536-square internal grid without resizing')
    image=torch.from_numpy(rgb.copy()).cuda().permute(2,0,1)[None].float()/255
    target=torch.from_numpy(depth_z.copy()).cuda()[None,None]
    disparity=torch.tensor([cam['K_pixel_edge'][0][0]/n],device='cuda',dtype=torch.float32)
    try:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
        gaussian=model(image,disparity,depth=target)
        torch.cuda.synchronize()
        info={'model_seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
              'api':'RGBGaussianPredictor(image, disparity_factor, depth=target_camera_z)',
              'grid_hw':[768,768],'layers':2}
        scene=cpu_scene(gaussian)
        del gaussian
        return scene,info
    finally:
        del image,target,disparity
        gc.collect(); torch.cuda.empty_cache()


@torch.inference_mode()
def infer_panorama_depth(model, rgb, depth_z, cam):
    """Initialize both layers from supplied depth; keep SHARP final predictions."""
    image=torch.from_numpy(rgb.copy()).cuda().permute(2,0,1)[None].float()/255
    target=torch.from_numpy(depth_z.copy()).cuda()[None,None]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    # Use pretrained image encodings, but not its predicted depth or alignment.
    encoded=model.monodepth_model(image)
    injected=target.expand(-1,encoded.disparity.shape[1],-1,-1)
    initial=model.init_model(image,injected)
    features=model.feature_model(initial.feature_input,encodings=encoded.output_features)
    delta=model.prediction_head(features)
    gaussian=model.gaussian_composer(delta=delta,base_values=initial.gaussian_base_values,
                                    global_scale=initial.global_scale)
    scene=cpu_scene(gaussian)
    torch.cuda.synchronize()
    info=dict(model_seconds=time.perf_counter()-start,
              peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
              api='external depth initializer -> unmodified SHARP final parameters',
              grid_hw=[768,768],layers=2,learned_alignment=False,final_reprojection=False)
    del image,target,encoded,injected,initial,features,delta,gaussian
    gc.collect();torch.cuda.empty_cache()
    return scene,info
