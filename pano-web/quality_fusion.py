"""Sparse six-face fusion: rejected/invalid samples have exactly zero weight."""
import numpy as np
from scipy.ndimage import map_coordinates
from novel_geometry import FACE_NAMES,erp_directions,camera,angular_weights,composite_outward,MASK_COLOR


def project_face(rays,index,size=512):
    cam=camera(index,size=size,fov=100);local=rays@np.asarray(cam['c2w'])[:3,:3]
    z=local[...,2];f=cam['K_pixel_edge'][0][0]
    x=local[...,0]/np.maximum(z,1e-8)*f+size/2-.5
    y=local[...,1]/np.maximum(z,1e-8)*f+size/2-.5
    inside=(z>0)&(x>=0)&(x<=size-1)&(y>=0)&(y<=size-1)
    return x,y,z,inside


def sample_face(source,x,y):
    return map_coordinates(source,[y,x],order=1,mode='nearest',prefilter=False)


def valid_weight(rays,index,valid):
    x,y,z,inside=project_face(rays,index,valid.shape[0])
    w=angular_weights(rays,index,fov=100)*inside*(sample_face(valid.astype(np.float32),x,y)>.999999)
    return w,x,y,z


def merge_accepted(rgb_faces,depth_faces,valid_faces,width=1024,height=512):
    rays=erp_directions(width,height);total=np.zeros((height,width),np.float64)
    color=np.zeros((height,width,3),np.float64);depth=np.zeros_like(total)
    provenance=np.zeros(total.shape,np.uint8)
    for i,(rgb,zmap,valid) in enumerate(zip(rgb_faces,depth_faces,valid_faces,strict=True)):
        if valid is None:continue
        valid=np.asarray(valid,bool)
        if zmap is not None:valid=valid&np.isfinite(zmap)&(zmap>.001)
        w,x,y,z=valid_weight(rays,i,valid);active=w>1e-6
        total+=w;provenance[active]|=1<<i
        for c in range(3):color[...,c]+=sample_face(np.where(valid,rgb[...,c],0),x,y)*w
        if zmap is not None:
            radial=sample_face(np.where(valid,zmap,0),x,y)/np.maximum(z,1e-8)
            depth+=radial*w
    supported=total>1e-6
    # Keep the saved diagnostic shape compatible; cross-face rejection is disabled.
    conflict=np.zeros_like(supported)
    color=np.rint(np.divide(color,total[...,None],out=np.zeros_like(color),where=total[...,None]>1e-6)).clip(0,255).astype(np.uint8)
    depth=np.divide(depth,total,out=np.zeros_like(depth),where=total>1e-6).astype(np.float32)
    color[~supported]=MASK_COLOR;depth[~supported]=0;provenance[~supported]=0
    return dict(rgb=color,depth=depth,support=supported,provenance=provenance,conflict=conflict)


def compose_accepted(original,generated,requested,support,periodic=True):
    effective=requested&support
    result=composite_outward(original,generated,effective,4,periodic=periodic)
    # Feather must not borrow generated values from unsupported rays.
    result[~support]=original[~support]
    result[requested&~support]=MASK_COLOR
    return result,effective,requested&~support


def normalized_patch_weight(rays,index,valid_faces):
    weights=[valid_weight(rays,i,v)[0] if v is not None else np.zeros(rays.shape[:-1]) for i,v in enumerate(valid_faces)]
    total=np.sum(weights,axis=0)
    return np.divide(weights[index],total,out=np.zeros_like(total),where=total>1e-6)
