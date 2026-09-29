"""CPU Gaussian geometry and calibrated six-face fusion."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement
from geometry import angular_weights, feather_alpha

FIELDS = ['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
DTYPE = np.dtype([(k,'<f4') for k in FIELDS])


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def summary(value):
    a=np.asarray(value,dtype=np.float64).reshape(-1)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Empty or nonfinite diagnostic')
    return dict(zip(['p01','p10','median','p90','p99','max'],map(float,np.quantile(a,[.01,.1,.5,.9,.99,1]))))


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[:,[1,2,3,0]]).as_matrix()


def quaternions(r):
    return Rotation.from_matrix(r).as_quat()[:,[3,0,1,2]].astype(np.float32)


def covariance(scene):
    r=rotation(scene['quats'])
    a=r*scene['scales'][:,None,:]
    return a@a.transpose(0,2,1)


def ndc_to_camera(scene, cam):
    size=cam['size_wh'][0]; focal=cam['K_pixel_edge'][0][0]
    scale=np.array([size/(2*focal),size/(2*focal),1.0])
    out={k:v.copy() for k,v in scene.items()}
    out['means']=(scene['means']*scale).astype(np.float32)
    c=covariance(scene)*scale[None,:,None]*scale[None,None,:]
    vals,vec=np.linalg.eigh(c)
    vec[np.linalg.det(vec)<0,:,-1]*=-1
    out['scales']=np.sqrt(np.maximum(vals,1e-16)).astype(np.float32)
    out['quats']=quaternions(vec)
    # Verify covariance, not just center positions, survived NDC unprojection.
    idx=np.arange(0,len(vals),max(1,len(vals)//4096))
    recovered=covariance({k:v[idx] for k,v in out.items()})
    rel=np.linalg.norm(recovered-c[idx],axis=(1,2))/np.maximum(np.linalg.norm(c[idx],axis=(1,2)),1e-16)
    if rel.max()>2e-5:
        raise RuntimeError('NDC covariance transform failed')
    return out, {'covariance_relative_error_max':float(rel.max()),'eigenvalue_floor_count':int((vals<1e-16).sum())}


def world_transform(scene, cam):
    r=np.asarray(cam['c2w'])[:3,:3]
    means=(scene['means'].astype(np.float64)@r.T).astype(np.float32)
    rot=rotation(scene['quats'])
    q=quaternions(r[None]@rot)
    return {**scene,'means':means,'quats':q}


def to_vertices(scene, face_index, fov=100., *, return_indices=False):
    weight=angular_weights(scene['means'],face_index,fov)
    keep=(weight>1e-4)&(scene['alpha']>1e-7)
    s={k:v[keep] for k,v in scene.items()}
    alpha=feather_alpha(s['alpha'],weight[keep])
    alpha=np.clip(alpha,1e-7,1-1e-7)
    colors=s['colors']
    srgb=np.where(colors<=.0031308,colors*12.92,1.055*np.maximum(colors,.0031308)**(1/2.4)-.055)
    sh=(srgb-.5)/np.sqrt(1/(4*np.pi))
    out=np.empty(len(alpha),dtype=DTYPE)
    blocks={'x':s['means'][:,0],'y':s['means'][:,1],'z':s['means'][:,2],
            'opacity':np.log(alpha/(1-alpha))}
    for i in range(3):
        blocks[f'f_dc_{i}']=sh[:,i]; blocks[f'scale_{i}']=np.log(np.maximum(s['scales'][:,i],1e-8))
    for i in range(4): blocks[f'rot_{i}']=s['quats'][:,i]
    for k,v in blocks.items(): out[k]=v
    validate_vertices(out)
    info = {'before':len(weight),'kept':len(out),'dropped_zero_edge_or_opacity':int((~keep).sum()),
            'angular_weight':summary(weight[keep]),'opacity_merge':'partition_optical_thickness'}
    # Preserve source identity through native angular/opacity culling. Adjusted
    # centers can move, so masks must never be joined by coordinates or PLY row.
    return (out, info, np.flatnonzero(keep)) if return_indices else (out, info)


def validate_vertices(v, *, allow_empty=False):
    if v.dtype.names!=tuple(FIELDS) or (len(v)==0 and not allow_empty):
        raise ValueError('Invalid Gaussian PLY schema/count')
    if len(v)==0:
        return
    for k in FIELDS:
        if not np.isfinite(v[k]).all(): raise ValueError(f'Nonfinite {k}')
    norms=np.sqrt(sum(v[f'rot_{i}'].astype(np.float64)**2 for i in range(4)))
    if np.max(np.abs(norms-1))>2e-5: raise ValueError('Non-unit quaternion')


def save_vertices(path, vertices, *, allow_empty=False):
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    partial=path.with_suffix('.partial.ply')
    if partial.exists(): raise FileExistsError(partial)
    validate_vertices(vertices, allow_empty=allow_empty)
    PlyData([PlyElement.describe(vertices,'vertex')],text=False,byte_order='<',
            comments=['SHARP six-face panorama; OpenCV world axes; units meters; SH0 sRGB']).write(partial)
    partial.rename(path)
    return {'path':str(path),'gaussians':len(vertices),'bytes':path.stat().st_size,'sha256':digest(path),
            'finite':True,'unit_quaternions':True,'bit_exact_readback':None}
