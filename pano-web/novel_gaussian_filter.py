"""First-generation point culling, recomputed on this camera's initializer."""
import time
import numpy as np
from novel_geometry import FACE_NAMES,angular_weights
from novel_pipeline import read,write_json
from point_normals import estimate_angles,PARAMETERS as NORMAL_PARAMETERS
from point_distances import normal_keep,iter_neighbor_means
from point_mask import fixed_keep,PARAMETERS,NEIGHBORS


def initializer_world(ndc,cam):
    """Same raw initializer NDC -> camera -> world transforms as worker.py."""
    ndc=np.asarray(ndc,np.float32)
    if ndc.ndim!=4 or ndc.shape[-1]!=3 or not np.isfinite(ndc).all():
        raise ValueError('Invalid initializer grid')
    width,height=cam['size_wh'];focal=np.asarray(cam['intrinsic_K_opencv'])[[0,1],[0,1]]
    camera=(ndc*np.array([width/(2*focal[0]),height/(2*focal[1]),1.])).astype(np.float32)
    pose=np.asarray(cam['c2w'],np.float64)
    return (camera.astype(np.float64)@pose[:3,:3].T+pose[:3,3]).astype(np.float32)


def capture_initializer(output,cam,face):
    """Read-only init_model hook, before any composer or learned displacement."""
    import torch
    base=output.gaussian_base_values
    z=base.mean_inverse_z_ndc.reciprocal()
    means=torch.cat((base.mean_x_ndc*z,base.mean_y_ndc*z,z),dim=1)
    means=means.permute(0,2,3,4,1)
    if output.global_scale is not None:means=means*output.global_scale[:,None,None,None,None]
    ndc=means[0].detach().float().cpu().numpy().copy()
    np.save(face/'initializer-ndc.npy',ndc)
    np.save(face/'initializer-world.npy',initializer_world(ndc,cam))


def filter_cloud(points,angles,progress=print,*,strict_quality=False):
    """Use the original normal -> full-cloud one-pass exact KNN -> mask code."""
    points=np.asarray(points);angles=np.asarray(angles,np.float32)
    if points.shape!=(len(angles),3):raise ValueError('Point/angle identity mismatch')
    normal_ids=np.flatnonzero(normal_keep(angles))
    scores=np.full(len(angles),-1,np.float32)
    reference=points[normal_ids]
    if strict_quality and len(reference)<65:
        from quality_retry import QualityNoContribution
        raise QualityNoContribution('有效近邻参考点不足 65 个')
    progress(f'固定 82°：保留 {len(reference):,} 个初始化点，计算全量 K=64 距离')
    last=-1
    for start,values in iter_neighbor_means(reference,neighbors=[NEIGHBORS]):
        stop=start+len(values);scores[normal_ids[start:stop]]=values[:,0]
        percent=int(100*stop/max(1,len(reference)))
        if percent//10!=last:
            last=percent//10;progress(f'补丁点剔除：K=64 距离 {percent}%')
    return fixed_keep(angles,scores),scores


def build_point_filter(out,progress=print,*,accepted_faces=None):
    start=time.perf_counter();points=[];angles=[];domains=[];records=[]
    for index,name in enumerate(FACE_NAMES):
        if accepted_faces is not None and name not in accepted_faces:continue
        face=out/name;cam=read(face/'camera.json');grid=np.load(face/'initializer-world.npy')
        if grid.shape!=(2,768,768,3):raise ValueError('Expected original two-layer initializer grid')
        center=np.asarray(cam['c2w'])[:3,3]
        progress(f'补丁法向 {index+1}/6 {name}：原始双层初始化点，5×5 平面拟合')
        support=None
        if accepted_faces is not None:
            from PIL import Image
            from scipy.ndimage import minimum_filter,zoom
            domain512=np.asarray(Image.open(face/'support.png'))>0
            # Every bilinear footprint and every 5x5 normal window must be valid.
            support=zoom(domain512.astype(np.float32),768/512,order=1,mode='nearest')>.999999
            supported_normals=minimum_filter(support.astype(np.uint8),size=5,mode='constant',cval=0)>0
        angle=np.stack([estimate_angles(layer if support is None else np.where(support[...,None],layer,np.nan),center,**NORMAL_PARAMETERS) for layer in grid]).reshape(-1)
        xyz=grid.reshape(-1,3);domain=angular_weights(xyz.astype(np.float64)-center,index,fov=100)>1e-4
        if support is not None:domain &= np.tile(supported_normals.ravel(),2)
        np.save(face/'point-angles.npy',angle);np.save(face/'point-domain.npy',domain)
        points.append(xyz[domain]);angles.append(angle[domain]);domains.append(domain)
        records.append(dict(face=name,source_count=len(xyz),cloud_count=int(domain.sum()),outside_cloud=int((~domain).sum())))
    if not points:
        from quality_retry import QualityNoContribution
        raise QualityNoContribution('没有通过检查的补丁参考面')
    full_angles=np.concatenate(angles);del angles
    keep,scores=filter_cloud(np.concatenate(points),full_angles,progress,strict_quality=accepted_faces is not None);del points
    cursor=0
    for domain,record in zip(domains,records):
        count=record['cloud_count'];face=out/record['face'];part=keep[cursor:cursor+count]
        full_keep=np.zeros(len(domain),bool);full_keep[domain]=part
        full_scores=np.full(len(domain),-1,np.float32);full_scores[domain]=scores[cursor:cursor+count]
        np.save(face/'point-keep.npy',full_keep);np.save(face/'point-distance.npy',full_scores)
        normal_removed=int((~normal_keep(full_angles[cursor:cursor+count])).sum())
        record.update(kept=int(part.sum()),removed=count-int(part.sum()),normal_removed=normal_removed,
            distance_removed=count-int(part.sum())-normal_removed)
        cursor+=count
    report=dict(parameters=PARAMETERS,normal_parameters=NORMAL_PARAMETERS,index='face -> layer -> row -> column (original initializer)',
        reference='accepted non-extrapolated new-camera initializer support, both layers, after normal filter; before hole selection' if accepted_faces is not None else 'full new-camera initializer cloud, both layers and all six faces, after normal filter; before hole selection',
        camera_origin='confirmed new camera position',count=len(keep),kept=int(keep.sum()),removed=int((~keep).sum()),
        normal_removed=int((~normal_keep(full_angles)).sum()),faces=records,seconds=time.perf_counter()-start)
    write_json(out/'point-filter.json',report)
    return report
