"""Stage-9 panorama projection and reference SHARP patch selection."""
from pathlib import Path
import sys
import numpy as np
from novel_geometry import camera,view_rays,sample_erp,angular_weights

from project_paths import REPAIR as REFERENCE
BLEND_REFERENCE_PIXELS=6
BLEND_REFERENCE_SHORT_SIDE=512


def project_inputs(rgb,radial,mask,index,position,size=512,*,support=None,face_valid=None,face_enabled=True):
    """ERP range -> registered perspective RGB / camera-Z / authority Mask.

    Invalid depth is never averaged as zero. Reference extrapolation fills only
    unsupported samples in this working copy, using valid LingBot depth alone.
    """
    sys.path.insert(0,str(REFERENCE))
    from fuse_and_validate_depth import fill_missing_hole_depth
    rgb=np.asarray(rgb,np.uint8);radial=np.asarray(radial,np.float32);mask=np.asarray(mask,bool)
    if rgb.shape!=(*radial.shape,3) or mask.shape!=radial.shape:raise ValueError('Unregistered RGB/depth/Mask')
    cam=camera(index,size=size,fov=100)
    local,rays=view_rays(cam)
    valid=np.isfinite(radial)&(radial>1e-3)
    if support is not None:valid &= np.asarray(support,bool)
    coverage=sample_erp(valid.astype(np.float32),rays)
    projected=sample_erp(np.where(valid,radial,0),rays)*local[...,2]
    supported=coverage>.999999
    projected[~supported]=0
    supported&=np.isfinite(projected)&(projected>1e-3)
    if face_valid is not None:
        if np.asarray(face_valid).shape!=supported.shape:raise ValueError('Face support shape mismatch')
        supported &= np.asarray(face_valid,bool)
    supported &= face_enabled
    projected[~supported]=0
    if supported.any():target,filled,distance=fill_missing_hole_depth(projected,supported,~supported)
    else:target=np.ones_like(projected);filled=np.ones_like(supported);distance=0.
    if not np.array_equal(target[supported],projected[supported]):raise AssertionError('Valid LingBot samples changed')
    if not np.all(np.isfinite(target)&(target>1e-3)):raise ValueError('SHARP requires dense positive depth')
    c2w=np.asarray(cam['c2w']);c2w[:3,3]=position
    cam.update(c2w=c2w.tolist(),w2c=np.linalg.inv(c2w).tolist(),intrinsic_K_opencv=cam['K_pixel_edge'],
        splats=[dict(model_view_opencv_for_original_splat=np.linalg.inv(c2w).tolist())],
        coordinates='OpenCV world, metres; fixed confirmed camera position')
    return dict(rgb=np.rint(sample_erp(rgb,rays)).clip(0,255).astype(np.uint8),
        depth_projected=projected,depth_input=target.astype(np.float32),mask=(sample_erp(mask.astype(np.float32),rays)>0)&supported,
        support=supported,
        extrapolated=filled,camera=cam,diagnostics=dict(extrapolated_pixels=int(filled.sum()),
            max_extrapolation_distance_px=distance,valid_lingbot_samples_unchanged=True,
            source='stage8_direct_lingbot_panorama',invalid_fill='reference_fill_missing_hole_depth'))


def select_patch(vertices,layer0_z,hole,index,cam,point_keep=None,*,support=None,valid_faces=None):
    """Reference row provenance, six opacity rings, rigid center/frame transform.

    The six-face adapter additionally partitions opacity in overlaps; it never
    moves centers or refits depth. Only layer-0 rows may enter the patch.
    """
    import cv2
    from merge_hard_patch import build_ring_blend_domain,scaled_reference_pixels
    from gaussian_patch_io import transform_patch_to_scene
    side=layer0_z.shape[0];layer_size=side*side
    if layer0_z.shape!=(side,side) or len(vertices)!=2*layer_size:raise ValueError('Unexpected SHARP lattice')
    radius=scaled_reference_pixels(BLEND_REFERENCE_PIXELS,hole.shape[1],hole.shape[0],BLEND_REFERENCE_SHORT_SIDE)
    labels,weights,rings=build_ring_blend_domain(hole,radius,6)
    if support is not None:
        labels=labels.copy();weights=weights.copy();labels[~support]=0;weights[~support]=0
    label_grid=cv2.resize(labels,(side,side),interpolation=cv2.INTER_NEAREST)
    weight_grid=cv2.resize(weights,(side,side),interpolation=cv2.INTER_NEAREST)
    ids=np.flatnonzero(label_grid.reshape(-1)>0)
    tail=vertices[ids].copy()
    xyz=np.column_stack([tail[n] for n in ('x','y','z')]).astype(np.float64)
    rot=np.asarray(cam['c2w'])[:3,:3]
    overlap=angular_weights(xyz@rot.T,index,fov=100)
    if valid_faces is not None:
        from quality_fusion import normalized_patch_weight
        overlap=normalized_patch_weight(xyz@rot.T,index,valid_faces)
    keep=overlap>1e-4
    ids=ids[keep];tail=tail[keep];overlap=overlap[keep]
    before_filter=len(ids)
    if point_keep is not None:
        if point_keep.dtype!=np.bool_ or point_keep.shape!=(len(vertices),):raise ValueError('Point mask identity mismatch')
        keep=point_keep[ids]
        ids=ids[keep];tail=tail[keep];overlap=overlap[keep]
    selected_weight=weight_grid.reshape(-1)[ids]*overlap
    old=1/(1+np.exp(-np.clip(tail['opacity'].astype(np.float64),-60,60)))
    alpha=np.clip(old*selected_weight,1e-6,1-1e-6)
    tail['opacity']=(np.log(alpha)-np.log1p(-alpha)).astype(tail['opacity'].dtype)
    transform=transform_patch_to_scene(tail,np.asarray(cam['w2c']))
    scene_xyz=np.asarray(transform['xyz'],np.float64)
    roundtrip=np.c_[scene_xyz,np.ones(len(tail))]@np.asarray(cam['w2c']).T
    camera_xyz=np.column_stack([tail[n] for n in ('x','y','z')])
    error=np.linalg.norm(roundtrip[:,:3]-camera_xyz,axis=1)
    zerror=np.abs(tail['z']-layer0_z.reshape(-1)[ids])
    if len(tail) and (error.max()>1e-5 or zerror.max()>1e-5):raise ValueError('Patch camera/depth roundtrip failed')
    stats=dict(count=len(tail),layer=0,grid_size=side,blend_reference_pixels=BLEND_REFERENCE_PIXELS,blend_reference_short_side=BLEND_REFERENCE_SHORT_SIDE,
        point_filter_applied=point_keep is not None,before_point_filter=before_filter,point_filter_removed=before_filter-len(tail),
        blend_radius_pixels=radius,blend_ring_count=6,core_opacity_mode='full',ring_weights=rings,
        selected_by_label={str(i):int((label_grid.reshape(-1)[ids]==i).sum()) for i in range(1,8)},
        center_z_error_max_m=float(zerror.max()) if len(tail) else 0.,
        world_camera_roundtrip_max_m=float(error.max()) if len(tail) else 0.,
        overlap='100-degree normalized angular opacity weights',posthoc_depth_fit=False)
    return tail,transform,ids,labels,selected_weight,stats
