"""Metric camera-Z / ERP range conversion and per-face depth comparison.

Pure CPU helpers: no planes, semantic classification, or Gaussian mutation.
"""
import numpy as np
from scipy.ndimage import map_coordinates, distance_transform_edt
from novel_geometry import erp_directions, camera, angular_weights


def merge_depth_faces(faces, width=1024, height=512):
    """Six FOV100 camera-Z maps -> radial distance in metres, with valid weights."""
    if len(faces)!=6:raise ValueError('Exactly six camera-Z faces required')
    rays=erp_directions(width,height)
    accum=np.zeros((height,width),np.float64);total=np.zeros_like(accum)
    for i,source in enumerate(faces):
        source=np.asarray(source,np.float32)
        if source.ndim!=2 or source.shape[0]!=source.shape[1]:raise ValueError('Expected square camera-Z')
        size=source.shape[0];cam=camera(i,size=size,fov=100)
        local=rays@np.asarray(cam['c2w'])[:3,:3]
        z=local[...,2];f=cam['K_pixel_edge'][0][0]
        x=local[...,0]/np.maximum(z,1e-8)*f+size/2-.5
        y=local[...,1]/np.maximum(z,1e-8)*f+size/2-.5
        valid=np.isfinite(source)&(source>1e-3)
        coverage=map_coordinates(valid.astype(np.float32),[y,x],order=1,mode='nearest',prefilter=False)
        sampled=map_coordinates(np.where(valid,source,0),[y,x],order=1,mode='nearest',prefilter=False)
        # Do not extrapolate a border sample beyond its pixel centre: in sparse
        # overlap regions that would silently bias the radial depth.
        inside=(x>=0)&(x<=size-1)&(y>=0)&(y<=size-1)
        weight=angular_weights(rays,i,fov=100)*(coverage>.999)*inside
        radial=sampled/np.maximum(z,1e-8)
        accum+=radial*weight;total+=weight
    output=np.divide(accum,total,out=np.zeros_like(accum),where=total>1e-6).astype(np.float32)
    return output,total>1e-6


def fuse_face(source,predicted,model_valid,hole,context_radius=48,edge_pixels=16,*,comparison_only=False):
    """Reference context IRLS + inside smoothstep; outside is authoritative.

    Functions are imported from the user-specified checkout by the worker.
    Its plane-based *acceptance* procedure is intentionally not used.
    """
    import cv2
    from fuse_and_validate_depth import robust_affine,fill_missing_hole_depth
    source=np.asarray(source,np.float32);predicted=np.asarray(predicted,np.float32);hole=np.asarray(hole,bool)
    observed=np.isfinite(source)&(source>1e-3)
    valid=np.asarray(model_valid,bool)&np.isfinite(predicted)&(predicted>1e-3)
    if any(a.shape!=source.shape for a in (predicted,hole,valid)):raise ValueError('Unregistered depth inputs')
    if np.any(~hole&~observed):raise ValueError('Invalid rendered pixels must be marked as depth holes')
    if not hole.any():return source.copy(),dict(skipped=True,outside_changed_pixels=0,hole_pixels=0)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(context_radius*2+1,context_radius*2+1))
    context=cv2.dilate(hole.astype(np.uint8),kernel).astype(bool)&~hole
    calibration=context&observed&valid
    if calibration.sum()<1000:raise ValueError(f'Too little observed context for LingBot calibration: {calibration.sum()}')
    safe=np.where(observed,source,0)
    gradient=np.hypot(cv2.Sobel(safe,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(safe,cv2.CV_32F,0,1,ksize=3))
    calibration&=gradient<=np.quantile(gradient[calibration],.8)
    scale,offset,residual=robust_affine(predicted[calibration],source[calibration])
    rejected_calibration=None
    if not np.isfinite([scale,offset]).all() or not .5<=scale<=2:
        if not comparison_only:
            raise ValueError(f'Unsafe LingBot calibration: scale={scale:.6f}, offset={offset:.6f}')
        rejected_calibration=dict(scale=float(scale) if np.isfinite(scale) else None,
            offset=float(offset) if np.isfinite(offset) else None)
        scale,offset=1.,0.
        residual=predicted[calibration]-source[calibration]
    calibrated=(scale*predicted+offset).astype(np.float32)
    calibrated_valid=valid&np.isfinite(calibrated)&(calibrated>1e-3)
    calibrated,extrapolated,max_distance=fill_missing_hole_depth(calibrated,calibrated_valid,hole)
    # Nearest authoritative *valid* observation, not a zero-depth hole.
    distance,nearest=distance_transform_edt(hole,return_indices=True)
    nearest_source=source[nearest[0],nearest[1]]
    if np.any(hole&~observed[nearest[0],nearest[1]]):raise ValueError('Hole boundary lacks valid depth')
    t=np.clip(distance/edge_pixels,0,1);blend=t*t*(3-2*t)
    result=source.copy()
    result[hole]=(nearest_source[hole]*(1-blend[hole])+calibrated[hole]*blend[hole]).astype(np.float32)
    if not np.array_equal(result[~hole],source[~hole]):raise AssertionError('Outside camera-Z changed')
    if not np.all(np.isfinite(result[hole])&(result[hole]>1e-3)):raise ValueError('Invalid completed hole depth')
    return result,dict(skipped=False,scale=scale,offset=offset,rejected_calibration=rejected_calibration,
        comparison_only_uncalibrated=bool(rejected_calibration),calibration_samples=int(calibration.sum()),
        calibration_residual_median_m=float(np.median(np.abs(residual))),
        calibration_residual_p95_m=float(np.quantile(np.abs(residual),.95)),
        hole_pixels=int(hole.sum()),model_valid_hole_pixels=int((hole&calibrated_valid).sum()),
        extrapolated_hole_pixels=int(extrapolated.sum()),max_extrapolation_distance_px=max_distance,
        context_radius_px=context_radius,edge_blend_px=edge_pixels,outside_changed_pixels=0)
