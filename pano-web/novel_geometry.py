"""Novel-camera selection and calibrated six-face image projection.

No Gaussian/model changes. OpenCV world coordinates agree with the five-step
PLY pipeline. Mask compositing is a pixel contract, not a model prompt promise.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, minimum_filter, maximum_filter, gaussian_filter, label

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'sharp-sixface'))
from geometry import camera, view_rays, sample_erp, angular_weights

FOV=100.
FACE_SIZE=512
PANO_WIDTH,PANO_HEIGHT=1024,512
FACE_NAMES=('front','right','back','left','up','down')
MASK_COLOR=(0,255,0)
CAPTURE_VERSION='unknown-plus-alpha070-open3-smooth1-green-v3'
ALPHA_THRESHOLD=.70
ALPHA_THRESHOLD_U8=int(np.ceil(255*ALPHA_THRESHOLD))
MASK_SMOOTHING=dict(method='gaussian_then_threshold',sigma_pixels=1.,radius_pixels=3,threshold=.5)
MASK_CLEANUP=dict(version='open3-area16-gauss1-v2',kernel_size=3,iterations=1,min_component_pixels=16,connectivity=8,
    smoothing=MASK_SMOOTHING)


def remove_small_mask_components(mask,periodic=False):
    """8-connected area filter, joining labels across the ERP longitude seam."""
    mask=np.asarray(mask,bool)
    ids,n=label(mask,structure=np.ones((3,3),bool))
    parent=np.arange(n+1)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    if periodic:
        for y in range(mask.shape[0]):
            a=ids[y,0]
            if not a:continue
            for yy in range(max(0,y-1),min(mask.shape[0],y+2)):
                b=ids[yy,-1]
                if b:parent[root(b)]=root(a)
    roots=np.array([root(i) for i in range(n+1)])
    counts=np.bincount(roots[ids].ravel(),minlength=n+1)
    keep=counts[roots]>=MASK_CLEANUP['min_component_pixels'];keep[0]=False
    return keep[ids]


def smooth_mask(mask,periodic=False):
    """Feather a binary mask with sigma=1 px, recover the 50% contour.

    This rounds both convex and concave edges; do not intersect with the old
    mask, which would prevent concave edges from being smoothed. The result is
    binary, and any small components created by smoothing are removed again.
    """
    feathered=gaussian_filter(np.asarray(mask,np.float64),sigma=MASK_SMOOTHING['sigma_pixels'],
        truncate=MASK_SMOOTHING['radius_pixels']/MASK_SMOOTHING['sigma_pixels'],
        mode=('nearest','wrap' if periodic else 'nearest'))
    return remove_small_mask_components(feathered>=MASK_SMOOTHING['threshold'],periodic)


def clean_mask(mask,periodic=False,*,smoothing=True):
    """3x3 opening, area >=16 px, then light feather/threshold smoothing.

    ERP longitude wraps throughout. Other borders clamp. Opening only removes
    pixels; the final 50% contour may move slightly inward or outward.
    """
    mask=np.asarray(mask,bool)
    if mask.ndim!=2 or min(mask.shape)==0:raise ValueError('Expected a nonempty 2-D mask')
    mode=('nearest','wrap' if periodic else 'nearest')
    opened=maximum_filter(minimum_filter(mask,size=3,mode=mode),size=3,mode=mode)&mask
    cleaned=remove_small_mask_components(opened,periodic)
    return smooth_mask(cleaned,periodic) if smoothing else cleaned


def repair_masks(geometry,alphas):
    raw_alpha=merge_alpha_holes(alphas)
    g=clean_mask(geometry,periodic=True);a=clean_mask(raw_alpha,periodic=True)
    combined=clean_mask(g|a,periodic=True)
    # Pixels filled only by smoothing the union have contributions from both
    # masks. Preserve this provenance so the color layers exactly cover Mask.
    shared=combined&~(g|a)
    return dict(geometry=(g|shared)&combined,alpha=(a|shared)&combined,combined=combined,
        raw_alpha=raw_alpha,shared_smoothing_pixels=int(shared.sum()))


def repair_faces(panorama,mask,alphas):
    faces=six_faces(panorama,mask)
    for i,face in enumerate(faces):
        # Clean AFTER native alpha union: small holes must not be reintroduced.
        face['mask']=clean_mask(face['mask']|(alphas[i]<ALPHA_THRESHOLD_U8))
    return faces


def merge_alpha_holes(alphas,width=PANO_WIDTH,height=PANO_HEIGHT):
    """Union alpha < ALPHA_THRESHOLD from every contributing calibrated face.

    Threshold before projection: averaging/rounding alpha would hide small
    transparent holes or let an opaque overlapping face erase them.
    """
    if len(alphas)!=6:raise ValueError('Exactly six alpha images required')
    directions=erp_directions(width,height);holes=np.zeros((height,width),bool)
    for index,alpha in enumerate(alphas):
        alpha=np.asarray(alpha)
        if alpha.dtype!=np.uint8 or alpha.ndim!=2 or alpha.shape[0]!=alpha.shape[1]:
            raise ValueError('Expected square uint8 render alpha')
        size=alpha.shape[0];cam=camera(index,size=size,fov=FOV)
        local=directions@np.asarray(cam['c2w'])[:3,:3];focal=cam['K_pixel_edge'][0][0]
        x=local[...,0]/np.maximum(local[...,2],1e-8)*focal+size/2-.5
        y=local[...,1]/np.maximum(local[...,2],1e-8)*focal+size/2-.5
        sampled=map_coordinates((alpha<ALPHA_THRESHOLD_U8).astype(np.float32),[y,x],order=1,mode='nearest',prefilter=False)
        holes|=(sampled>0)&(angular_weights(directions,index,fov=FOV)>1e-4)
    return holes


def next_camera(candidates, selected_ids, policy='four-neighbor'):
    """Global frontier of ALL selected cameras, never just the last position.

    C000 is the initial anchor. Preview clicks must not mutate selected_ids.
    Ties follow existing square-ring numbering. Return None when exhausted.
    """
    if policy not in ('four-neighbor','max-mask'):raise ValueError('Unknown camera selection policy')
    by_id={c['id']:c for c in candidates}
    selected=set(selected_ids)|{'C000'}
    if not selected <= by_id.keys():
        raise ValueError('Selected camera is not in the current candidate set')
    coords={tuple(by_id[c]['grid']) for c in selected}
    frontier=[]
    for c in candidates:
        if c['id'] in selected:continue
        x,z=c['grid']
        if policy=='max-mask' or any((x+dx,z+dz) in coords for dx,dz in ((1,0),(-1,0),(0,1),(0,-1))):
            value=c['mask_stats']['mask_fraction']
            if not np.isfinite(value) or not 0<=value<=.15:
                raise ValueError('Candidate does not satisfy the fixed Mask threshold')
            frontier.append(c)
    frontier.sort(key=lambda c:(-c['mask_stats']['mask_fraction'],int(c['id'][1:])))
    return (frontier[0]['id'] if frontier else None),[c['id'] for c in frontier]


def erp_directions(width=PANO_WIDTH,height=PANO_HEIGHT):
    lon=((np.arange(width)+.5)/width-.5)*2*np.pi
    lat=((np.arange(height)+.5)/height-.5)*np.pi
    return np.stack([np.cos(lat)[:,None]*np.sin(lon)[None,:],
        np.broadcast_to(np.sin(lat)[:,None],(height,width)),
        np.cos(lat)[:,None]*np.cos(lon)[None,:]],axis=-1).astype(np.float32)


def six_faces(panorama,mask,size=FACE_SIZE):
    panorama=np.asarray(panorama,np.uint8);mask=np.asarray(mask,bool)
    if panorama.shape[:2]!=mask.shape or panorama.shape[-1]!=3:raise ValueError('RGB/Mask size mismatch')
    result=[]
    for index,name in enumerate(FACE_NAMES):
        cam=camera(index,size=size,fov=FOV);_,rays=view_rays(cam)
        rgb=np.rint(sample_erp(panorama,rays)).clip(0,255).astype(np.uint8)
        # Any positive bilinear contribution protects even thin Mask regions.
        face_mask=sample_erp(mask.astype(np.float32),rays)>0
        result.append(dict(name=name,rgb=rgb,mask=face_mask,camera=cam))
    return result


def merge_faces(faces,width=PANO_WIDTH,height=PANO_HEIGHT):
    """Raised-cosine overlap fusion: 100-degree images, no seam hard crop."""
    if len(faces)!=6:raise ValueError('Exactly six images required')
    directions=erp_directions(width,height)
    accum=np.zeros((height,width,3),np.float32)
    total=np.zeros((height,width),np.float32)
    for index,image in enumerate(faces):
        image=np.asarray(image)
        if image.ndim!=3 or image.shape[-1]!=3 or image.shape[0]!=image.shape[1]:
            raise ValueError('Cube face must be square RGB')
        size=image.shape[0];cam=camera(index,size=size,fov=FOV)
        local=directions@np.asarray(cam['c2w'])[:3,:3]
        focal=cam['K_pixel_edge'][0][0]
        x=local[...,0]/np.maximum(local[...,2],1e-8)*focal+size/2-.5
        y=local[...,1]/np.maximum(local[...,2],1e-8)*focal+size/2-.5
        weight=angular_weights(directions,index,fov=FOV)
        # Clamp only for border interpolation; angular weights reject out-of-FOV rays.
        sampled=np.stack([map_coordinates(image[...,k].astype(np.float32),[y,x],order=1,mode='nearest',prefilter=False) for k in range(3)],axis=-1)
        accum+=sampled*weight[...,None];total+=weight
    if np.min(total)<.999:raise ValueError('Six-face spherical coverage gap')
    return np.rint(accum/total[...,None]).clip(0,255).astype(np.uint8)


def composite_inside_mask(original,generated,mask,feather_pixels=4,periodic=False):
    """Mask-only inner feathering; every outside RGB byte remains identical.

    ERP longitude wraps. Feathering never dilates the authorized edit domain.
    If every pixel is masked there is no observed boundary to blend against.
    """
    original=np.asarray(original,np.uint8);generated=np.asarray(generated,np.uint8)
    mask=np.asarray(mask,bool)
    if original.shape!=generated.shape or original.shape[:2]!=mask.shape or original.shape[-1]!=3:
        raise ValueError('Composite shapes must match')
    if feather_pixels<0:raise ValueError('Feather width cannot be negative')
    if not mask.any():return original.copy()
    if mask.all() or feather_pixels==0:
        weight=mask.astype(np.float32)
    else:
        if periodic:
            w=mask.shape[1]
            distance=distance_transform_edt(np.tile(mask,(1,3)))[:,w:2*w]
        else:
            distance=distance_transform_edt(np.pad(mask,1,constant_values=False))[1:-1,1:-1]
        weight=np.minimum(distance/max(float(feather_pixels),1),1).astype(np.float32)*mask
    output=original.copy()
    blended=np.rint(original.astype(np.float32)*(1-weight[...,None])+generated.astype(np.float32)*weight[...,None]).clip(0,255).astype(np.uint8)
    output[mask]=blended[mask]
    if not np.array_equal(output[~mask],original[~mask]):raise AssertionError('Unmasked RGB changed')
    return output


def outward_blend_weights(mask,feather_pixels=4,periodic=False):
    """One throughout every hole (even a single pixel); fade only OUTSIDE it.

    Pixels at least feather_pixels away from the mask have zero weight. ERP
    longitude wraps; top/bottom do not. No semantic or planar surface priors.
    """
    mask=np.asarray(mask,bool)
    if mask.ndim!=2 or not np.isfinite(feather_pixels) or feather_pixels<0:
        raise ValueError('Expected a 2-D mask and finite nonnegative feather width')
    if not mask.any() or mask.all() or feather_pixels==0:return mask.astype(np.float32)
    if periodic:
        w=mask.shape[1]
        distance=distance_transform_edt(np.tile(~mask,(1,3)))[:,w:2*w]
    else:distance=distance_transform_edt(~mask)
    return np.clip(1-distance/float(feather_pixels),0,1).astype(np.float32)


def composite_outward(original,generated,mask,feather_pixels=4,periodic=False):
    original=np.asarray(original,np.uint8);generated=np.asarray(generated,np.uint8)
    mask=np.asarray(mask,bool)
    if original.shape!=generated.shape or original.shape[:2]!=mask.shape or original.shape[-1]!=3:
        raise ValueError('Composite shapes must match')
    weight=outward_blend_weights(mask,feather_pixels,periodic)
    result=np.rint(original.astype(np.float32)*(1-weight[...,None])+
        generated.astype(np.float32)*weight[...,None]).clip(0,255).astype(np.uint8)
    result[mask]=generated[mask]
    if not np.array_equal(result[weight==0],original[weight==0]):raise AssertionError('Outside feather changed')
    return result
