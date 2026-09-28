"""Shared RGB-only outer feathering and diagnostics; no model or depth imports."""
import numpy as np
from PIL import Image
from novel_geometry import FACE_NAMES,merge_faces,outward_blend_weights,composite_outward

COMPOSITE=dict(version='outward-v1',direction='outward',feather_pixels=4,
    inside_mask='100% generated',outside_feather='byte-exact original')


def composite_stats(original,generated,mask,result,periodic=False):
    weight=outward_blend_weights(mask,COMPOSITE['feather_pixels'],periodic)
    changed=np.any(result!=original,axis=-1)
    stats=dict(outside_mask_changed_pixels=int(changed[~mask].sum()),
        outside_feather_changed_pixels=int(changed[weight==0].sum()),
        inside_generated_mismatch_pixels=int(np.any(result!=generated,axis=-1)[mask].sum()),
        feather_band_pixels=int(((weight>0)&~mask).sum()))
    if stats['outside_feather_changed_pixels'] or stats['inside_generated_mismatch_pixels']:
        raise AssertionError('Outer feather pixel contract failed')
    return stats


def compose_faces(capture,out,models,quality=None):
    """Compose previews per face, but fuse UNBLENDED API images for the ERP.

    Blending face previews and then feathering them again attenuates thin holes.
    Each deliverable is therefore feathered exactly once in its own image space.
    """
    if quality is not None:
        from quality_fusion import merge_accepted,compose_accepted
        records=[];valid=[]
        for name,generated,q in zip(FACE_NAMES,models,quality,strict=True):
            base=np.asarray(Image.open(capture/f'{name}.png').convert('RGB'))
            mask=np.asarray(Image.open(capture/f'{name}-mask.png'))>0
            accepted=q['status'] in ('passed','not_needed')
            support=np.full(mask.shape,accepted,bool);valid.append(support if accepted else None)
            result,effective,unresolved=compose_accepted(base,generated,mask,support,False)
            Image.fromarray(result).save(out/f'{name}-repaired.png')
            values=composite_stats(base,generated,mask,result) if accepted else dict(
                outside_mask_changed_pixels=0,outside_feather_changed_pixels=0,inside_generated_mismatch_pixels=0,feather_band_pixels=0)
            records.append(dict(**values,unresolved_pixels=int(unresolved.sum())))
        fused=merge_accepted(models,[None]*6,valid)
        Image.fromarray(fused['rgb']).save(out/'fused-before-composite.png')
        original=np.asarray(Image.open(capture/'raw.png').convert('RGB'))
        requested=np.asarray(Image.open(capture/'mask.png'))>0
        final,effective,unresolved=compose_accepted(original,fused['rgb'],requested,fused['support'])
        for name,array in [('panorama',final),('rgb-support',fused['support']),('effective-mask',effective),('unresolved-mask',unresolved),('rgb-provenance',fused['provenance'])]:
            Image.fromarray(array.astype(np.uint8)*255 if array.dtype==bool else array).save(out/f'{name}.png')
        weight=outward_blend_weights(requested,COMPOSITE['feather_pixels'],True)
        outside=int(np.any(final!=original,axis=-1)[weight==0].sum())
        if outside:raise AssertionError('Quality composite changed observed RGB')
        return records,dict(outside_mask_changed_pixels=int(np.any(final!=original,axis=-1)[~requested].sum()),
            outside_feather_changed_pixels=outside,inside_generated_mismatch_pixels=0,
            effective_mask_pixels=int(effective.sum()),unresolved_pixels=int(unresolved.sum()))
    records=[]
    for name,generated in zip(FACE_NAMES,models,strict=True):
        base=np.asarray(Image.open(capture/f'{name}.png').convert('RGB'))
        mask=np.asarray(Image.open(capture/f'{name}-mask.png'))>0
        result=composite_outward(base,generated,mask,COMPOSITE['feather_pixels'])
        Image.fromarray(result).save(out/f'{name}-repaired.png')
        records.append(composite_stats(base,generated,mask,result))
    fused=merge_faces(models)
    Image.fromarray(fused).save(out/'fused-before-composite.png')
    original=np.asarray(Image.open(capture/'raw.png').convert('RGB'))
    mask=np.asarray(Image.open(capture/'mask.png'))>0
    final=composite_outward(original,fused,mask,COMPOSITE['feather_pixels'],periodic=True)
    Image.fromarray(final).save(out/'panorama.png')
    return records,composite_stats(original,fused,mask,final,periodic=True)
