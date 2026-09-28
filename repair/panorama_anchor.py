"""Put Gaussian centers on the shared ERP surface at their actual rays."""
import numpy as np
from geometry import sample_erp


def anchor_points(points, radial, *, rotation=None, support=None):
    """Same-ray centers and distance ratios, with no learned depth alignment.

    Points are relative to the panorama camera. Rotation maps camera to ERP
    axes. Missing ERP samples retain existing extrapolated depth. Valid bilinear
    samples are normalized without including invalid depths as zero.
    """
    points = np.asarray(points)
    world = points.astype(np.float64)
    if rotation is not None:
        world = world @ np.asarray(rotation, dtype=np.float64).T
    old_range = np.linalg.norm(world, axis=-1)
    valid = np.isfinite(radial) & (radial > 0)
    if support is not None:
        valid &= np.asarray(support, dtype=bool)
    coverage = sample_erp(valid.astype(np.float32), world)
    sampled = sample_erp(np.where(valid, radial, 0), world)
    target = np.divide(sampled, coverage, out=old_range.copy(), where=coverage > 0)
    ratio = np.divide(target, old_range, out=np.ones_like(target), where=old_range > 0)
    return (points * ratio[..., None]).astype(points.dtype), ratio.astype(np.float32)


def anchor_ndc_scene(scene, radial, cam):
    """Preserve predicted rays/appearance, transporting scale with ray distance."""
    size = cam['size_wh'][0]
    scale = np.array([size/(2*cam['K_pixel_edge'][0][0])]*2+[1.])
    camera_means = scene['means'].astype(np.float64) * scale
    _, ratio = anchor_points(camera_means, radial, rotation=np.asarray(cam['c2w'])[:3,:3])
    return {**scene, 'means': scene['means']*ratio[:,None],
            'scales': scene['scales']*ratio[:,None]}
