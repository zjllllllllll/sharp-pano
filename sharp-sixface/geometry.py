"""Calibrated 100-degree cube views. OpenCV camera/world axes, meters.

Pixel coordinates in K use the pixel-edge convention: pixel centers are
(column + .5, row + .5). Thus cx=cy=size/2, matching SHARP's NDC unprojection.
"""
import math
import numpy as np

FACE_SPECS = (
    ('front', (0, 0, 1), (0, 1, 0)),
    ('right', (1, 0, 0), (0, 1, 0)),
    ('back', (0, 0, -1), (0, 1, 0)),
    ('left', (-1, 0, 0), (0, 1, 0)),
    ('up', (0, -1, 0), (0, 0, 1)),
    ('down', (0, 1, 0), (0, 0, -1)),
)


def camera(face, size=1536, fov=100.0):
    name, forward, down_hint = FACE_SPECS[face] if isinstance(face, int) else next(x for x in FACE_SPECS if x[0] == face)
    forward = np.asarray(forward, dtype=np.float64)
    right = np.cross(np.asarray(down_hint, dtype=np.float64), forward)
    down = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.stack([right, down, forward], axis=1)
    focal = size / (2 * math.tan(math.radians(fov) / 2))
    k = np.array([[focal, 0, size/2], [0, focal, size/2], [0, 0, 1]], dtype=np.float64)
    return {'face': name, 'size_wh': [size, size], 'hfov_deg': fov, 'vfov_deg': fov,
            'K_pixel_edge': k.tolist(), 'K_pixel_index': (k - np.array([[0,0,.5],[0,0,.5],[0,0,0]])).tolist(),
            'c2w': c2w.tolist(), 'w2c': np.linalg.inv(c2w).tolist(),
            'coordinates': 'OpenCV: X right, Y down, Z forward; all six cameras at origin; meters',
            'pixel_convention': 'K_pixel_edge projects to (column+0.5,row+0.5); K_pixel_index projects to array indices',
            'pose_estimated': False, 'focal_estimated': False}


def view_rays(cam, grid_size=None):
    size = cam['size_wh'][0]
    n = grid_size or size
    focal = cam['K_pixel_edge'][0][0]
    axis = ((np.arange(n, dtype=np.float64) + .5) * size/n - size/2) / focal
    x, y = np.meshgrid(axis, axis)
    local = np.stack([x, y, np.ones_like(x)], axis=-1)
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    world = local @ np.asarray(cam['c2w'])[:3, :3].T
    return local.astype(np.float32), world.astype(np.float32)


def sample_erp(panorama, directions):
    """Bilinear periodic-longitude sampling, clamped at the polar pixel rows."""
    d = np.asarray(directions, dtype=np.float64)
    h, w = panorama.shape[:2]
    x = (np.arctan2(d[..., 0], d[..., 2])/(2*np.pi)+.5)*w-.5
    y = (np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2]))/np.pi+.5)*h-.5
    y = np.clip(y, 0, h-1)
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    dx = (x-ix).astype(np.float32)
    dy = (y-iy).astype(np.float32)
    if panorama.ndim == 3:
        dx, dy = dx[..., None], dy[..., None]
    a = panorama[iy, ix % w].astype(np.float32)
    b = panorama[iy, (ix+1) % w].astype(np.float32)
    c = panorama[np.minimum(iy+1, h-1), ix % w].astype(np.float32)
    e = panorama[np.minimum(iy+1, h-1), (ix+1) % w].astype(np.float32)
    return (a*(1-dx)+b*dx)*(1-dy)+(c*(1-dx)+e*dx)*dy


def split_face(rgb, radial, cam):
    local, world = view_rays(cam)
    image = np.clip(np.rint(sample_erp(rgb, world)), 0, 255).astype(np.uint8)
    ranges = sample_erp(radial, world)
    camera_z = ranges * local[..., 2]
    if not np.all(np.isfinite(camera_z) & (camera_z > 0)):
        raise ValueError('Invalid depth is not silently filled')
    return image, camera_z.astype(np.float32), ranges.astype(np.float32)


def angular_weights(world_directions, face_index, fov=100.0):
    """Partition of unity over the overlapping six views, no pose/depth fitting.

    A separable raised-cosine taper from 40 to 50 degrees for the 100-degree
    views. Normalization accounts for two-face edges and three-face corners.
    """
    d = np.asarray(world_directions, dtype=np.float64)
    weights = []
    outer = math.tan(math.radians(fov/2))
    inner = math.tan(math.radians(90-fov/2))
    for i in range(6):
        rot = np.asarray(camera(i, fov=fov)['c2w'])[:3,:3]
        local = d @ rot
        uv = np.abs(local[..., :2]) / np.maximum(local[..., 2:3], 1e-12)
        t = np.clip((uv-inner)/(outer-inner), 0, 1)
        weight = np.prod(0.5*(1+np.cos(np.pi*t)), axis=-1)
        weights.append(np.where(local[..., 2] > 0, weight, 0))
    denom = sum(weights)
    if np.any(denom < 1e-8):
        raise ValueError('Six-face coverage gap')
    return (weights[face_index]/denom).astype(np.float32)


def feather_alpha(alpha, weight):
    """Partition optical thickness, rather than simply multiplying opacity."""
    return -np.expm1(weight * np.log1p(-np.clip(alpha, 0, 1-1e-7)))
