"""Orient stage-5 Gaussians to reliable local depth surfaces, without resizing."""
import numpy as np
from scipy.spatial.transform import Rotation
from point_normals import PARAMETERS, estimate_angles

VERSION = 'local-depth-normal-short-axis-v1'


def align_short_axes(scene, base, cam, width):
    """Apply the smallest rotation taking the shortest axis onto the surface normal.

    Surface normals use the existing initializer grid and reliability rules.
    Each layer is independent. Unreliable rows and all non-rotation fields are
    retained exactly; normal sign is chosen to avoid unnecessary half-turns.
    """
    grids = np.asarray(base['means']).reshape(-1, width, width, 3)
    center = np.asarray(cam['c2w'])[:3, 3]
    quats = scene['quats'].copy()
    count = width * width
    reliable_count = 0
    max_error = 0.0
    for layer, grid in enumerate(grids):
        angles, normals = estimate_angles(grid, center, **PARAMETERS, return_normals=True)
        local_ids = np.flatnonzero(angles.ravel() >= 0)
        normals = normals.reshape(-1, 3)
        reliable_count += len(local_ids)
        for first in range(0, len(local_ids), 65536):
            local = local_ids[first:first + 65536]
            ids = local + layer * count
            old = Rotation.from_quat(quats[ids][:, [1, 2, 3, 0]])
            shortest = np.argmin(scene['scales'][ids], axis=1)
            axes = np.take_along_axis(old.as_matrix(), shortest[:, None, None], axis=2)[:, :, 0]
            target = normals[local].copy()
            dot = np.sum(axes * target, axis=1)
            target[dot < 0] *= -1
            dot = np.clip(np.abs(dot), 0, 1)
            delta = np.column_stack((np.cross(axes, target), 1 + dot))
            aligned = Rotation.from_quat(delta) * old
            quats[ids] = aligned.as_quat()[:, [3, 0, 1, 2]].astype(quats.dtype)
            actual = np.take_along_axis(
                Rotation.from_quat(quats[ids][:, [1, 2, 3, 0]]).as_matrix(),
                shortest[:, None, None], axis=2)[:, :, 0]
            error = np.linalg.norm(np.cross(actual, target), axis=1)
            max_error = max(max_error, float(error.max()))
    return {**scene, 'quats': quats}, dict(
        version=VERSION, source='stage-3 depth initializer, per layer and source pixel',
        parameters=dict(PARAMETERS), reliable=reliable_count,
        unchanged_unreliable=len(quats)-reliable_count,
        max_alignment_error_degrees=float(np.degrees(np.arcsin(min(1., max_error)))),
        changed_fields=['quats'])
