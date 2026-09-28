"""Preview-only organized-cloud normals. Never rewrites base/LAS/PLY geometry."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

VERSION = 1
FIXED_NORMAL_ANGLE = 82.0  # Confirmed threshold in degrees (not a percentage).
PARAMETERS = dict(window=5, min_neighbors=6, max_relative_range_jump=0.10,
                  max_plane_ratio=0.25, min_tangent_ratio=1e-6)
FULL = 'cloud-angles-full.bin'
SAMPLE = 'cloud-angles-preview.bin'
MANIFEST = 'point-normals.json'


def source_signature(job):
    names = ['stage-3.json', 'cameras.json', 'cloud-full.bin', 'cloud-preview.bin']
    names += [f'base/{i}.npz' for i in range(6)]
    return {name: [p.stat().st_size, p.stat().st_mtime_ns]
            for name in names for p in [job / name]}


def cached_result(job):
    try:
        info = json.loads((job / MANIFEST).read_text(encoding='utf-8'))
        if (info['version'] != VERSION or info['parameters'] != PARAMETERS
                or info['sources'] != source_signature(job)):
            return None
        for name, count in [(FULL, info['count']), (SAMPLE, info['preview_count'])]:
            if (job / name).stat().st_size != count * 4:
                return None
        return info
    except (OSError, ValueError, KeyError, TypeError):
        return None


def estimate_angles(points, camera_center=(0, 0, 0), *, window=5,
                    min_neighbors=6, max_relative_range_jump=0.10,
                    max_plane_ratio=0.25, min_tangent_ratio=1e-6, chunk_rows=24):
    """Return HxW acute normal/ray angles in degrees; -1 means unreliable.

    Fit a plane to XYZ values in each pixel window, independently per layer.
    Exclude nonfinite samples and large radial-depth jumps. The covariance is
    accumulated relative to the query point for numerical stability. Reject
    nonplanar and line/point-degenerate neighborhoods. No filling across edges.
    The sign of the eigenvector is irrelevant because we use abs(dot(n, ray)).
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 3 or p.shape[2] != 3 or window < 3 or window % 2 != 1:
        raise ValueError('Expected an H x W x 3 grid and an odd window >= 3')
    if chunk_rows < 1:
        raise ValueError('chunk_rows must be positive')
    h, w, _ = p.shape
    rays = p - np.asarray(camera_center, dtype=np.float64)
    ranges = np.linalg.norm(rays, axis=2)
    valid = np.isfinite(p).all(axis=2) & np.isfinite(ranges) & (ranges > 1e-8)
    radius = window // 2
    padded = np.pad(p, ((radius, radius), (radius, radius), (0, 0)),
                    constant_values=np.nan)
    padded_range = np.pad(ranges, radius, constant_values=np.nan)
    result = np.full((h, w), -1, dtype=np.float32)
    for first in range(0, h, chunk_rows):
        last = min(first + chunk_rows, h)
        center = p[first:last]
        depth = ranges[first:last]
        good = valid[first:last]
        shape = (last - first, w)
        counts = np.zeros(shape, dtype=np.int32)
        sums = np.zeros((*shape, 3), dtype=np.float64)
        products = np.zeros((*shape, 3, 3), dtype=np.float64)
        for dy in range(window):
            for dx in range(window):
                neighbor = padded[first+dy:last+dy, dx:dx+w]
                distance = padded_range[first+dy:last+dy, dx:dx+w]
                include = (good & np.isfinite(neighbor).all(axis=2)
                           & (distance > 1e-8)
                           & (np.abs(distance-depth) <= max_relative_range_jump*depth))
                delta = np.where(include[..., None], neighbor-center, 0)
                counts += include
                sums += delta
                for a in range(3):
                    for b in range(a, 3):
                        products[..., a, b] += delta[..., a] * delta[..., b]
        divisor = np.maximum(counts, 1)
        mean = sums / divisor[..., None]
        cov = products / divisor[..., None, None]
        for a in range(3):
            for b in range(a, 3):
                cov[..., a, b] -= mean[..., a] * mean[..., b]
                cov[..., b, a] = cov[..., a, b]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        e = np.maximum(eigenvalues, 0)
        reliable = (good & (counts >= min_neighbors) & (e[..., 2] > 0)
                    & (e[..., 1] > min_tangent_ratio*e[..., 2])
                    & (e[..., 0] <= max_plane_ratio*e[..., 1]))
        normals = eigenvectors[..., :, 0]
        ray = np.divide(rays[first:last], depth[..., None],
                        out=np.zeros_like(center), where=good[..., None])
        cosine = np.clip(np.abs(np.sum(normals * ray, axis=2)), 0, 1)
        angles = np.rad2deg(np.arccos(cosine))
        result[first:last] = np.where(reliable, angles, -1).astype(np.float32)
    return result


def build_angles(job, progress=print):
    """Backfill old runs or prepare new ones without any model inference."""
    job = Path(job)
    existing = cached_result(job)
    if existing:
        return existing
    start = time.perf_counter()
    signature = source_signature(job)
    stage = json.loads((job / 'stage-3.json').read_text(encoding='utf-8'))
    cameras = json.loads((job / 'cameras.json').read_text(encoding='utf-8'))
    h, w = stage['diagnostics']['initializer_grid']
    layers = stage['layers']
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sharp-sixface'))
    from geometry import angular_weights
    # Memory-map only the previews; read one face's means from the NPZ at a time.
    full_xyz = np.memmap(job / 'cloud-full.bin', dtype='<f4', mode='r').reshape(-1, 6)
    sample_xyz = np.memmap(job / 'cloud-preview.bin', dtype='<f4', mode='r').reshape(-1, 6)
    full_part, sample_part = job / (FULL+'.partial'), job / (SAMPLE+'.partial')
    offset = sample_offset = unknown = 0
    faces = []
    histogram = np.zeros(181, dtype=np.int64)
    try:
        with full_part.open('wb') as full_out, sample_part.open('wb') as sample_out:
            for i, cam in enumerate(cameras):
                progress(f'法向估计 {i+1}/6：{cam["face"]} · 每层独立 5×5 平面拟合')
                with np.load(job / 'base' / f'{i}.npz') as base:
                    means = base['means']
                grids = means.reshape(layers, h, w, 3)
                center = np.asarray(cam['c2w'], dtype=np.float64)[:3, 3]
                angle_grid = np.stack([estimate_angles(grid, center, **PARAMETERS)
                                       for grid in grids])
                keep = angular_weights(means, i, cam['hfov_deg']) > 1e-4
                xyz = means[keep]
                angles = angle_grid.reshape(-1)[keep]
                count = len(angles)
                if not np.array_equal(full_xyz[offset:offset+count, :3], xyz):
                    raise ValueError('初始化缓存与全量点云顺序不匹配；拒绝使用错位法向')
                indices = np.random.default_rng(42+i).choice(count, count//16, replace=False)
                n_sample = len(indices)
                if not np.array_equal(sample_xyz[sample_offset:sample_offset+n_sample, :3], xyz[indices]):
                    raise ValueError('抽样点云与初始化缓存顺序不匹配；拒绝使用错位法向')
                angles.astype('<f4').tofile(full_out)
                angles[indices].astype('<f4').tofile(sample_out)
                valid = angles >= 0
                n_unknown = int((~valid).sum())
                histogram += np.bincount(np.ceil(angles[valid]*2).astype(int), minlength=181)
                faces.append(dict(face=cam['face'], count=count, unreliable=n_unknown))
                offset += count
                sample_offset += n_sample
                unknown += n_unknown
        if offset != len(full_xyz) or sample_offset != len(sample_xyz):
            raise ValueError('点云数量与法向数量不一致')
        if offset != stage['count'] or sample_offset != stage['preview_count']:
            raise ValueError('点云数量与步骤记录不一致')
        if source_signature(job) != signature:
            raise ValueError('计算过程中点云发生变化，请重试')
        full_part.replace(job / FULL)
        sample_part.replace(job / SAMPLE)
        info = dict(version=VERSION, parameters=PARAMETERS, sources=signature,
                    full=FULL, preview=SAMPLE, count=offset, preview_count=sample_offset,
                    unreliable=unknown, faces=faces, seconds=time.perf_counter()-start,
                    histogram_half_degree=histogram.tolist(),
                    angle='degrees(acos(abs(dot(unit_normal, unit_ray_from_original_camera))))',
                    invalid_value=-1, unreliable_policy='keep visible', preview_only=True)
        manifest_part = job / (MANIFEST+'.partial')
        manifest_part.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest_part.replace(job / MANIFEST)
        progress(f'法向估计完成：{offset:,} 点，{unknown:,} 个法向不可靠（保留显示）')
        return info
    finally:
        full_part.unlink(missing_ok=True)
        sample_part.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    build_angles(args.job, progress=lambda message: print(message, flush=True))
