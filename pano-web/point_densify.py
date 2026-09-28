"""Distance-adaptive interpolation preview only; never a SHARP model input."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

import point_mask as mask

VERSION = 1
PARAMETERS = dict(start_distance_m=2.0, max_local_multiplier=4,
                  max_extra_fraction=.25, target_spacing_m=.0075,
                  max_edge_m=.10, max_range_jump_m=.10,
                  max_relative_range_jump=.05, seed=20260920)
FULL = 'cloud-dense-added.bin'
SAMPLE = 'cloud-dense-added-preview.bin'
INDEX = 'cloud-dense-sources.bin'
MANIFEST = 'point-densify.json'
UV = np.array([[.5, 0], [0, .5], [.5, .5]])
BLEND = np.array([[.5, .5, 0, 0], [.5, 0, .5, 0], [.25, .25, .25, .25]])
RANGE_BINS = [0, 2, 4, 8, np.inf]


def source_signature(job):
    sources = mask.source_signature(job)
    for name in (mask.FILE, mask.MANIFEST):
        stat = (job/name).stat()
        sources[name] = [stat.st_size, stat.st_mtime_ns]
    return sources


def cached_result(job):
    try:
        info = json.loads((job/MANIFEST).read_text(encoding='utf-8'))
        current_mask = mask.cached_result(job)
        if (info['version'] != VERSION or info['parameters'] != PARAMETERS
                or not current_mask or info['mask_sha256'] != current_mask['sha256']
                or info['sources'] != source_signature(job)):
            return None
        for name, size in ((FULL, info['added']*24), (SAMPLE, info['preview_added']*24),
                           (INDEX, info['added']*4)):
            stat = (job/name).stat()
            if stat.st_size != size or info['files'][name]['mtime_ns'] != stat.st_mtime_ns:
                return None
        return info
    except (OSError, ValueError, KeyError, TypeError):
        return None


def cell_weights(points, keep, center=(0, 0, 0)):
    """Three possible new samples per 2x2 cell, no mixing faces or layers.

    Distant cells receive a higher budget, but sufficiently dense axes do not.
    Four retained corners, short 3D edges and consistent ranges are mandatory.
    These are conservative discontinuity tests, not proof of a visible surface.
    """
    points = np.asarray(points, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3 or keep.shape != points.shape[:2]:
        raise ValueError('需要一层 H×W 点云及对应 Mask')
    a, b, c, d = points[:-1, :-1], points[:-1, 1:], points[1:, :-1], points[1:, 1:]
    corners = np.stack((a, b, c, d), axis=-2)
    valid = keep[:-1, :-1] & keep[:-1, 1:] & keep[1:, :-1] & keep[1:, 1:]
    valid &= np.isfinite(corners).all(axis=(-2, -1))
    ranges = np.linalg.norm(corners-np.asarray(center), axis=-1)
    near = ranges.min(axis=-1)
    valid &= near > PARAMETERS['start_distance_m']
    jump = ranges.max(axis=-1)-near
    valid &= jump <= np.minimum(PARAMETERS['max_range_jump_m'],
                                near*PARAMETERS['max_relative_range_jump'])
    horizontal = np.stack((np.linalg.norm(a-b, axis=-1), np.linalg.norm(c-d, axis=-1)))
    vertical = np.stack((np.linalg.norm(a-c, axis=-1), np.linalg.norm(b-d, axis=-1)))
    valid &= np.maximum(horizontal.max(axis=0), vertical.max(axis=0)) <= PARAMETERS['max_edge_m']
    # Up to 2x along each axis => at most three extra samples per original cell.
    sx = np.clip(horizontal.mean(axis=0)/PARAMETERS['target_spacing_m'], 1, 2)-1
    sy = np.clip(vertical.mean(axis=0)/PARAMETERS['target_spacing_m'], 1, 2)-1
    weights = np.stack((sx, sy, sx*sy), axis=-1)
    extra = weights.sum(axis=-1)
    distance_extra = np.clip((near/PARAMETERS['start_distance_m'])**2,
                             1, PARAMETERS['max_local_multiplier'])-1
    weights *= np.minimum(1, distance_extra/np.maximum(extra, 1e-12))[..., None]
    return np.where(valid[..., None], weights, 0).astype(np.float32)


def best_candidates(keys, ids, count):
    if count <= 0:
        return keys[:0], ids[:0]
    if len(keys) <= count:
        return keys, ids
    selected = np.argpartition(keys, count-1)[:count]
    return keys[selected], ids[selected]


def candidate_keys(weights, parent_offset, width, rng):
    """Weighted reservoir priorities: higher weight => more likely under cap."""
    row, col, slot = np.nonzero(weights > 0)
    weight = weights[row, col, slot].astype(np.float64)
    keys = -np.log(np.maximum(rng.random(len(weight)), np.finfo(float).tiny))/weight
    ids = ((parent_offset+row*width+col)*3+slot).astype('<u4')
    return keys, ids


def interpolate_children(means, colors, local_ids, cam, width):
    """Subpixel rays + interpolated inverse camera-Z; color in linear RGB.

    Inverse-Z interpolation is perspective-correct for a local planar patch,
    without identifying a semantic plane or fitting one to the scene.
    """
    parent, slot = local_ids.astype(np.int64)//3, local_ids.astype(np.int64)%3
    indices = parent[:, None]+np.array([0, 1, width, width+1])
    transform = np.asarray(cam['c2w'], dtype=np.float64)
    corners = (means[indices]-transform[:3, 3]) @ transform[:3, :3]
    z = corners[..., 2]
    if not np.isfinite(corners).all() or (z <= 0).any():
        raise ValueError('新增点的父邻域不在有效相机空间')
    blend = BLEND[slot]
    inverse_z = np.sum(blend/z, axis=1)
    xy = np.sum(blend[..., None]*corners[..., :2]/z[..., None], axis=1)
    local = np.column_stack((xy, np.ones(len(xy))))/inverse_z[:, None]
    xyz = local @ transform[:3, :3].T + transform[:3, 3]
    linear = np.sum(colors[indices]*blend[..., None], axis=1)
    srgb = np.clip(np.where(linear <= .0031308, linear*12.92,
                           1.055*np.maximum(linear, 0)**(1/2.4)-.055), 0, 1)
    result = np.column_stack((xyz, srgb)).astype('<f4')
    if not np.isfinite(result).all():
        raise ValueError('加密插值产生非有限数值')
    return result


def build_densification(job, progress=print):
    job = Path(job)
    existing = cached_result(job)
    if existing:
        return existing
    start = time.perf_counter()
    progress('准备既有 82° / K=64 / 15 cm Mask；只建立第三步加密预览')
    mask_info = mask.build_mask(job, progress)
    signature = source_signature(job)
    stage = json.loads((job/'stage-3.json').read_text(encoding='utf-8'))
    cams = json.loads((job/'cameras.json').read_text(encoding='utf-8'))
    h, w = stage['diagnostics']['initializer_grid']
    layers = stage['layers']
    face_size = h*w*layers
    if len(cams)*face_size*3 > np.iinfo(np.uint32).max:
        raise ValueError('加密源索引超出 uint32 容量')
    budget = int(mask_info['kept']*PARAMETERS['max_extra_fraction'])
    keys, ids = np.empty(0), np.empty(0, dtype='<u4')
    expected = 0.
    candidates = 0
    before_hist = np.zeros(4, dtype=np.int64)
    for i, cam in enumerate(cams):
        progress(f'分析加密区域 {i+1}/{len(cams)}：{cam["face"]}，双层独立、禁止跨 Mask 插值')
        with np.load(job/'base'/f'{i}.npz') as base:
            means = base['means']
        keep = mask.load_face(job, i, len(means)).reshape(layers, h, w)
        grid = means.reshape(layers, h, w, 3)
        center = np.asarray(cam['c2w'])[:3, 3]
        before_hist += np.histogram(np.linalg.norm(means[keep.ravel()]-center, axis=1), RANGE_BINS)[0]
        for layer in range(layers):
            weights = cell_weights(grid[layer], keep[layer], center)
            expected += float(weights.sum(dtype=np.float64))
            rng = np.random.default_rng(PARAMETERS['seed']+i*layers+layer)
            local_keys, local_ids = candidate_keys(weights, i*face_size+layer*h*w, w, rng)
            candidates += len(local_ids)
            keys, ids = best_candidates(np.concatenate((keys, local_keys)),
                                        np.concatenate((ids, local_ids)), budget)
    wanted = min(budget, int(np.floor(expected)), len(ids))
    _, ids = best_candidates(keys, ids, wanted)
    ids.sort()  # Deterministic face/layer/pixel/slot order, not a random permutation.
    del keys
    sample_indices = np.sort(np.random.default_rng(PARAMETERS['seed']).choice(
        len(ids), len(ids)//16, replace=False))
    files = (FULL, SAMPLE, INDEX)
    partials = {name: job/(name+'.partial') for name in files}
    after_hist = np.zeros(4, dtype=np.int64)
    written = 0
    face_stats = []
    try:
        with partials[FULL].open('wb') as output, partials[SAMPLE].open('wb') as sampled:
            for i, cam in enumerate(cams):
                chosen = ids[(ids//(3*face_size)) == i]
                local_ids = chosen-i*face_size*3
                progress(f'反投影新增点 {i+1}/{len(cams)}：{len(chosen):,} 个')
                with np.load(job/'base'/f'{i}.npz') as base:
                    means, colors = base['means'], base['colors']
                if means.shape != (face_size, 3) or colors.shape != means.shape:
                    raise ValueError('初始化坐标/颜色与面层网格不一致')
                keep = mask.load_face(job, i, face_size)
                parent = local_ids.astype(np.int64)//3
                if not keep[parent[:, None]+np.array([0, 1, w, w+1])].all():
                    raise ValueError('新增点引用了被 Mask 删除的点')
                # Chunking bounds transient interpolation memory on large panoramas.
                for first in range(0, len(local_ids), 65536):
                    chunk_ids = local_ids[first:first+65536]
                    points = interpolate_children(means, colors, chunk_ids, cam, w)
                    points.tofile(output)
                    sample = sample_indices[(sample_indices >= written) & (sample_indices < written+len(points))]-written
                    points[sample].tofile(sampled)
                    center = np.asarray(cam['c2w'])[:3, 3]
                    after_hist += np.histogram(np.linalg.norm(points[:, :3]-center, axis=1), RANGE_BINS)[0]
                    written += len(points)
                face_stats.append(dict(face=cam['face'], added=len(chosen)))
        ids.tofile(partials[INDEX])
        if written != wanted or source_signature(job) != signature:
            raise ValueError('加密数量或源文件状态在计算中发生变化')
        expected_sizes = {FULL: wanted*24, SAMPLE: len(sample_indices)*24, INDEX: wanted*4}
        for name in files:
            if partials[name].stat().st_size != expected_sizes[name]:
                raise ValueError('加密预览文件长度错误')
        for name in files:
            partials[name].replace(job/name)
        info = dict(version=VERSION, parameters=PARAMETERS, sources=signature,
                    mask_sha256=mask_info['sha256'], original_count=mask_info['count'],
                    before=mask_info['kept'], added=wanted, after=mask_info['kept']+wanted,
                    preview_added=len(sample_indices), budget=budget,
                    candidate_count=candidates, expected_before_budget=expected,
                    full=FULL, preview=SAMPLE, source_indices=INDEX,
                    source_index_encoding='((face * layers + layer) * H * W + row * W + col) * 3 + slot',
                    slots=UV.tolist(), faces=face_stats, seed=PARAMETERS['seed'],
                    range_labels=['0–2 m', '2–4 m', '4–8 m', '8 m+'],
                    before_by_range=before_hist.tolist(), added_by_range=after_hist.tolist(),
                    files={name: dict(bytes=(job/name).stat().st_size,
                          mtime_ns=(job/name).stat().st_mtime_ns, sha256=mask.digest(job/name)) for name in files},
                    seconds=time.perf_counter()-start, preview_only=True, downstream=False,
                    method='new subpixel rays with interpolated inverse camera-Z; no new measured detail',
                    safety='four retained same-face/same-layer corners, bounded edges and range jumps')
        temp = job/(MANIFEST+'.partial')
        temp.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(job/MANIFEST)
        progress(f'点云加密预览完成：{info["before"]:,} → {info["after"]:,}；后续高斯未改动')
        return info
    finally:
        for partial in partials.values():
            partial.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    build_densification(args.job, lambda message: print(message, flush=True))
