"""One-pass KNN scores AFTER the fixed normal mask, for preview only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import point_normals as normals

VERSION = 1
NEIGHBORS = [4, 8, 16, 32, 64]
FULL = 'cloud-distances-full.bin'
SAMPLE = 'cloud-distances-preview.bin'
MANIFEST = 'point-distances.json'
NO_NEIGHBOR = np.finfo(np.float32).max


def source_signature(job):
    sources = normals.source_signature(job)
    for name in (normals.FULL, normals.SAMPLE, normals.MANIFEST):
        stat = (job / name).stat()
        sources[name] = [stat.st_size, stat.st_mtime_ns]
    return sources


def cached_result(job):
    try:
        info = json.loads((job / MANIFEST).read_text(encoding='utf-8'))
        if (info['version'] != VERSION or info['neighbors'] != NEIGHBORS
                or info['normal_angle_degrees'] != normals.FIXED_NORMAL_ANGLE
                or not normals.cached_result(job)
                or info['sources'] != source_signature(job)):
            return None
        for name, count in [(FULL, info['count']), (SAMPLE, info['preview_count'])]:
            if (job / name).stat().st_size != count * len(NEIGHBORS) * 4:
                return None
        return info
    except (OSError, ValueError, KeyError, TypeError):
        return None


def normal_keep(angles):
    # Same convention as the normal preview: unreliable angles remain candidates.
    return (~np.isfinite(angles) | (angles < 0) | (angles > 90)
            | (angles <= normals.FIXED_NORMAL_ANGLE))


def iter_neighbor_means(points, neighbors=NEIGHBORS, chunk_size=16384, workers=4):
    """Exact Euclidean KNN, excluding the query's own index (including ties).

    Other coincident points are legitimate neighbors. With fewer than K other
    points, average the available neighbors. A lone point gets NO_NEIGHBOR.
    Every query uses the same fixed normal-filtered reference cloud, not an
    iteratively eroded cloud and not the 1/16 display sample.
    """
    from scipy.spatial import cKDTree
    xyz = np.ascontiguousarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite N x 3 coordinates')
    if not neighbors or min(neighbors) < 1 or chunk_size < 1:
        raise ValueError('Neighbors and chunk size must be positive')
    n = len(xyz)
    if not n:
        return
    if n == 1:
        yield 0, np.full((1, len(neighbors)), NO_NEIGHBOR, dtype=np.float32)
        return
    tree = cKDTree(xyz)
    maximum = min(max(neighbors), n-1)
    columns = np.arange(maximum)[None, :]
    effective = np.minimum(neighbors, n-1)
    for start in range(0, n, chunk_size):
        stop = min(start+chunk_size, n)
        distance, indices = tree.query(xyz[start:stop], k=maximum+1, workers=workers)
        is_self = indices == np.arange(start, stop)[:, None]
        position = np.where(is_self.any(axis=1), is_self.argmax(axis=1), maximum+1)
        # If a duplicate tie excludes self from the returned set, take the first K.
        take = columns + (columns >= position[:, None])
        distance = np.take_along_axis(distance, take, axis=1)
        cumulative = np.cumsum(distance, axis=1)
        yield start, (cumulative[:, effective-1] / effective).astype(np.float32)


def build_distances(job, progress=print):
    job = Path(job)
    existing = cached_result(job)
    if existing:
        return existing
    start = time.perf_counter()
    normal_info = normals.build_angles(job, progress)
    signature = source_signature(job)
    full = np.memmap(job/'cloud-full.bin', dtype='<f4', mode='r').reshape(-1, 6)
    sample = np.memmap(job/'cloud-preview.bin', dtype='<f4', mode='r').reshape(-1, 6)
    angles = np.memmap(job/normals.FULL, dtype='<f4', mode='r')
    if len(full) != len(angles) or len(full) != normal_info['count']:
        raise ValueError('法向与原始点云数量不一致')
    kept_indices = np.flatnonzero(normal_keep(angles))
    reference = np.ascontiguousarray(full[kept_indices, :3], dtype=np.float64)
    progress(f'固定 {normals.FIXED_NORMAL_ANGLE:g}°：保留 {len(reference):,} / {len(full):,} 点；建立三维近邻索引')
    full_part, sample_part = job/(FULL+'.partial'), job/(SAMPLE+'.partial')
    scores = None
    try:
        scores = np.memmap(full_part, dtype='<f4', mode='w+', shape=(len(full), len(NEIGHBORS)))
        scores[:] = -1  # Already removed by the normal mask; never queried.
        reported = -1
        for offset, values in iter_neighbor_means(reference):
            stop = offset+len(values)
            scores[kept_indices[offset:stop]] = values
            percent = int(100*stop/max(1, len(reference)))
            if percent//5 != reported or stop == len(reference):
                reported = percent//5
                progress(f'距离计算 {percent}% · {stop:,}/{len(reference):,} 点 · K=4/8/16/32/64')
        scores.flush()
        del reference
        # Regenerate the existing deterministic sample selection, with XYZ checks.
        full_offset = sample_offset = 0
        with sample_part.open('wb') as output:
            for i, face in enumerate(normal_info['faces']):
                count = face['count']
                indices = np.random.default_rng(42+i).choice(count, count//16, replace=False)
                if not np.array_equal(sample[sample_offset:sample_offset+len(indices), :3],
                                      full[full_offset+indices, :3]):
                    raise ValueError('抽样点云顺序不匹配；拒绝映射距离')
                np.asarray(scores[full_offset+indices], dtype='<f4').tofile(output)
                full_offset += count
                sample_offset += len(indices)
        if full_offset != len(full) or sample_offset != len(sample):
            raise ValueError('抽样点云数量不匹配')
        summary = {}
        for column, k in enumerate(NEIGHBORS):
            values = np.asarray(scores[kept_indices, column])
            summary[str(k)] = np.quantile(values, [.5, .9, .95, .99]).tolist() if len(values) else []
        # Close mappings before atomic rename on Windows.
        scores._mmap.close()
        scores = None
        if source_signature(job) != signature:
            raise ValueError('计算期间源数据发生变化，请重试')
        full_part.replace(job/FULL)
        sample_part.replace(job/SAMPLE)
        info = dict(version=VERSION, neighbors=NEIGHBORS, sources=signature,
                    normal_angle_degrees=normals.FIXED_NORMAL_ANGLE,
                    full=FULL, preview=SAMPLE, count=len(full), preview_count=len(sample),
                    normal_kept=len(kept_indices), normal_removed=len(full)-len(kept_indices),
                    metric='mean Euclidean distance in meters to K other points',
                    reference='full cloud AFTER fixed normal filter, both layers and all faces',
                    iterations=1, exclude_self=True, insufficient_neighbors='average available; lone point=FLOAT32_MAX',
                    layout='point-major float32, columns in neighbors order',
                    distance_quantiles_p50_p90_p95_p99_m=summary,
                    seconds=time.perf_counter()-start, preview_only=True)
        manifest_part = job/(MANIFEST+'.partial')
        manifest_part.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest_part.replace(job/MANIFEST)
        progress(f'距离缓存完成：{len(kept_indices):,} 点；滑块调整不再重新搜索近邻')
        return info
    finally:
        if scores is not None:
            scores._mmap.close()
        full_part.unlink(missing_ok=True)
        sample_part.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    build_distances(args.job, lambda message: print(message, flush=True))
