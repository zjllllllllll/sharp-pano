"""Fixed point-domain mask, indexed by original face/layer/pixel, not moved XYZ."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import point_distances as distances
import point_normals as normals

VERSION = 1
NEIGHBORS = 64
DISTANCE_METERS = 0.15
FILE = 'point-mask.npz'
MANIFEST = 'point-mask.json'
PARAMETERS = dict(normal_angle_degrees=normals.FIXED_NORMAL_ANGLE,
                  neighbors=NEIGHBORS, distance_meters=DISTANCE_METERS, iterations=1)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_signature(job):
    sources = distances.source_signature(job)
    for name in (distances.FULL, distances.MANIFEST):
        stat = (job/name).stat()
        sources[name] = [stat.st_size, stat.st_mtime_ns]
    return sources


def cached_result(job):
    try:
        info = json.loads((job/MANIFEST).read_text(encoding='utf-8'))
        if (info['version'] != VERSION or info['parameters'] != PARAMETERS
                or not distances.cached_result(job) or info['sources'] != source_signature(job)
                or info['sha256'] != digest(job/FILE)):
            return None
        return info
    except (OSError, ValueError, KeyError, TypeError):
        return None


def fixed_keep(angles, mean_distance):
    """Same float32 boundary as the point viewer; equality is retained."""
    if angles.shape != mean_distance.shape:
        raise ValueError('角度与距离数量不匹配')
    normal = distances.normal_keep(angles)
    if not np.isfinite(mean_distance[normal]).all() or (mean_distance[normal] < 0).any():
        raise ValueError('法向保留点缺少有效近邻距离')
    return normal & (mean_distance <= np.float32(DISTANCE_METERS))


def build_mask(job, progress=print):
    job = Path(job)
    existing = cached_result(job)
    if existing:
        return existing
    info = distances.build_distances(job, progress)
    signature = source_signature(job)
    angles = np.fromfile(job/normals.FULL, dtype='<f4')
    scores = np.memmap(job/distances.FULL, dtype='<f4', mode='r').reshape(-1, len(distances.NEIGHBORS))
    keep = fixed_keep(angles, scores[:, distances.NEIGHBORS.index(NEIGHBORS)])
    full = np.memmap(job/'cloud-full.bin', dtype='<f4', mode='r').reshape(-1, 6)
    if len(full) != len(keep) or len(full) != info['count']:
        raise ValueError('Mask 与第三步点云数量不一致')
    cams = json.loads((job/'cameras.json').read_text(encoding='utf-8'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sharp-sixface'))
    from geometry import angular_weights
    masks, faces, offset = {}, [], 0
    for i, cam in enumerate(cams):
        with np.load(job/'base'/f'{i}.npz') as base:
            means = base['means']
        domain = angular_weights(means, i, cam['hfov_deg']) > 1e-4
        count = int(domain.sum())
        if not np.array_equal(means[domain], full[offset:offset+count, :3]):
            raise ValueError('原始面/层/像素索引与点云不匹配，拒绝传播 Mask')
        # Outside the original point-cloud domain stays excluded, even if a
        # learned offset later moves that Gaussian inside a face's valid region.
        face_keep = np.zeros(len(means), dtype=bool)
        face_keep[domain] = keep[offset:offset+count]
        masks[f'face_{i}'] = face_keep
        faces.append(dict(face=cam['face'], source_count=len(means), cloud_count=count,
                          kept=int(face_keep.sum()), removed=count-int(face_keep.sum()),
                          outside_cloud=int((~domain).sum())))
        offset += count
    if offset != len(keep):
        raise ValueError('Mask 面索引总数不匹配')
    if source_signature(job) != signature:
        raise ValueError('构建 Mask 期间源文件变化，请重试')
    partial = job/(FILE+'.partial')
    try:
        with partial.open('wb') as output:
            np.savez(output, **masks)
        partial.replace(job/FILE)
    finally:
        partial.unlink(missing_ok=True)
    result = dict(version=VERSION, parameters=PARAMETERS, sources=signature, file=FILE,
                  sha256=digest(job/FILE), count=len(keep), kept=int(keep.sum()),
                  removed=int((~keep).sum()), normal_removed=int((~distances.normal_keep(angles)).sum()),
                  faces=faces, index='face -> layer -> row -> column (original initializer)',
                  outside_cloud_policy='exclude', downstream=True)
    partial = job/(MANIFEST+'.partial')
    partial.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    partial.replace(job/MANIFEST)
    progress(f'Mask 固定 82° / K=64 / 15 cm：保留 {result["kept"]:,}，删除 {result["removed"]:,} 点')
    return result


def load_face(job, index, expected_count):
    with np.load(Path(job)/FILE) as data:
        keep = data[f'face_{index}']
    if keep.dtype != np.bool_ or keep.shape != (expected_count,):
        raise ValueError('Mask 与原始高斯索引数量不一致')
    return keep
