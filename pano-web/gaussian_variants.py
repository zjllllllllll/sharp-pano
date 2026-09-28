"""Publish baseline and masked PLYs, retaining original Gaussian attributes."""
from pathlib import Path
import numpy as np


def save_variants(job, filename, vertices, keep, mask_info):
    from plyfile import PlyData
    from gaussian_ops import save_vertices, digest
    job = Path(job)
    if keep.dtype != np.bool_ or keep.shape != (len(vertices),):
        raise ValueError('导出高斯与 Mask 索引不一致')
    target = job/filename
    if target.exists():
        # Backfilling a completed stage must not silently replace its baseline.
        baseline = PlyData.read(target, mmap='r')['vertex'].data
        try:
            if len(baseline) != len(vertices) or baseline.dtype != vertices.dtype:
                raise ValueError('已有未筛选 PLY 与当前模型输出不匹配；保留原文件，请用新任务生成')
            for start in range(0, len(vertices), 65536):
                if not np.array_equal(baseline[start:start+65536], vertices[start:start+65536]):
                    raise ValueError('已有未筛选 PLY 与当前模型输出不一致；保留原文件，请用新任务生成')
        finally:
            del baseline
        original_info = dict(bytes=target.stat().st_size, sha256=digest(target))
    else:
        target.with_suffix('.partial.ply').unlink(missing_ok=True)
        original_info = save_vertices(target, vertices)
    masked_name = f'{target.stem}-masked.ply'
    staging = job/f'{target.stem}-masked.new.ply'
    # These are only our own unpublished staging files from an interrupted retry.
    for path in (staging, staging.with_suffix('.partial.ply')):
        path.unlink(missing_ok=True)
    try:
        masked_info = save_vertices(staging, vertices[keep], allow_empty=True)
        staging.replace(job/masked_name)
    finally:
        for path in (staging, staging.with_suffix('.partial.ply')):
            path.unlink(missing_ok=True)
    def variant(name, count, info):
        return dict(preview=name, download=name, count=count, bytes=info['bytes'], sha256=info['sha256'])
    return dict(unmasked=variant(filename, len(vertices), original_info),
                masked=variant(masked_name, int(keep.sum()), masked_info)), dict(
                    parameters=mask_info['parameters'], sha256=mask_info['sha256'],
                    point_count=mask_info['count'], point_kept=mask_info['kept'],
                    point_removed=mask_info['removed'], gaussian_removed=int((~keep).sum()),
                    index=mask_info['index'], outside_cloud_policy=mask_info['outside_cloud_policy'])
