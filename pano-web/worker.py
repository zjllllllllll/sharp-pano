"""Depth backprojection and Gaussian stages; preserve six-face source indexing."""
from pathlib import Path
import argparse
import gc
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, str(ROOT / 'sharp-sixface'))
import numpy as np
from PIL import Image


def record(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def progress(message):
    print(message, flush=True)


def depth_stage(job):
    sys.path.insert(0, str(ROOT / 'third_party/DepthMaster'))
    import torch
    from infer_panoramas import load_verified_model, infer
    import cv2
    torch.set_num_threads(8)
    torch.manual_seed(0)
    progress('加载 DepthMaster 权重')
    model = load_verified_model(ROOT / 'models/depthmaster.pt', torch.device('cuda'))
    rgb = np.asarray(Image.open(job / 'input.png').convert('RGB'))
    progress('六面联合估深，融合为全景径向深度')
    depth, stats = infer(model, rgb)
    valid = np.isfinite(depth) & (depth > 0)
    if valid.mean() < 0.95:
        raise ValueError(f'有效深度比例仅 {valid.mean():.1%}，不足以可靠注入 SHARP。')
    np.save(job / 'depth.npy', depth, allow_pickle=False)
    # Keep the raw metric depth untouched. Fill rare invalid pixels only for
    # model conditioning; report this explicitly and persist a separate array.
    conditioned = depth.copy()
    if not valid.all():
        from scipy.ndimage import distance_transform_edt
        indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        conditioned[~valid] = depth[tuple(indices[:, ~valid])]
    np.save(job / 'conditioning.npy', conditioned, allow_pickle=False)
    lo, hi = np.quantile(depth[valid], [.01, .99])
    scaled = np.clip((np.nan_to_num(depth, nan=float(hi)) - lo) / max(hi-lo, 1e-6), 0, 1)
    color = cv2.applyColorMap(np.rint(scaled * 255).astype('uint8'), cv2.COLORMAP_TURBO)[..., ::-1]
    color[~valid] = 0
    Image.fromarray(color).save(job / 'depth.png')
    Image.fromarray(np.where(valid, np.rint((1-scaled)*255), 0).astype('uint8')).save(job / 'depth-gray.png')
    stats.update(valid_fraction=float(valid.mean()), filled_pixels=int((~valid).sum()),
                 range_min_m=float(depth[valid].min()), range_max_m=float(depth[valid].max()),
                 display_p01_p99_m=[float(lo), float(hi)], width=rgb.shape[1], height=rgb.shape[0])
    return dict(kind='image', image='depth.png', gray='depth-gray.png', download='depth.npy',
                description='DepthMaster 米制径向距离；预览按 P1–P99 拉伸，蓝近红远。注入使用浮点深度，不使用颜色图。', **stats)


def base_scene(initializer):
    """Decode exact initializer base values, NOT composer(zero_delta).

    The composer's inverse-depth epsilon and color clamps are intentionally
    absent here: this is the real pre-composer Gaussian initialization.
    """
    import torch
    from sharp.utils.color_space import sRGB2linearRGB
    b = initializer.gaussian_base_values
    z = b.mean_inverse_z_ndc.reciprocal()
    mean = torch.cat((b.mean_x_ndc*z, b.mean_y_ndc*z, z), dim=1)
    shape = mean.shape
    scale = b.scales.expand(shape)
    quat = b.quaternions.expand(shape[0], 4, *shape[2:])
    color = sRGB2linearRGB(b.colors.expand(shape))
    alpha = b.opacities.expand(shape[0], *shape[2:]).reshape(shape[0], -1)
    def flat(t):
        return t.permute(0, 2, 3, 4, 1).flatten(1, 3)
    mean, scale = flat(mean), flat(scale)
    if initializer.global_scale is not None:
        mean = mean * initializer.global_scale[:, None, None]
        scale = scale * initializer.global_scale[:, None, None]
    return {k: v[0].detach().float().cpu().numpy().copy() for k, v in
            [('means', mean), ('scales', scale), ('quats', flat(quat)), ('colors', flat(color)), ('alpha', alpha)]}


def to_srgb(color):
    return np.clip(np.where(color <= .0031308, color*12.92,
                           1.055*np.maximum(color, .0031308)**(1/2.4)-.055), 0, 1)


CLOUD_GRID_SIZE = 768


def direct_depth_scene(rgb, radial, cam, grid_size=None):
    """Backproject the same ERP range at every face ray; no learned depth.

    Retain the existing face/layer/pixel indexing and initial Gaussian fields.
    Both storage layers use the one observed surface, not a predicted back layer.
    """
    from geometry import view_rays, sample_erp
    from gaussian_ops import ndc_to_camera, world_transform
    n = grid_size or CLOUD_GRID_SIZE
    local, world = view_rays(cam, n)
    ranges = sample_erp(radial, world)
    z = ranges * local[..., 2]
    axis = (np.arange(n, dtype=np.float32) + .5) * (2.0/n) - 1
    x, y = np.meshgrid(axis, axis)
    means = np.stack((x*z, y*z, z), axis=-1).reshape(-1, 3)
    scales = np.repeat((z.reshape(-1, 1) * (2.0/n)), 3, axis=1)
    # Keep SHARP's original 2x2 RGB averaging and initial field conventions.
    _, rgb_rays = view_rays(cam, 2*n)
    face = np.clip(np.rint(sample_erp(rgb, rgb_rays)), 0, 255).astype(np.uint8)
    srgb = face.reshape(n, 2, n, 2, 3).mean(axis=(1, 3), dtype=np.float32) / 255
    linear = np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
    scene = dict(means=np.tile(means, (2, 1)).astype(np.float32),
                 scales=np.tile(scales, (2, 1)).astype(np.float32),
                 quats=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (2*n*n, 1)),
                 colors=np.tile(linear.reshape(-1, 3), (2, 1)).astype(np.float32),
                 alpha=np.full(2*n*n, .5, dtype=np.float32))
    scene, _ = ndc_to_camera(scene, cam)
    scene = world_transform(scene, cam)
    # Use the direct XYZ expression, avoiding any inverse-depth/pooling changes.
    scene['means'] = np.tile((world * ranges[..., None]).reshape(-1, 3), (2, 1))
    return scene, ranges


def cloud_stage(job):
    import laspy
    from geometry import camera, angular_weights
    progress('读取全景深度，直接反投影生成点云')
    rgb = np.asarray(Image.open(job / 'input.png').convert('RGB'))
    radial = np.load(job / 'depth.npy')
    cache = job / 'base'
    cache.mkdir(exist_ok=True)
    header = laspy.LasHeader(point_format=2, version='1.2')
    header.scales = np.array([.00001]*3)
    count = 0
    preview = []
    cams = []
    depth_stats = []
    with laspy.open(job / 'cloud.las', mode='w', header=header) as writer, (job/'cloud-full.bin').open('wb') as full:
        for i in range(6):
            cam = camera(i)
            progress(f'反投影 {i+1}/6：{cam["face"]} · 直接使用全景深度')
            scene, ranges = direct_depth_scene(rgb, radial, cam)
            if not all(np.isfinite(a).all() for a in scene.values()):
                raise ValueError('深度反投影初始化含非有限值')
            restored = np.linalg.norm(scene['means'].reshape(2, CLOUD_GRID_SIZE, CLOUD_GRID_SIZE, 3), axis=-1)
            depth_stats.append(dict(face=cam['face'],
                                   radial_error_max_m=float(np.max(np.abs(restored-ranges[None])))))
            np.savez(cache / f'{i}.npz', **scene)
            keep = angular_weights(scene['means'], i, cam['hfov_deg']) > 1e-4
            xyz = scene['means'][keep]
            colors = to_srgb(scene['colors'][keep])
            points = laspy.ScaleAwarePointRecord.zeros(len(xyz), header=header)
            points.x, points.y, points.z = xyz.T
            points.red, points.green, points.blue = np.rint(colors.T*65535).astype('uint16')
            writer.write_points(points)
            np.concatenate((xyz,colors),axis=1).astype('<f4').tofile(full)
            # Random sampling avoids row/column aliasing in the diagnostic viewer.
            indices = np.random.default_rng(42+i).choice(len(xyz),len(xyz)//16,replace=False)
            preview.append(np.concatenate((xyz[indices], colors[indices]), axis=1).astype('<f4'))
            count += len(xyz)
            cams.append(cam)
            del scene, ranges, restored, xyz, colors, points
            gc.collect()
    sampled = np.concatenate(preview)
    sampled.tofile(job / 'cloud-preview.bin')
    record(job / 'cameras.json', cams)
    record(job / 'model.json', dict(source='direct_panorama_depth', depth='depth.npy',
                                  learned_depth=False, learned_alignment=False))
    return dict(kind='points', preview='cloud-preview.bin', full_preview='cloud-full.bin', download='cloud.las', count=count,
                preview_count=len(sampled), layers=2, units='m',
                diagnostics={'backprojection_vs_external_depth':depth_stats,
                             'depth_semantics':'ERP radial depth -> world ray * range (no learned alignment)',
                             'depth_source':'depth.npy', 'learned_depth':False, 'learned_alignment':False,
                             'face_resolution':1536, 'face_fov_degrees':100,
                             'initializer_grid':[CLOUD_GRID_SIZE,CLOUD_GRID_SIZE],
                             'base_coordinates':'direct ERP depth backprojection; both layers share the observed surface'},
                description='直接使用全景深度反投影的点云与初始化中心，不经过 SHARP 深度预测或学习对齐。保留六面双层索引；默认全量预览，可切换 1/16 随机抽样。LAS 为全量（精度 0.01 mm）。')


def gaussian_stage(job, adjusted):
    from geometry import camera, split_face
    from gaussian_ops import ndc_to_camera, world_transform, to_vertices
    from point_mask import build_mask, load_face
    from gaussian_variants import save_variants
    progress('准备固定点云 Mask：82° / 64 邻居 / 15 cm')
    mask_info = build_mask(job, progress)
    from gaussian_densify import (build_densification, INDEX, inherit_children,
                                 ComparisonWriter, publish)
    density_start = time.perf_counter()
    density = build_densification(job, progress)
    dense_ids = np.fromfile(job/INDEX, dtype='<u4')
    writer = ComparisonWriter(job, 5 if adjusted else 4)
    grid = json.loads((job/'stage-3.json').read_text(encoding='utf-8'))['diagnostics']['initializer_grid']
    height, width = grid
    face_size = height*width*2
    parts = []
    mask_parts = []
    changes = []
    if adjusted:
        import torch
        from sharp_model import load_model, infer_panorama_depth
        torch.set_num_threads(8)
        torch.manual_seed(42)
        progress('加载 SHARP，使用全景深度初始化并保留最终预测，不做回投')
        model, _ = load_model()
        rgb = np.asarray(Image.open(job / 'input.png').convert('RGB'))
        radial = np.load(job / 'conditioning.npy')
    for i in range(6):
        cam = camera(i)
        ids = dense_ids[dense_ids//(face_size*3) == i].astype('i8')-i*face_size*3
        progress(f'{"预测调整" if adjusted else "导出初始化高斯"} {i+1}/6：{cam["face"]}')
        if adjusted:
            face, z, _ = split_face(rgb, radial, cam)
            scene, _ = infer_panorama_depth(model, face, z, cam)
            scene, _ = ndc_to_camera(scene, cam)
            scene = world_transform(scene, cam)
            with np.load(job / 'base' / f'{i}.npz') as base:
                sample = slice(None,None,32)
                movement = np.linalg.norm(scene['means'][sample]-base['means'][sample],axis=1)
                # Eigenvalue ordering can change when the covariance is rotated.
                scale_ratio = np.sort(scene['scales'][sample],axis=1)/np.sort(base['scales'][sample],axis=1)
                changes.append(dict(face=cam['face'], position_change_m_p50_p90_p99=
                    np.quantile(movement,[.5,.9,.99]).tolist(),
                    principal_scale_ratio_p10_p50_p90=np.quantile(scale_ratio,[.1,.5,.9]).tolist(),
                    opacity_change_abs_mean=float(np.abs(scene['alpha'][sample]-base['alpha'][sample]).mean())))
        else:
            with np.load(job / 'base' / f'{i}.npz') as data:
                scene = {k: data[k] for k in data.files}
        face_keep = load_face(job, i, len(scene['means']))
        vertices, _, source_indices = to_vertices(scene, i, cam['hfov_deg'], return_indices=True)
        with np.load(job/'base'/f'{i}.npz') as data:
            base = {k: data[k] for k in data.files}
        children, inheritance = inherit_children(vertices, source_indices, face_keep, base, ids, cam, width, adjusted)
        writer.append(vertices, source_indices, face_keep, children, inheritance)
        del base, children
        mask_parts.append(face_keep[source_indices])
        parts.append(vertices)
        del scene
        gc.collect()
    if adjusted:
        del model, rgb, radial
        gc.collect()
        torch.cuda.empty_cache()
    vertices = np.concatenate(parts)
    del parts
    filename = 'adjusted.ply' if adjusted else 'initialized.ply'
    keep = np.concatenate(mask_parts)
    progress('写入未筛选与 Mask 后双层高斯 PLY，并逐项回读验证')
    variants, applied = save_variants(job, filename, vertices, keep, mask_info)
    views, files = writer.finish()
    dense_info = publish(job, 5 if adjusted else 4, views, files, density, density_start, writer.inheritance)
    return dict(kind='gaussians', **variants['masked'], variants=variants, mask=applied,
                densification=dense_info,
                layers=2,
                diagnostics={'native_sharp':not adjusted,'external_depth':'panorama_initializer' if adjusted else 'direct',
                             'learned_alignment':False, 'layers':2,
                             'face_resolution':1536, 'face_fov_degrees':100, 'precision':'float32',
                             'changes_from_raw_initializer':changes,
                             'stat_sampling':'every 32nd Gaussian before edge feather; both layers; includes composer activation'},
                description=('统一全景深度初始化双层高斯；保留 SHARP 最终预测的位置、尺度、旋转、颜色和透明度，跳过学习深度对齐，不做最终深度回投。' if adjusted else
                             '导出点云阶段缓存的初始化中心、尺度、旋转、颜色与透明度，尚未加上网络预测修正。') +
                ' 六面 100°，已统一坐标并作边缘透明度融合；默认显示 Mask 后结果，可切换原始结果。')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', type=int, choices=[2,3,4,5])
    p.add_argument('job', type=Path)
    args = p.parse_args()
    start = time.perf_counter()
    result = {2: depth_stage, 3: cloud_stage, 4: lambda j: gaussian_stage(j, False),
              5: lambda j: gaussian_stage(j, True)}[args.stage](args.job)
    result['seconds'] = time.perf_counter()-start
    record(args.job / f'stage-{args.stage}.json', result)
    progress('本步骤完成')


if __name__ == '__main__':
    main()
