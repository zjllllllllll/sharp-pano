"""Official joint-panoramic inference -> metric range maps and backprojected PLY.

Output per image: .range_m.npy (float32 meters, invalid=NaN), .depth.png
(fixed 0..10 m visualization), .ply (one XYZRGB point per valid ERP pixel).
PLY convention: x right, y up, z forward; camera at origin. No scale fitting,
plane fitting, mesh reconstruction, image enhancement, or Gaussian generation.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from depthmaster.model import DepthMasterModel
# Upstream has both eval_panorama/eval.py and eval_panorama/eval/, so the
# README's ordinary import resolves to the package rather than the helper file.
_projection_path = Path(__file__).resolve().parent / 'eval_panorama' / 'eval.py'
_projection_spec = importlib.util.spec_from_file_location('_depthmaster_pano_projection', _projection_path)
_projection = importlib.util.module_from_spec(_projection_spec)
_projection_spec.loader.exec_module(_projection)
erp_to_cubemap_gpu = _projection.erp_to_cubemap_gpu
cubemap_to_erp_gpu = _projection.cubemap_to_erp_gpu
_get_camera_params_cached = _projection._get_camera_params_cached


def erp_directions(height, width):
    """Pixel-center rays matching the official ERP projection, with +Y up."""
    lon = ((np.arange(width, dtype=np.float32) + 0.5) / width - 0.5) * (2 * np.pi)
    lat = (0.5 - (np.arange(height, dtype=np.float32) + 0.5) / height) * np.pi
    cos_lat = np.cos(lat)[:, None]
    rays = np.empty((height, width, 3), dtype=np.float32)
    rays[..., 0] = cos_lat * np.sin(lon)[None, :]
    rays[..., 1] = np.sin(lat)[:, None]
    rays[..., 2] = cos_lat * np.cos(lon)[None, :]
    return rays


def save_cloud(path, depth, rgb):
    valid = np.isfinite(depth) & (depth > 0)
    rays = erp_directions(*depth.shape)
    xyz = rays[valid] * depth[valid, None]
    colors = rgb[valid]
    radial_error = np.max(np.abs(np.linalg.norm(xyz, axis=1) - depth[valid]))
    if radial_error > max(1e-5, float(depth[valid].max()) * 1e-6):
        raise RuntimeError(f'Backprojection range mismatch: {radial_error}')
    vertices = np.empty(len(xyz), dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    for axis, field in enumerate(('x', 'y', 'z')):
        vertices[field] = xyz[:, axis]
    for axis, field in enumerate(('red', 'green', 'blue')):
        vertices[field] = colors[:, axis]
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        'comment DepthMaster spherical range backprojection; units=meters\n'
        'comment coordinates x=right y=up z=forward; camera=(0,0,0)\n'
        f'comment source ERP width={depth.shape[1]} height={depth.shape[0]}\n'
        f'element vertex {len(vertices)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    )
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        vertices.tofile(stream)
    return len(vertices), float(radial_error)


def save_depth_png(path, depth, name, max_m=10.0):
    valid = np.isfinite(depth)
    normalized = np.nan_to_num(depth, nan=0, posinf=max_m, neginf=0)
    u8 = np.rint(np.clip(normalized / max_m, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    bgr[~valid] = 0
    height, width = depth.shape
    footer = np.full((128, width, 3), 245, dtype=np.uint8)
    cv2.putText(footer, f'{name} | DepthMaster | radial distance (meters) | black = invalid',
                (36, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2, cv2.LINE_AA)
    gradient = np.tile(np.linspace(0, 255, width - 72, dtype=np.uint8), (24, 1))
    footer[50:74, 36:width - 36] = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    for index in range(6):
        x = int(36 + (width - 73) * index / 5)
        label = f'{max_m * index / 5:g}m' + ('+' if index == 5 else '')
        cv2.putText(footer, label, (min(x, width - 94), 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), np.concatenate((bgr, footer), axis=0)):
        raise RuntimeError(f'Cannot write {path}')


@torch.inference_mode()
def infer(model, rgb, face_size=518, fov=95.0):
    height, width = rgb.shape[:2]
    device = next(model.parameters()).device
    pano = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device).float() / 255.0
    torch.cuda.synchronize()
    started = time.perf_counter()
    faces = erp_to_cubemap_gpu(pano, face_w=face_size, fov_deg=fov)
    w2c, intrinsics = _get_camera_params_cached(fov, face_size, device, torch.float32)
    torch.cuda.synchronize()
    network_started = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.float16):
        raw = model(
            faces.to(model.dtype), num_tokens=0, camera_type='Panorama',
            W2C=w2c[None].to(model.dtype), intrinsics=intrinsics[None].to(model.dtype),
        )
    torch.cuda.synchronize()
    network_seconds = time.perf_counter() - network_started
    scale = raw['metric_scale'].float().reshape(-1)
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
        raise RuntimeError(f'Invalid metric scale {scale}')
    points = raw['pts3d'].float() * scale.reshape(1, 1, 1, 1, 1)
    distances = torch.linalg.vector_norm(points[0], dim=-1)
    probability = raw['pts3d_mask'][0].float()
    if not torch.isfinite(distances).all():
        raise RuntimeError('Non-finite model depth output; do not silently export it')
    # Official evaluation caps projection resolution at the cubemap's natural
    # resolution, then resizes to the input ERP dimensions. No extra detail is
    # claimed by the final upsampling.
    proj_h, proj_w = min(height, face_size * 2), min(width, face_size * 4)
    distance = cubemap_to_erp_gpu(distances, proj_h, proj_w, fov_deg=fov)
    mask_prob = cubemap_to_erp_gpu(probability, proj_h, proj_w, fov_deg=fov)
    if (proj_h, proj_w) != (height, width):
        distance = F.interpolate(distance[None, None], (height, width),
                                 mode='bilinear', align_corners=False)[0, 0]
        mask_prob = F.interpolate(mask_prob[None, None], (height, width),
                                  mode='bilinear', align_corners=False)[0, 0]
    valid = (mask_prob > 0.5) & torch.isfinite(distance) & (distance > 0)
    distance = torch.where(valid, distance, torch.nan)
    depth = distance.cpu().numpy().astype(np.float32)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return depth, {
        'model_seconds': network_seconds,
        'projection_and_model_seconds': elapsed,
        'metric_scale': scale.item(),
        'projection_size_wh': [proj_w, proj_h],
    }


def load_verified_model(checkpoint_path, device):
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    print('MODEL_CONFIG', json.dumps(checkpoint['model_config'], ensure_ascii=True), flush=True)
    model = DepthMasterModel(**checkpoint['model_config'])
    # Do not accept the upstream loader's silent strict=False missing weights.
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model = model.eval().to(device)
    print(f'MODEL_LOADED seconds={time.perf_counter() - started:.3f}', flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('../pano-tests'))
    parser.add_argument('--output-dir', type=Path, default=Path('../depthmaster-results'))
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/depthmaster.pt'))
    parser.add_argument('--images', nargs='+', default=['0.1.png', '1.1.png', '2.1.png', '3.1.png', '4.1.png'])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this run')
    torch.set_num_threads(8)
    torch.manual_seed(0)
    np.random.seed(0)
    print('DEVICE', torch.cuda.get_device_name(), 'TORCH', torch.__version__, flush=True)
    paths = [args.input_dir / name for name in args.images]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_verified_model(args.checkpoint, torch.device('cuda'))
    for index, path in enumerate(paths, 1):
        stem = path.stem
        outputs = [args.output_dir / f'{stem}{extension}' for extension in ('.range_m.npy', '.depth.png', '.ply')]
        if any(p.exists() for p in outputs) and not args.overwrite:
            raise FileExistsError(f'Existing outputs for {stem}; use --overwrite explicitly')
        print(f'RUN {index}/{len(paths)} {path.name}', flush=True)
        rgb = np.asarray(Image.open(path).convert('RGB'))
        if rgb.shape[1] != rgb.shape[0] * 2:
            raise ValueError(f'Not a full 2:1 ERP panorama: {path}')
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        depth, stats = infer(model, rgb)
        finite = np.isfinite(depth)
        if finite.mean() < 0.05:
            raise RuntimeError(f'Too few valid pixels for {path}: {finite.mean():.4f}')
        np.save(outputs[0], depth, allow_pickle=False)
        save_depth_png(outputs[1], depth, stem)
        count, error = save_cloud(outputs[2], depth, rgb)
        stats.update({
            'name': path.name, 'depth_size_wh': [depth.shape[1], depth.shape[0]],
            'points': count, 'valid_fraction': float(finite.mean()),
            'range_m_p01_p50_p99': np.quantile(depth[finite], [0.01, 0.5, 0.99]).tolist(),
            'radial_reprojection_max_error_m': error,
            'peak_cuda_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
            'inference_and_export_seconds': time.perf_counter() - started,
        })
        print('RESULT', json.dumps(stats, ensure_ascii=True), flush=True)
    print(f'COMPLETE {len(paths)} panoramas -> {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
