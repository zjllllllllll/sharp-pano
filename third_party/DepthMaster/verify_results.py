"""Verify every exported XYZRGB point against its source ERP and float depth."""
from pathlib import Path
import json

import numpy as np
from PIL import Image
from plyfile import PlyData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from infer_panoramas import erp_directions


ROOT = Path(__file__).resolve().parent.parent
NAMES = ['0.1', '1.1', '2.1', '3.1', '4.1']


def main():
    results = ROOT / 'depthmaster-results'
    expected = {f'{name}{suffix}' for name in NAMES
                for suffix in ('.range_m.npy', '.depth.png', '.ply')}
    actual = {path.name for path in results.iterdir()}
    if expected != actual:
        raise AssertionError(f'Unexpected result inventory: missing={expected - actual}, extra={actual - expected}')
    figure = plt.figure(figsize=(15, 19), facecolor='white')
    for index, name in enumerate(NAMES):
        depth = np.load(results / f'{name}.range_m.npy', allow_pickle=False)
        rgb = np.asarray(Image.open(ROOT / 'pano-tests' / f'{name}.png').convert('RGB'))
        assert depth.dtype == np.float32 and depth.shape == rgb.shape[:2] == (2048, 4096)
        valid = np.isfinite(depth) & (depth > 0)
        assert valid.sum() > 0
        cloud = PlyData.read(results / f'{name}.ply', mmap='r')['vertex']
        assert len(cloud) == valid.sum()
        xyz = np.stack([cloud[field] for field in ('x', 'y', 'z')], axis=-1)
        color = np.stack([cloud[field] for field in ('red', 'green', 'blue')], axis=-1)
        assert np.isfinite(xyz).all()
        expected_xyz = erp_directions(*depth.shape)[valid] * depth[valid, None]
        max_xyz_error = float(np.max(np.abs(expected_xyz - xyz)))
        np.testing.assert_allclose(xyz, expected_xyz, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(color, rgb[valid])
        with Image.open(results / f'{name}.depth.png') as preview:
            preview.load()
            assert preview.size == (4096, 2176)
            # Show only the spatial depth image; use the same fixed scale for all.
            ax = figure.add_subplot(5, 2, index * 2 + 1)
            ax.imshow(preview)
            ax.set_title(f'{name}  |  range depth, 0-10 m', loc='left', fontsize=13)
            ax.axis('off')
        # Read-only cutaway preview: remove the ceiling for inspection, but leave
        # the full exported cloud untouched. Random sampling avoids aliasing the
        # flattened ERP into longitude stripes when the stride divides its width.
        sample = np.random.default_rng(42).choice(len(xyz), min(180000, len(xyz)), replace=False)
        preview_xyz = xyz[sample]
        preview_rgb = color[sample] / 255.0
        cut = preview_xyz[:, 1] < 0.3
        points, colors = preview_xyz[cut], preview_rgb[cut]
        ax3 = figure.add_subplot(5, 2, index * 2 + 2, projection='3d')
        ax3.scatter(points[:, 0], points[:, 2], points[:, 1], c=colors,
                    s=0.65, linewidths=0, depthshade=False, rasterized=True)
        ax3.view_init(elev=55, azim=-55)
        ax3.set_box_aspect(np.maximum(np.ptp(points[:, [0, 2, 1]], axis=0), 0.1))
        ax3.set_axis_off()
        ax3.set_title(f'{name}  |  point cloud (ceiling hidden for preview)', loc='left', fontsize=13)
        print('VERIFIED', json.dumps({
            'name': name, 'shape': list(depth.shape), 'points': len(cloud),
            'range_min_max_m': [float(depth[valid].min()), float(depth[valid].max())],
            'xyz_max_error_m': max_xyz_error, 'rgb_matches_source': True,
        }), flush=True)
        del expected_xyz, xyz, color, cloud, rgb, depth
    figure.tight_layout(pad=2)
    preview_dir = Path(__file__).resolve().parent / 'preview'
    preview_dir.mkdir(exist_ok=True)
    path = preview_dir / 'depthmaster-overview.png'
    figure.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(figure)
    print(f'ALL_5_VERIFIED; PREVIEW={path}', flush=True)


if __name__ == '__main__':
    main()
