"""Small deterministic checks for projection, attention fallback and export."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

from infer_panoramas import erp_directions, save_cloud, save_depth_png
from infer_panoramas import erp_to_cubemap_gpu, cubemap_to_erp_gpu
from depthmaster.model.mirror_vgt.layers.attention import Attention


class InferenceTests(unittest.TestCase):
    def test_unit_rays_and_camera_convention(self):
        rays = erp_directions(5, 10)
        np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1, atol=2e-7)
        self.assertTrue((rays[0, :, 1] > 0).all())
        self.assertTrue((rays[-1, :, 1] < 0).all())
        self.assertGreater(rays[2, 5, 0], 0)
        self.assertGreater(rays[2, 5, 2], 0)

    def test_full_sphere_coverage(self):
        faces = torch.ones(6, 37, 37) * 3.0
        sphere = cubemap_to_erp_gpu(faces, 64, 128)
        torch.testing.assert_close(sphere, torch.full_like(sphere, 3))
        sampled = erp_to_cubemap_gpu(torch.ones(1, 3, 64, 128), 37)
        torch.testing.assert_close(sampled, torch.ones(1, 6, 3, 37, 37))

    def test_attention_matches_explicit_softmax(self):
        torch.manual_seed(4)
        attention = Attention(32, num_heads=4).eval()
        x = torch.randn(2, 13, 32)
        q, k, v = attention.qkv(x).reshape(2, 13, 3, 4, 8).permute(2, 0, 3, 1, 4).unbind(0)
        reference = ((q @ k.transpose(-1, -2) / (8 ** 0.5)).softmax(-1) @ v)
        reference = attention.proj(reference.transpose(1, 2).reshape(2, 13, 32))
        torch.testing.assert_close(attention(x), reference, rtol=2e-5, atol=2e-6)

    def test_cloud_matches_depth_and_color(self):
        depth = np.full((64, 128), 2.5, dtype=np.float32)
        depth[0, 0] = np.nan
        rgb = np.full((64, 128, 3), [50, 100, 150], dtype=np.uint8)
        with tempfile.TemporaryDirectory(prefix='depthmaster-test-') as directory:
            root = Path(directory)
            count, error = save_cloud(root / 'test.ply', depth, rgb)
            self.assertEqual(count, depth.size - 1)
            self.assertLess(error, 1e-5)
            cloud = PlyData.read(root / 'test.ply', mmap=False)['vertex']
            points = np.stack([cloud[name] for name in ('x', 'y', 'z')], axis=-1)
            np.testing.assert_allclose(np.linalg.norm(points, axis=-1), 2.5, atol=1e-6)
            np.testing.assert_array_equal(cloud['red'], 50)
            save_depth_png(root / 'test.png', depth, 'test')
            with Image.open(root / 'test.png') as image:
                self.assertEqual(image.size, (128, 192))


if __name__ == '__main__':
    unittest.main()
