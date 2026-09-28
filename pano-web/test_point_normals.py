import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from point_normals import (estimate_angles, build_angles, cached_result,
                           FULL, SAMPLE, MANIFEST)


class NormalTests(unittest.TestCase):
    @staticmethod
    def plane(n=17):
        x, y = np.meshgrid(np.linspace(-.1, .1, n), np.linspace(-.1, .1, n))
        return np.stack((x, y, np.ones_like(x)*2), axis=-1)

    def test_front_plane_and_acute_angle(self):
        p = self.plane()
        got = estimate_angles(p)
        expected = np.degrees(np.arccos(2/np.linalg.norm(p, axis=-1)))
        self.assertTrue((got >= 0).all())
        np.testing.assert_allclose(got, expected, atol=1e-4)

    def test_grazing_plane(self):
        y, z = np.meshgrid(np.linspace(-.08, .08, 17), np.linspace(1.8, 2.2, 17))
        p = np.stack((np.ones_like(y)*.1, y, z), axis=-1)
        got = estimate_angles(p)
        expected = np.degrees(np.arccos(.1/np.linalg.norm(p, axis=-1)))
        self.assertTrue((got > 85).all())
        np.testing.assert_allclose(got, expected, atol=1e-4)

    def test_rotation_translation_scale_and_chunks(self):
        p = self.plane()
        a = .7
        rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                             [-np.sin(a), 0, np.cos(a)]])
        center = np.array([13, -8, 4])
        expected = estimate_angles(p, chunk_rows=3)
        got = estimate_angles((p @ rotation.T)*7+center, center, chunk_rows=8)
        np.testing.assert_allclose(got, expected, atol=2e-5)

    def test_depth_jump_does_not_create_fake_bridge(self):
        p = self.plane()
        p[:, 8:, 2] = 4
        got = estimate_angles(p)
        expected = np.degrees(np.arccos(p[..., 2]/np.linalg.norm(p, axis=-1)))
        self.assertTrue((got >= 0).all())
        np.testing.assert_allclose(got, expected, atol=1e-4)

    def test_invalid_isolated_and_line_are_unreliable(self):
        p = np.zeros((9, 9, 3))
        self.assertTrue((estimate_angles(p) == -1).all())
        p[:, :, 2] = np.linspace(2, 2.1, 9)
        self.assertTrue((estimate_angles(p) == -1).all())
        p = self.plane()
        p[8, 8] = np.nan
        got = estimate_angles(p)
        self.assertEqual(got[8, 8], -1)
        self.assertTrue(np.isfinite(got).all())
        self.assertLess(got[8, 7], 2)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.job = Path(self.temp.name)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sharp-sixface'))
        from geometry import camera, view_rays, angular_weights
        (self.job/'base').mkdir()
        cams, full, sample = [], [], []
        for i in range(6):
            cam = camera(i, size=24)
            local, _ = view_rays(cam, grid_size=12)
            grids = np.stack([local/local[..., 2:]*z for z in (2, 3)])
            means = (grids.reshape(-1, 3) @ np.asarray(cam['c2w'])[:3, :3].T).astype('float32')
            np.savez(self.job/'base'/f'{i}.npz', means=means)
            keep = angular_weights(means, i, cam['hfov_deg']) > 1e-4
            points = np.column_stack((means[keep], np.full((keep.sum(), 3), .5))).astype('<f4')
            indices = np.random.default_rng(42+i).choice(len(points), len(points)//16, replace=False)
            full.append(points)
            sample.append(points[indices])
            cams.append(cam)
        self.points, self.sample = np.concatenate(full), np.concatenate(sample)
        self.points.tofile(self.job/'cloud-full.bin')
        self.sample.tofile(self.job/'cloud-preview.bin')
        (self.job/'cameras.json').write_text(json.dumps(cams))
        (self.job/'stage-3.json').write_text(json.dumps(dict(
            layers=2, diagnostics={'initializer_grid': [12, 12]},
            count=len(self.points), preview_count=len(self.sample))))

    def test_full_and_sample_alignment_cache_and_no_geometry_mutation(self):
        originals = {p: p.read_bytes() for p in self.job.rglob('*') if p.is_file()}
        info = build_angles(self.job, progress=lambda _: None)
        self.assertEqual(info['count'], len(self.points))
        self.assertEqual(info['preview_count'], len(self.sample))
        self.assertEqual(info, cached_result(self.job))
        a = np.fromfile(self.job/FULL, dtype='<f4')
        b = np.fromfile(self.job/SAMPLE, dtype='<f4')
        full_offset = sample_offset = 0
        for face in info['faces']:
            n = face['count']
            i = full_offset // n
            indices = np.random.default_rng(42+i).choice(n, n//16, replace=False)
            np.testing.assert_array_equal(b[sample_offset:sample_offset+n//16], a[full_offset+indices])
            full_offset += n
            sample_offset += n//16
        timestamp = (self.job/FULL).stat().st_mtime_ns
        build_angles(self.job, progress=lambda _: None)
        self.assertEqual(timestamp, (self.job/FULL).stat().st_mtime_ns)
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)
        (self.job/SAMPLE).write_bytes(b'bad')
        self.assertIsNone(cached_result(self.job))

    def test_wrong_point_order_fails_without_publishing_angles(self):
        self.points[::-1].tofile(self.job/'cloud-full.bin')
        with self.assertRaisesRegex(ValueError, '顺序不匹配'):
            build_angles(self.job, progress=lambda _: None)
        self.assertFalse((self.job/MANIFEST).exists())
        self.assertEqual(list(self.job.glob('*.partial')), [])


if __name__ == '__main__':
    unittest.main()
