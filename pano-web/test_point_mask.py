import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from plyfile import PlyData

import point_distances as distances
import point_normals as normals
import point_mask as mask
import test_point_normals

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sharp-sixface'))
from gaussian_ops import to_vertices, DTYPE
from gaussian_variants import save_variants


class MaskTests(unittest.TestCase):
    def test_fixed_thresholds_and_equality_and_unknown_normals(self):
        angles = np.array([82, 82.01, -1, -1, 0, 0], dtype=np.float32)
        scores = np.array([.15, -1, .15, .151, .151, .149], dtype=np.float32)
        np.testing.assert_array_equal(mask.fixed_keep(angles, scores), [1, 0, 1, 0, 0, 1])
        self.assertEqual(mask.NEIGHBORS, 64)
        self.assertEqual(mask.DISTANCE_METERS, .15)
        with self.assertRaises(ValueError):
            mask.fixed_keep(np.zeros(1), np.array([np.nan]))

    def test_face_layer_pixel_mapping_cache_and_originals_unchanged(self):
        fixture = test_point_normals.CacheTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        job = fixture.job
        originals = {p:p.read_bytes() for p in job.rglob('*') if p.is_file()}
        info = distances.build_distances(job, lambda _:None)
        # Controlled cached scores cover both filters and both sides of .15 m.
        angles = np.fromfile(job/normals.FULL, dtype='<f4')
        angles[0], angles[1] = 83, -1
        angles.tofile(job/normals.FULL)
        values = np.full((len(angles), 5), .05, dtype='<f4')
        values[0] = -1
        values[1, -1] = .2
        values[-1, -1] = .15
        values.tofile(job/distances.FULL)
        info['sources'] = distances.source_signature(job)
        (job/distances.MANIFEST).write_text(json.dumps(info), encoding='utf-8')
        result = mask.build_mask(job, lambda _:None)
        self.assertEqual(result['removed'], 2)
        self.assertEqual(result, mask.cached_result(job))
        from geometry import angular_weights
        cams = json.loads((job/'cameras.json').read_text())
        offset = 0
        expected = mask.fixed_keep(angles, values[:, -1])
        for i, cam in enumerate(cams):
            with np.load(job/'base'/f'{i}.npz') as base:
                means = base['means']
            domain = angular_weights(means, i, cam['hfov_deg']) > 1e-4
            keep = mask.load_face(job, i, len(means))
            np.testing.assert_array_equal(keep[domain], expected[offset:offset+domain.sum()])
            self.assertFalse(keep[~domain].any())
            offset += int(domain.sum())
        timestamp = (job/mask.FILE).stat().st_mtime_ns
        mask.build_mask(job, lambda _:None)
        self.assertEqual(timestamp, (job/mask.FILE).stat().st_mtime_ns)
        for p, content in originals.items():
            self.assertEqual(p.read_bytes(), content)
        with self.assertRaisesRegex(ValueError, '索引数量'):
            mask.load_face(job, 0, 123)
        (job/mask.FILE).write_bytes(b'bad')
        self.assertIsNone(mask.cached_result(job))

    @staticmethod
    def scene():
        return dict(means=np.array([[0,0,2], [.1,0,2], [.2,0,2], [100,0,2], [.4,0,2]], dtype='float32'),
                    scales=np.full((5,3), .01, dtype='float32'),
                    quats=np.tile(np.array([1,0,0,0], dtype='float32'), (5,1)),
                    colors=np.full((5,3), .5, dtype='float32'),
                    alpha=np.array([.5,0,.5,.5,.5], dtype='float32'))

    def test_mask_uses_source_identity_after_movement_and_native_culling(self):
        scene = self.scene()
        source_keep = np.array([True, True, False, False, True])
        before, _, indices = to_vertices(scene, 0, return_indices=True)
        np.testing.assert_array_equal(indices, [0,2,4])
        np.testing.assert_array_equal(before[source_keep[indices]]['x'], np.array([0,.4], dtype='float32'))
        # A previously culled boundary point moves into view, and a deleted point
        # moves onto a retained point. Neither may evade its source mask.
        scene['means'][3] = [0,0,2]
        scene['means'][2] = scene['means'][0]
        after, _, indices = to_vertices(scene, 0, return_indices=True)
        np.testing.assert_array_equal(indices, [0,2,3,4])
        self.assertEqual(int(source_keep[indices].sum()), 2)
        np.testing.assert_array_equal(after[source_keep[indices]], before[[0,2]])

    def test_ply_variants_are_bit_exact_subset_and_baseline_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            vertices, _ = to_vertices(self.scene(), 0)
            keep = np.array([True, False, True])
            info = dict(parameters=mask.PARAMETERS, sha256='fixture', count=5, kept=2, removed=3,
                        index='original source', outside_cloud_policy='exclude')
            variants, applied = save_variants(job, 'initialized.ply', vertices, keep, info)
            original = (job/'initialized.ply').read_bytes()
            masked = PlyData.read(job/'initialized-masked.ply', mmap=False)['vertex'].data
            np.testing.assert_array_equal(masked, vertices[keep])
            self.assertEqual(applied['gaussian_removed'], 1)
            self.assertEqual(variants['masked']['count'], 2)
            save_variants(job, 'initialized.ply', vertices, keep, info)
            self.assertEqual((job/'initialized.ply').read_bytes(), original)
            changed = vertices.copy()
            changed['x'][0] += 1
            with self.assertRaisesRegex(ValueError, '不一致'):
                save_variants(job, 'initialized.ply', changed, keep, info)
            self.assertEqual((job/'initialized.ply').read_bytes(), original)
            save_variants(job, 'initialized.ply', vertices, np.zeros(3, dtype=bool), info)
            self.assertEqual(len(PlyData.read(job/'initialized-masked.ply', mmap=False)['vertex'].data), 0)


if __name__ == '__main__':
    unittest.main()
