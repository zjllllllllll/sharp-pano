import json
import unittest

import numpy as np

import point_distances as distance
import point_normals as normals
import test_point_normals


def means(points, ks, chunk_size=2):
    blocks = list(distance.iter_neighbor_means(points, ks, chunk_size=chunk_size, workers=1))
    return np.concatenate([v for _, v in blocks]) if blocks else np.empty((0, len(ks)))


class DistanceTests(unittest.TestCase):
    def test_exact_mean_excludes_self_and_uses_meters(self):
        points = np.array([[0, 0, 0], [.01, 0, 0], [.03, 0, 0], [0, .04, 0]])
        ks = [1, 2, 3, 8]
        pairwise = np.linalg.norm(points[:, None]-points[None, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        sorted_dist = np.sort(pairwise, axis=1)[:, :-1]
        expected = np.column_stack([sorted_dist[:, :min(k, 3)].mean(1) for k in ks])
        np.testing.assert_allclose(means(points, ks), expected, rtol=1e-6)

    def test_neighbors_are_searched_after_normal_mask(self):
        points = np.array([[0, 0, 1], [.001, 0, 1], [1, 0, 1], [2, 0, 1]])
        angles = np.array([82, 82.1, 0, -1])
        keep = distance.normal_keep(angles)
        np.testing.assert_array_equal(keep, [True, False, True, True])
        np.testing.assert_allclose(means(points[keep], [1])[:, 0], [1, 1, 1])
        self.assertLess(means(points, [1])[0, 0], .002)

    def test_duplicates_do_not_break_self_exclusion(self):
        points = np.vstack((np.zeros((80, 3)), [[1, 0, 0]]))
        values = means(points, [4, 16, 64], chunk_size=7)
        np.testing.assert_array_equal(values[:80], 0)
        np.testing.assert_array_equal(values[80], 1)

    def test_empty_and_singleton(self):
        self.assertEqual(means(np.zeros((0, 3)), [4]).shape, (0, 1))
        self.assertEqual(means(np.zeros((1, 3)), [4])[0, 0], distance.NO_NEIGHBOR)

    def test_preview_cache_alignment_and_geometry_unchanged(self):
        fixture = test_point_normals.CacheTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        job = fixture.job
        normals.build_angles(job, lambda _: None)
        originals = {p: p.read_bytes() for p in job.rglob('*') if p.is_file()}
        info = distance.build_distances(job, lambda _: None)
        self.assertEqual(info, distance.cached_result(job))
        self.assertEqual(info['normal_angle_degrees'], 82)
        self.assertEqual(info['iterations'], 1)
        full = np.fromfile(job/distance.FULL, dtype='<f4').reshape(-1, len(distance.NEIGHBORS))
        sample = np.fromfile(job/distance.SAMPLE, dtype='<f4').reshape(-1, len(distance.NEIGHBORS))
        faces = json.loads((job/normals.MANIFEST).read_text())['faces']
        a = b = 0
        for i, face in enumerate(faces):
            n = face['count']
            indices = np.random.default_rng(42+i).choice(n, n//16, replace=False)
            np.testing.assert_array_equal(sample[b:b+len(indices)], full[a+indices])
            a += n
            b += len(indices)
        angles = np.fromfile(job/normals.FULL, dtype='<f4')
        keep = distance.normal_keep(angles)
        expected = means(fixture.points[keep, :3], distance.NEIGHBORS, chunk_size=33)
        np.testing.assert_allclose(full[keep], expected, rtol=1e-6)
        self.assertTrue((full[~keep] == -1).all())
        timestamp = (job/distance.FULL).stat().st_mtime_ns
        distance.build_distances(job, lambda _: None)
        self.assertEqual(timestamp, (job/distance.FULL).stat().st_mtime_ns)
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)
        (job/distance.SAMPLE).write_bytes(b'bad')
        self.assertIsNone(distance.cached_result(job))


if __name__ == '__main__':
    unittest.main()
