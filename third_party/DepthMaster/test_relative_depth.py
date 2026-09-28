import unittest

import numpy as np

from export_relative_depth import normalize_relative, colorize, grayscale


class RelativeDepthTests(unittest.TestCase):
    def test_normalization_preserves_order_and_source(self):
        depth = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
        source = depth.copy()
        relative, valid, limits = normalize_relative(depth)
        np.testing.assert_array_equal(depth, source)
        np.testing.assert_allclose(limits, [1.99, 99.01])
        self.assertTrue((np.diff(relative.ravel()) >= 0).all())
        self.assertTrue(valid.all())
        self.assertEqual(float(relative.min()), 0)
        self.assertEqual(float(relative.max()), 1)
        gray = grayscale(relative, valid)
        self.assertEqual(gray[0, 0], 255)
        self.assertEqual(gray[-1, -1], 0)

    def test_invalid_samples_are_excluded_and_black(self):
        depth = np.array([[1, 2, 3], [np.nan, np.inf, -1]], dtype=np.float32)
        relative, valid, limits = normalize_relative(depth, 0, 100)
        self.assertEqual(limits, (1, 3))
        np.testing.assert_allclose(relative[0], [0, 0.5, 1])
        self.assertTrue((colorize(relative, valid)[1] == 0).all())
        self.assertTrue((grayscale(relative, valid)[1] == 0).all())

    def test_flat_depth_stays_flat(self):
        relative, valid, limits = normalize_relative(np.full((4, 8), 3.0))
        np.testing.assert_allclose(relative, 0.5)
        self.assertEqual(limits, (3, 3))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_relative(np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            normalize_relative(np.ones((2, 4)), 99, 1)


if __name__ == '__main__':
    unittest.main()
