"""Read-only route guards for the independent camera lab."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import camera_lab


class CameraLabTests(unittest.TestCase):
    def test_whitelisted_artifacts(self):
        for name in ['0.1/cameras.png', '0.1/camera-map-base.png', '9.2/outline.json',
                     '0.1/panoramas/C001.png', '0.1/panoramas/C001-coverage.png',
                     '0.1/panoramas/C001-mask.png','0.1/panoramas/C001-reason-bits.png',
                     '0.1/panoramas/C001-depths.npz','0.1/panoramas/C001-overlay.png']:
            self.assertTrue(camera_lab.artifact(name).is_relative_to(camera_lab.RUN.resolve()))

    def test_no_traversal_or_arbitrary_files(self):
        for name in ['../server.py', '0.1/../../server.py', '0.1/cloud-full.bin',
                     '0.1/panoramas/../../input.png', '0.1\\input.png',
                     '0.1/panoramas/C001.png/extra', '/0.1/input.png',
                     '0.1/panoramas/C001-anything.png','0.1/panoramas/C001-depths.py']:
            with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                camera_lab.artifact(name)

    def test_catalog_pending_and_ready(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(camera_lab, 'RUN', Path(temp)):
            self.assertEqual(camera_lab.catalog()['total'], 0)
            expected = {'total': 1, 'scenes': [{'id': '0.1'}]}
            (Path(temp) / 'catalog.json').write_text(json.dumps(expected), encoding='utf-8')
            self.assertEqual(camera_lab.catalog(), expected)


if __name__ == '__main__':
    unittest.main()
