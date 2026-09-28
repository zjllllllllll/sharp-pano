"""Exercise downloads using tiny checkpoints and mocked HTTP responses."""
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch
import hashlib
import json

import download_models


class Response(BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}


class ModelDownloadTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "models").mkdir()
        self.content = b"tiny-checkpoint-for-download-tests"
        self.target = self.root / "models/model.pt"
        self.partial = self.target.with_suffix(".partial")
        manifest = [{"name": "test", "file": "model.pt",
                     "url": "https://example.invalid/model.pt",
                     "bytes": len(self.content),
                     "sha256": hashlib.sha256(self.content).hexdigest()}]
        (self.root / "models/manifest.json").write_text(json.dumps(manifest))
        self.addCleanup(patch.stopall)
        patch.object(download_models, "ROOT", self.root).start()
        self.http = patch.object(download_models, "urlopen").start()

    def run_download(self, *arguments):
        with patch("sys.argv", ["download_models.py", *arguments]), redirect_stdout(StringIO()):
            download_models.main()

    def test_new_download_is_verified_then_published(self):
        self.http.return_value = Response(self.content)
        self.run_download()
        self.assertEqual(self.target.read_bytes(), self.content)
        self.assertFalse(self.partial.exists())

    def test_interrupted_download_resumes_with_exact_range(self):
        offset = 7
        self.partial.write_bytes(self.content[:offset])
        self.http.return_value = Response(self.content[offset:], 206,
                                          {"Content-Range": f"bytes {offset}-{len(self.content)-1}/{len(self.content)}"})
        self.run_download()
        request = self.http.call_args.args[0]
        self.assertEqual(request.get_header("Range"), f"bytes={offset}-")
        self.assertEqual(self.target.read_bytes(), self.content)

    def test_server_ignoring_range_restarts_without_appending(self):
        self.partial.write_bytes(self.content[:7])
        self.http.return_value = Response(self.content)
        self.run_download()
        self.assertEqual(self.target.read_bytes(), self.content)

    def test_completed_partial_is_promoted_without_http(self):
        self.partial.write_bytes(self.content)
        self.run_download()
        self.http.assert_not_called()
        self.assertEqual(self.target.read_bytes(), self.content)

    def test_corrupt_complete_partial_restarts_from_zero(self):
        self.partial.write_bytes(b"x" * len(self.content))
        self.http.return_value = Response(self.content)
        self.run_download()
        self.assertIsNone(self.http.call_args.args[0].get_header("Range"))
        self.assertEqual(self.target.read_bytes(), self.content)

    def test_bad_range_is_rejected_without_publishing(self):
        self.partial.write_bytes(self.content[:7])
        self.http.return_value = Response(self.content[8:], 206, {"Content-Range": "bytes 8-32/33"})
        with self.assertRaisesRegex(ValueError, "Unexpected download range"):
            self.run_download()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.partial.read_bytes(), self.content[:7])

    def test_corrupt_download_never_becomes_a_checkpoint(self):
        self.http.return_value = Response(b"x" * len(self.content))
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.run_download()
        self.assertFalse(self.target.exists())

    def test_verify_only_never_downloads(self):
        with self.assertRaises(FileNotFoundError):
            self.run_download("--verify-only")
        self.target.write_bytes(self.content)
        self.run_download("--verify-only")
        self.http.assert_not_called()

    def test_existing_bad_checkpoint_is_preserved(self):
        self.target.write_bytes(b"keep-for-inspection")
        with self.assertRaisesRegex(ValueError, "preserved"):
            self.run_download()
        self.http.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b"keep-for-inspection")


if __name__ == "__main__":
    main()
