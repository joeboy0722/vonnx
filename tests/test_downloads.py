import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from vonnx_tools.prepare import download_archive


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}


class DownloadTests(unittest.TestCase):
    def test_resume_and_server_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            partial = Path(folder) / "archive.download"
            partial.write_bytes(b"abc")
            with patch(
                "urllib.request.urlopen",
                return_value=Response(
                    b"def",
                    206,
                    {
                        "Content-Length": "3",
                        "Content-Range": "bytes 3-5/6",
                    },
                ),
            ) as request:
                download_archive("https://example.org/data.zip", partial)
                self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=3-")
            self.assertEqual(partial.read_bytes(), b"abcdef")
            with patch("urllib.request.urlopen", return_value=Response(b"replacement")):
                download_archive("https://example.org/data.zip", partial)
            self.assertEqual(partial.read_bytes(), b"replacement")

    def test_wrong_range_is_not_appended(self):
        with tempfile.TemporaryDirectory() as folder:
            partial = Path(folder) / "archive.download"
            partial.write_bytes(b"abc")
            with patch(
                "urllib.request.urlopen",
                side_effect=lambda *a, **k: Response(
                    b"bad",
                    206,
                    {
                        "Content-Length": "3",
                        "Content-Range": "bytes 0-2/6",
                    },
                ),
            ), patch("time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "incorrect range"):
                    download_archive("https://example.org/data.zip", partial)
            self.assertEqual(partial.read_bytes(), b"abc")
