import concurrent.futures
import unittest
import numpy as np
from vonnx import PixelFormat, load_image
from vonnx_tools.common import engine, images
from vonnx_tools.validate import box_array


class MultiModelTests(unittest.TestCase):
    def test_independent_engines_match_concurrent_results(self):
        models = ["yolov8n", "yolov8n-pose", "yolov8n-seg"]
        image = load_image(str(images("pose")[0]))
        engines = [engine(m, 320, batch=4) for m in models]
        try:
            alone = [e.predict_numpy(i, 10, image, PixelFormat.RGB24)[0] for i, e in enumerate(engines)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [
                    pool.submit(e.predict_numpy, i, 10, image, PixelFormat.RGB24)
                    for i, e in enumerate(engines)
                ]
                together = [j.result(timeout=60)[0] for j in jobs]
            for a, b in zip(alone, together):
                self.assertEqual((a.stream_id, a.frame_id), (b.stream_id, b.frame_id))
                np.testing.assert_allclose(box_array(a), box_array(b), atol=1e-4, rtol=1e-3)
                for ka, kb in zip(a.keypoints, b.keypoints):
                    np.testing.assert_allclose(
                        [[p.x, p.y, p.confidence] for p in ka],
                        [[p.x, p.y, p.confidence] for p in kb],
                        atol=1e-4,
                        rtol=1e-3,
                    )
                for ma, mb in zip(a.masks, b.masks):
                    self.assertEqual((ma.x, ma.y, ma.width, ma.height), (mb.x, mb.y, mb.width, mb.height))
                    np.testing.assert_array_equal(ma.data, mb.data)
        finally:
            for e in engines:
                e.stop()


if __name__ == "__main__":
    unittest.main()
