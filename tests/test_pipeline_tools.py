import unittest
import contextlib
import io
from types import SimpleNamespace as NS
import numpy as np
from vonnx_tools.benchmark import cases, request_quota
from vonnx_tools.validate import coco_records


class MatrixTests(unittest.TestCase):
    def test_request_budget(self):
        for limit in (1, 100, 500):
            for lanes in (1, 4, 8, 16, 32):
                quotas = [request_quota(limit, lanes, i) for i in range(lanes)]
                self.assertEqual(sum(quotas), limit)
                self.assertLessEqual(max(quotas) - min(quotas), 1)
        self.assertIsNone(request_quota(0, 32, 0))

    def test_complete_coverage(self):
        matrix = cases([320, 640, 960], [1, 4, 8, 16, 32], 3, 30, 30)
        self.assertEqual(len(matrix), 3015)
        self.assertEqual(len({c["id"] for c in matrix}), 3015)
        native = [c for c in matrix if c["backend"] == "native"]
        mixed = [c for c in native if c["kind"] == "mixed"]
        same = [c for c in native if c["kind"] == "same-task"]
        self.assertEqual(len(mixed), 225)
        self.assertEqual(len(same), 540)
        self.assertTrue(all(len(c["models"]) == 3 for c in mixed))
        self.assertEqual(sum(c["seconds"] for c in matrix), 90450)

    def test_coco_evaluation_adapters(self):
        from pycocotools import mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        roi = NS(x=2, y=3, width=2, height=2, data=np.ones((2, 2), np.uint8))
        result = NS(
            boxes=[NS(x1=2, y1=3, x2=4, y2=5, score=0.9, class_id=0)],
            masks=[roi],
            keypoints=[[NS(x=3, y=4, confidence=0.8) for _ in range(17)]],
        )
        full = np.zeros((8, 8), np.uint8)
        full[3:5, 2:4] = 1
        segmentation = mask_utils.encode(np.asfortranarray(full))
        segmentation["counts"] = segmentation["counts"].decode("ascii")
        gt = COCO()
        gt.dataset = {
            "info": {},
            "images": [{"id": 42, "width": 8, "height": 8}],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 42,
                    "category_id": 1,
                    "bbox": [2, 3, 2, 2],
                    "area": 4,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                    "keypoints": [3, 4, 2] * 17,
                    "num_keypoints": 17,
                }
            ],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()
            for task, kind in [("detect", "bbox"), ("pose", "keypoints"), ("segment", "segm")]:
                records = coco_records(result, 42, task, [1], (8, 8))
                self.assertEqual(records[0]["category_id"], 1)
                if kind == "segm":
                    np.testing.assert_array_equal(mask_utils.decode(records[0]["segmentation"]), full)
                    records[0].pop("bbox")
                evaluation = COCOeval(gt, gt.loadRes(records), kind)
                evaluation.params.imgIds = [42]
                evaluation.evaluate()
                evaluation.accumulate()
                evaluation.summarize()
                self.assertAlmostEqual(evaluation.stats[0], 1.0)


if __name__ == "__main__":
    unittest.main()
