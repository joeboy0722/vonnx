"""Small deterministic GPU fixtures; no model downloads or long benchmarks."""

import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from vonnx import EngineConfig, vONNXEngine, PixelFormat, load_image


def fixture(path, task="detect", static=False):
    extra = 6 if task == "pose" else 2 if task == "segment" else 0
    pred = np.zeros((1, 5 + extra, 4), np.float32)
    pred[0, 0] = [5, 7, 9, 25]
    pred[0, 1] = 5
    pred[0, 2] = 10
    pred[0, 3] = 10
    pred[0, 4] = [0.9, 0.8, 0.7, 0.1]
    if task == "pose":
        pred[0, 5:] = np.array([[2] * 4, [3] * 4, [0.7] * 4, [8] * 4, [9] * 4, [0.8] * 4])
    if task == "segment":
        pred[0, 5] = [1, 1, -1, 1]
    nodes = [
        helper.make_node("Shape", ["images"], ["shape"]),
        helper.make_node("Gather", ["shape", "zero"], ["batch"], axis=0),
        helper.make_node("Unsqueeze", ["batch", "zero"], ["batch_vector"]),
        helper.make_node("Concat", ["batch_vector", "tail"], ["output_shape"], axis=0),
        helper.make_node("Expand", ["prediction", "output_shape"], ["output0"]),
    ]
    init = [
        numpy_helper.from_array(np.array([0], np.int64), "zero"),
        numpy_helper.from_array(np.array([5 + extra, 4], np.int64), "tail"),
        numpy_helper.from_array(pred, "prediction"),
    ]
    # Gather with indices [0] already gives [1]; Unsqueeze is unnecessary here.
    nodes.pop(2)
    nodes[2].input[0] = "batch"
    outputs = [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", 5 + extra, 4])]
    if task == "segment":
        init += [
            numpy_helper.from_array(np.ones((1, 2, 8, 8), np.float32), "proto"),
            numpy_helper.from_array(np.array([2, 8, 8], np.int64), "proto_tail"),
        ]
        nodes += [
            helper.make_node("Concat", ["batch", "proto_tail"], ["proto_shape"], axis=0),
            helper.make_node("Expand", ["proto", "proto_shape"], ["output1"]),
        ]
        outputs += [helper.make_tensor_value_info("output1", TensorProto.FLOAT, ["batch", 2, 8, 8])]
    graph = helper.make_graph(
        nodes,
        "fixture",
        [
            helper.make_tensor_value_info(
                "images",
                TensorProto.FLOAT,
                [1 if static else "batch", 3, 32 if static else "height", 32 if static else "width"],
            )
        ],
        outputs,
        init,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    helper.set_model_props(
        model,
        {
            "task": task,
            "names": "{0: 'person'}",
            "stride": "32",
            **({"kpt_shape": "[2, 3]"} if task == "pose" else {}),
        },
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make(self, task="detect", static=False, batch=4, **settings):
        path = self.root / (task + str(static) + ".onnx")
        fixture(path, task, static)
        c = EngineConfig()
        c.model_path = str(path)
        c.imgsz = 32
        c.max_batch_size = batch
        c.max_input_width = c.max_input_height = 64
        c.capture_debug = True
        c.iou_threshold = 0.5
        for name, value in settings.items():
            setattr(c, name, value)
        return vONNXEngine(c)

    def test_empty_and_worker_failure_dispatch(self):
        image = np.zeros((32, 32, 3), np.uint8)
        e = self.make(conf_threshold=1.0)
        try:
            self.assertEqual(len(e.predict_numpy(0, 7, image, PixelFormat.RGB24)[0].boxes), 0)
        finally:
            e.stop()
        e = self.make("segment", max_mask_bytes=1)
        try:
            for iteration in range(2):
                futures = [e.submit_numpy(i, iteration, image, PixelFormat.RGB24) for i in range(4)]
                for future in futures:
                    with self.assertRaisesRegex(RuntimeError, "max_mask_bytes"):
                        future.result(timeout=10)
        finally:
            e.stop()

    def test_greedy_chain_and_async_ownership(self):
        e = self.make()
        image = np.full((32, 32, 3), 77, np.uint8)
        try:
            futures = [e.submit_numpy(i, i, image, PixelFormat.RGB24) for i in range(4)]
            image.fill(0)
            results = [f.result(timeout=10)[0] for f in futures]
            self.assertEqual([r.frame_id for r in results], list(range(4)))
            for r in results:
                self.assertEqual(len(r.boxes), 2)
                self.assertAlmostEqual(r.boxes[0].x1, 0)
                self.assertAlmostEqual(r.boxes[1].x1, 4)
                np.testing.assert_allclose(r.debug_input, 77 / 255, atol=1e-7)
        finally:
            e.stop()

    def test_formats_and_validation(self):
        e = self.make()
        try:
            rgb = np.full((32, 32, 3), 128, np.uint8)
            rgba = np.concatenate([rgb, np.zeros((32, 32, 1), np.uint8)], axis=2)
            nv12 = np.full((48, 32), 128, np.uint8)
            for fmt, image in [
                (PixelFormat.RGB24, rgb),
                (PixelFormat.BGR24, rgb),
                (PixelFormat.RGBA32, rgba),
                (PixelFormat.NV12, nv12),
            ]:
                r = e.predict_numpy(0, 0, image, fmt)[0]
                np.testing.assert_allclose(r.debug_input, 128 / 255, atol=1e-7)
            for image in [rgb[:, ::2], rgb.astype(np.float32), np.zeros((65, 32, 3), np.uint8)]:
                with self.assertRaises((RuntimeError, TypeError, ValueError)):
                    e.predict_numpy(0, 0, image, PixelFormat.RGB24)
        finally:
            e.stop()

    def test_pose_and_segment_indices(self):
        image = np.full((32, 32, 3), 128, np.uint8)
        for task in ("pose", "segment"):
            e = self.make(task)
            try:
                r = e.predict_numpy(0, 0, image, PixelFormat.RGB24)[0]
                if task == "pose":
                    self.assertEqual(len(r.keypoints), 2)
                    self.assertEqual(r.keypoints[1][0].x, 2)
                else:
                    self.assertEqual(len(r.masks), 2)
                    self.assertTrue(np.all(r.masks[0].data == 1))
                    self.assertTrue(np.all(r.masks[1].data == 0))
            finally:
                e.stop()

    def test_unicode_image_path(self):
        import cv2

        success, encoded = cv2.imencode(".jpg", np.zeros((17, 23, 3), dtype=np.uint8))
        self.assertTrue(success)
        path = self.root / "圖片.jpg"
        path.write_bytes(encoded.tobytes())
        image = load_image(str(path))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.shape[2], 3)
        self.assertEqual(image.shape[:2], (17, 23))

    def test_static_and_stop(self):
        with self.assertRaises(ValueError):
            self.make(static=True, batch=4)
        e = self.make(static=True, batch=1)
        e.stop()
        e.stop()
        with self.assertRaises(RuntimeError):
            e.start()
        with self.assertRaises(RuntimeError):
            e.submit_numpy(0, 0, np.zeros((32, 32, 3), np.uint8), PixelFormat.RGB24)

    def test_pause_resume_drain_and_resource_reuse(self):
        e = self.make(max_latency_ms=2, pipeline_depth=2)
        image = np.full((32, 32, 3), 77, np.uint8)
        initial = e.resource_stats
        try:
            for cycle in range(3):
                futures = [e.submit_numpy(i, cycle * 8 + i, image, PixelFormat.RGB24) for i in range(8)]
                e.pause()
                self.assertEqual(e.resource_stats.state, "paused")
                for i, f in enumerate(futures):
                    r = f.result(timeout=10)[0]
                    self.assertEqual(r.frame_id, cycle * 8 + i)
                    np.testing.assert_allclose(r.debug_input, 77 / 255, atol=1e-7)
                with self.assertRaisesRegex(RuntimeError, "paused"):
                    e.submit_numpy(0, 0, image, PixelFormat.RGB24)
                e.resume()
                now = e.resource_stats
                self.assertEqual(now.initialization_count, 1)
                self.assertEqual(now.explicit_device_allocations, initial.explicit_device_allocations)
                self.assertEqual(now.hot_path_device_allocations, 0)
                self.assertEqual(now.reserved_device_bytes, initial.reserved_device_bytes)
            e.close()
            e.close()
            self.assertEqual(e.resource_stats.state, "closed")
            for action in (e.resume, e.start):
                with self.assertRaises(RuntimeError):
                    action()
        finally:
            e.close()

    def test_two_slots_do_not_overwrite_inputs(self):
        e = self.make(max_latency_ms=0, pipeline_depth=2, concurrency=32)
        try:
            futures = []
            for i in range(32):
                image = np.full((32, 32, 3), i * 7, np.uint8)
                futures.append(e.submit_numpy(i, i, image, PixelFormat.RGB24))
                image.fill(0)
            for i, f in enumerate(futures):
                r = f.result(timeout=10)[0]
                self.assertEqual(r.stream_id, i)
                np.testing.assert_allclose(r.debug_input, i * 7 / 255, atol=1e-7)
        finally:
            e.close()

    def test_concurrency_budget_and_optional_timings(self):
        image = np.zeros((32, 32, 3), np.uint8)
        e = self.make(concurrency=1, max_latency_ms=1000, capture_debug=False)
        try:
            f = e.submit_numpy(0, 1, image, PixelFormat.RGB24)
            with self.assertRaisesRegex(RuntimeError, "Concurrency capacity"):
                e.submit_numpy(1, 2, image, PixelFormat.RGB24)
            e.pause()  # Flush the pending deadline immediately.
            r = f.result(timeout=10)[0]
            self.assertEqual(r.inference_ms, 0)
            e.resume()
        finally:
            e.close()
        with self.assertRaisesRegex(RuntimeError, "budget"):
            self.make(gpu_memory_budget_mb=1)
        for settings in ({"pipeline_depth": 3}, {"batch_mode": "invalid"}, {"max_latency_ms": -2}):
            with self.assertRaises(ValueError):
                self.make(**settings)
        e = self.make(collect_timings=True)
        try:
            self.assertGreater(e.predict_numpy(0, 0, image, PixelFormat.RGB24)[0].inference_ms, 0)
        finally:
            e.close()


if __name__ == "__main__":
    unittest.main()
