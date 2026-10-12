import ctypes as C
import unittest
from pathlib import Path
import numpy as np
import onnx
from onnx import helper
from test_native import fixture
from vonnx_tools.common import ROOT
import vonnx


class Config(C.Structure):
    _fields_ = (
        [("version", C.c_uint32), ("struct_size", C.c_uint32), ("model_path", C.c_char_p)]
        + [
            (k, C.c_int)
            for k in [
                "input_height",
                "input_width",
                "max_input_height",
                "max_input_width",
                "max_batch_size",
                "gpu_id",
                "max_det",
                "full_masks",
            ]
        ]
        + [(k, C.c_float) for k in ["max_latency_ms", "conf_threshold", "iou_threshold"]]
        + [(k, C.c_size_t) for k in ["queue_capacity", "max_mask_bytes"]]
    )


class ConfigV2(C.Structure):
    _fields_ = [
        ("version", C.c_uint32),
        ("struct_size", C.c_uint32),
        ("base", Config),
        ("concurrency", C.c_size_t),
        ("gpu_memory_budget_mb", C.c_size_t),
        ("batch_mode", C.c_int),
        ("pipeline_depth", C.c_int),
        ("collect_timings", C.c_int),
    ]


class Image(C.Structure):
    _fields_ = (
        [
            ("version", C.c_uint32),
            ("struct_size", C.c_uint32),
            ("data", C.c_void_p),
            ("data_size", C.c_size_t),
        ]
        + [(k, C.c_int) for k in ["width", "height", "stride", "pixel_format", "is_gpu"]]
        + [("user_stream", C.c_size_t), ("stream_id", C.c_uint64), ("frame_id", C.c_uint64)]
    )


class Box(C.Structure):
    _fields_ = [(k, C.c_float) for k in ["x1", "y1", "x2", "y2", "score"]] + [("class_id", C.c_int)]


class Keypoint(C.Structure):
    _fields_ = [(k, C.c_float) for k in ["x", "y", "confidence"]]


class Mask(C.Structure):
    _fields_ = [(k, C.c_int) for k in ["x", "y", "width", "height"]] + [
        ("data", C.POINTER(C.c_uint8)),
        ("data_size", C.c_size_t),
    ]


class Result(C.Structure):
    _fields_ = [
        ("stream_id", C.c_uint64),
        ("frame_id", C.c_uint64),
        ("count", C.c_size_t),
        ("keypoints_per_box", C.c_size_t),
        ("boxes", C.POINTER(Box)),
        ("keypoints", C.POINTER(Keypoint)),
        ("masks", C.POINTER(Mask)),
        ("task", C.c_int),
        ("_owner", C.c_void_p),
    ]


class CAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile

        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.dll = C.CDLL(str(ROOT / "vonnx_standalone_sdk/vonnx.dll"))
        cls.dll.vonnx_config_init.argtypes = [C.POINTER(Config)]
        cls.dll.vonnx_config_init.restype = None
        cls.dll.vonnx_create_engine_v1.argtypes = [C.POINTER(Config)]
        cls.dll.vonnx_create_engine_v1.restype = C.c_void_p
        cls.dll.vonnx_destroy_engine.argtypes = [C.c_void_p]
        cls.dll.vonnx_destroy_engine.restype = None
        cls.dll.vonnx_predict_v1.argtypes = [C.c_void_p, C.POINTER(Image), C.POINTER(C.POINTER(Result))]
        cls.dll.vonnx_predict_v1.restype = C.c_int
        cls.dll.vonnx_result_free.argtypes = [C.POINTER(Result)]
        cls.dll.vonnx_result_free.restype = None
        cls.dll.vonnx_last_error.restype = C.c_char_p

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make(self, task="detect", dependent=False):
        path = self.folder / (task + ".onnx")
        fixture(path, task)
        if dependent:
            model = onnx.load(path)
            model.graph.node[-1].output[0] = "expanded"
            model.graph.node.extend(
                [
                    helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
                    helper.make_node("Mul", ["expanded", "mean"], ["output0"]),
                ]
            )
            onnx.checker.check_model(model)
            onnx.save(model, path)
        cfg = Config()
        self.dll.vonnx_config_init(C.byref(cfg))
        cfg.model_path = str(path).encode("utf-8")
        cfg.input_height = cfg.input_width = 32
        cfg.max_input_height = cfg.max_input_width = 32
        cfg.max_batch_size = 4
        cfg.iou_threshold = 0.5
        handle = self.dll.vonnx_create_engine_v1(C.byref(cfg))
        self.assertTrue(handle, self.dll.vonnx_last_error())
        return handle

    def descriptor(self, image):
        d = Image()
        d.version = 1
        d.struct_size = C.sizeof(Image)
        d.data = image.ctypes.data
        d.data_size = image.nbytes
        d.width = d.height = 32
        d.stride = 96
        d.pixel_format = 1
        d.stream_id = 7
        d.frame_id = 9
        return d

    def test_v2_lifecycle_with_v1_image_result(self):
        dll = self.dll
        dll.vonnx_config_init_v2.argtypes = [C.POINTER(ConfigV2)]
        dll.vonnx_create_engine_v2.argtypes = [C.POINTER(ConfigV2)]
        dll.vonnx_create_engine_v2.restype = C.c_void_p
        for method in ("pause", "resume", "stop", "close"):
            getattr(dll, "vonnx_" + method + "_v2").argtypes = [C.c_void_p]
        path = self.folder / "v2.onnx"
        fixture(path)
        cfg = ConfigV2()
        dll.vonnx_config_init_v2(C.byref(cfg))
        self.assertEqual(cfg.base.max_batch_size, 8)
        self.assertEqual(cfg.concurrency, 32)
        cfg.base.model_path = str(path).encode("utf-8")
        cfg.base.input_height = cfg.base.input_width = 32
        cfg.base.max_input_height = cfg.base.max_input_width = 32
        handle = dll.vonnx_create_engine_v2(C.byref(cfg))
        self.assertTrue(handle, dll.vonnx_last_error())
        image = np.zeros((32, 32, 3), np.uint8)
        desc = self.descriptor(image)
        result = C.POINTER(Result)()
        try:
            self.assertEqual(dll.vonnx_pause_v2(handle), 0)
            self.assertEqual(dll.vonnx_predict_v1(handle, C.byref(desc), C.byref(result)), -1)
            self.assertFalse(result)
            self.assertEqual(dll.vonnx_resume_v2(handle), 0)
            self.assertEqual(dll.vonnx_predict_v1(handle, C.byref(desc), C.byref(result)), 0)
            dll.vonnx_result_free(result)
            self.assertEqual(dll.vonnx_close_v2(handle), 0)
            self.assertEqual(dll.vonnx_resume_v2(handle), -1)
            self.assertIn(b"closed", dll.vonnx_last_error())
        finally:
            dll.vonnx_destroy_engine(handle)

    def test_host_results_and_errors(self):
        for task in ["detect", "pose", "segment"]:
            handle = self.make(task)
            image = np.full((32, 32, 3), 200, np.uint8)
            desc = self.descriptor(image)
            result = C.POINTER(Result)()
            try:
                self.assertEqual(
                    self.dll.vonnx_predict_v1(handle, C.byref(desc), C.byref(result)),
                    0,
                    self.dll.vonnx_last_error(),
                )
                self.assertEqual(result.contents.count, 2)
                self.assertEqual(result.contents.frame_id, 9)
                if task == "pose":
                    self.assertEqual(result.contents.keypoints_per_box, 2)
                    self.assertEqual(result.contents.keypoints[0].x, 2)
                if task == "segment":
                    self.assertTrue(result.contents.masks)
                    self.assertGreater(result.contents.masks[0].data_size, 0)
                self.dll.vonnx_result_free(result)
                desc.data_size = 1
                result = C.POINTER(Result)()
                self.assertEqual(self.dll.vonnx_predict_v1(handle, C.byref(desc), C.byref(result)), -1)
                self.assertFalse(result)
                self.assertIn(b"too small", self.dll.vonnx_last_error())
            finally:
                self.dll.vonnx_destroy_engine(handle)

    def test_gpu_pointer_and_external_stream(self):
        self._gpu_pointer_producer(True)

    def test_gpu_pointer_and_default_stream(self):
        self._gpu_pointer_producer(False)

    def _gpu_pointer_producer(self, external):
        import torch

        handle = self.make(dependent=True)
        source = torch.full((32, 32, 3), 200, dtype=torch.uint8, pin_memory=True)
        gpu = torch.zeros_like(source, device="cuda")
        torch.cuda.synchronize()
        stream = torch.cuda.Stream() if external else torch.cuda.default_stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
            gpu.copy_(source, non_blocking=True)
        desc = Image()
        desc.version = 1
        desc.struct_size = C.sizeof(desc)
        desc.data = gpu.data_ptr()
        desc.data_size = gpu.numel()
        desc.width = desc.height = 32
        desc.stride = 96
        desc.pixel_format = 1
        desc.is_gpu = 1
        desc.user_stream = stream.cuda_stream
        result = C.POINTER(Result)()
        try:
            self.assertEqual(
                self.dll.vonnx_predict_v1(handle, C.byref(desc), C.byref(result)),
                0,
                self.dll.vonnx_last_error(),
            )
            self.assertEqual(result.contents.count, 2)
            self.dll.vonnx_result_free(result)
        finally:
            self.dll.vonnx_destroy_engine(handle)


if __name__ == "__main__":
    unittest.main()
