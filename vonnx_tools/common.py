from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/ultralytics"))
TASKS = ("detect", "pose", "segment")
SCALES = ("n", "s", "m", "l", "x")
RESOLUTIONS = (320, 640, 960)
STREAMS = (1, 4, 8, 16, 32)
VERSIONS = {"ultralytics": "8.4.115", "onnx": "1.22.0", "onnxruntime-gpu": "1.28.0"}


def names(tasks=TASKS, scales=SCALES):
    return [
        f"yolov8{s}" + ("" if t == "detect" else "-pose" if t == "pose" else "-seg")
        for t in tasks
        for s in scales
    ]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def environment():
    import importlib.metadata as metadata

    versions = {}
    for p in ("ultralytics", "onnx", "onnxruntime-gpu", "torch", "numpy", "opencv-python", "pycocotools"):
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = "missing"
    import subprocess

    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True
    ).strip()
    return {"python": sys.version, "platform": platform.platform(), "gpu": gpu, "versions": versions}


def manifest(model):
    path = ROOT / "models" / model / "manifest.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    if sha256(ROOT / obj["weight"]) != obj["weight_sha256"]:
        raise RuntimeError(f"Weight hash mismatch: {model}")
    if sha256(ROOT / obj["onnx"]) != obj["onnx_sha256"]:
        raise RuntimeError(f"Model hash mismatch: {model}")
    return obj


def engine(
    model, imgsz, *, batch=16, conf=0.25, debug=False, full_masks=False, max_width=1920, max_height=1080,
    precision="auto", model_path=None, graph_specialization="off", batch_shape_policy="auto",cuda_graph=False
):
    from vonnx import EngineConfig, vONNXEngine

    c = EngineConfig()
    c.model_path = str(model_path or ROOT / manifest(model)["onnx"])
    c.precision = precision
    if hasattr(c,"graph_specialization"):c.graph_specialization=graph_specialization
    elif graph_specialization!="off":raise RuntimeError("Binary does not support graph specialization")
    c.batch_shape_policy=batch_shape_policy
    if hasattr(c,'cuda_graph'):c.cuda_graph=cuda_graph
    elif cuda_graph:raise RuntimeError('Binary does not support CUDA Graph')
    c.model_cache_dir = str(ROOT / "runs/fp16/cache")
    c.imgsz = imgsz
    c.max_batch_size = batch
    c.conf_threshold = conf
    c.iou_threshold = 0.7
    c.max_det = 300
    c.max_input_width = max_width
    c.max_input_height = max_height
    c.capture_debug = debug
    c.full_masks = full_masks
    return vONNXEngine(c)


def images(task, full=False):
    if full:
        return sorted((ROOT / "datasets/coco/val2017").glob("*.jpg"))
    folder = {"detect": "coco8", "pose": "coco8-pose", "segment": "coco8-seg"}[task]
    return sorted((ROOT / "datasets" / folder / "images").rglob("*.jpg"))


def load_rgb(paths):
    import cv2
    import numpy as np

    result = []
    for path in paths:
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot decode {path}")
        result.append(np.ascontiguousarray(bgr[:, :, ::-1]))
    if not result:
        raise RuntimeError("Dataset missing; run prepare --datasets smoke/full")
    return result


def build_fingerprint():
    binaries = list((ROOT / "vonnx").glob("vonnx_py*.pyd")) + list((ROOT / "vonnx").glob("vonnx_py*.so"))
    if len(binaries) != 1:
        raise RuntimeError("Expected one compiled native extension; rebuild SDK")
    return sha256(binaries[0])
