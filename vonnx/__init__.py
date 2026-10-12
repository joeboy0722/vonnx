"""Native vONNX SDK. CUDA/ORT directories remain registered for module lifetime."""

import os
import sys
from pathlib import Path

_DLL_HANDLES = []
if sys.platform == "win32":
    candidates = [
        Path(sys.prefix) / "Lib/site-packages/onnxruntime/capi",
        Path(sys.prefix) / "Lib/site-packages/torch/lib",
        Path(sys.prefix) / "Library/bin",
    ]
    cuda = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"))
    candidates += [cuda / "bin", cuda / "bin/x64"]
    candidates += [Path(p) for p in os.environ.get("VONNX_DLL_DIRS", "").split(os.pathsep) if p]
    for path in candidates:
        if path.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(path)))
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
from .vonnx_py import (
    vONNXEngine,
    EngineConfig,
    PixelFormat,
    BBox,
    DetectionResult,
    Keypoint,
    Mask,
    ModelSpec,
    PredictionFuture,
    load_image,
    DeadlineWaiter,
    ResourceStats,
)

__version__ = "0.3.0"
__all__ = [
    "vONNXEngine",
    "EngineConfig",
    "PixelFormat",
    "BBox",
    "DetectionResult",
    "Keypoint",
    "Mask",
    "ModelSpec",
    "PredictionFuture",
    "load_image",
    "DeadlineWaiter",
    "ResourceStats",
]
