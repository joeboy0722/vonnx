"""Build standalone DLL from the same core objects as the Python extension."""

from pathlib import Path
import os
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import onnxruntime
import setuptools
from setuptools._distutils.util import get_platform

folder = (
    ROOT
    / "build"
    / f"temp.{get_platform()}-cpython-{sys.version_info.major}{sys.version_info.minor}"
    / "Release"
)
cuda = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"))
include = [ROOT / "include", ROOT / "include/onnxruntime", cuda / "include"]
output = ROOT / "vonnx_standalone_sdk"
output.mkdir(exist_ok=True)
obj = folder / "src/vonnx_c_api.obj"
subprocess.run(
    [
        "cl",
        "/c",
        "/nologo",
        "/O2",
        "/MD",
        "/EHsc",
        "/std:c++17",
        "/utf-8",
        *[f"/I{p}" for p in include],
        str(ROOT / "src/vonnx_c_api.cpp"),
        f"/Fo{obj}",
    ],
    check=True,
)
objects = (
    [folder / (name + ".cu.obj") for name in ["paged_tensor_pool", "cuda_preprocess", "cuda_nms"]]
    + [folder / "src" / (name + ".obj") for name in ["continuous_scheduler", "onnx_runner", "vonnx_engine"]]
    + [obj]
)
subprocess.run(
    [
        "link",
        "/DLL",
        "/NOLOGO",
        "/MANIFEST:NO",
        *[str(p) for p in objects],
        f"/LIBPATH:{cuda}/lib/x64",
        f"/LIBPATH:{folder}",
        "cudart.lib",
        "onnxruntime.lib",
        f"/OUT:{output}/vonnx.dll",
        f"/IMPLIB:{folder}/vonnx.lib",
    ],
    check=True,
)
shutil.copyfile(ROOT / "include/vonnx_c_api.h", output / "vonnx_c_api.h")
for p in (Path(onnxruntime.__file__).parent / "capi").glob("*.dll"):
    shutil.copyfile(p, output / p.name)
print(output / "vonnx.dll")
