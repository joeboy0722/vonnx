import os
import sys
import glob
import shutil
import subprocess
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# 自動尋找 Windows MSVC cl.exe 的工具函數
def find_msvc_cl_dir():
    if shutil.which("cl"):
        return None  # cl 已在系統 PATH 中

    search_patterns = [
        "C:/Program Files/Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*/bin/Hostx64/x64",
        "C:/Program Files (x86)/Microsoft Visual Studio/2019/*/VC/Tools/MSVC/*/bin/Hostx64/x64",
        "C:/Program Files/Microsoft Visual Studio/2019/*/VC/Tools/MSVC/*/bin/Hostx64/x64",
        "C:/Program Files (x86)/Microsoft Visual Studio/2017/*/VC/Tools/MSVC/*/bin/Hostx64/x64",
    ]

    for pattern in search_patterns:
        matches = glob.glob(pattern)
        for m in matches:
            if os.path.exists(os.path.join(m, "cl.exe")):
                return m

    vswhere_path = os.path.expandvars("%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe")
    if os.path.exists(vswhere_path):
        try:
            vs_path = subprocess.check_output(
                [
                    vswhere_path,
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                text=True,
            ).strip()
            if vs_path:
                msvc_pattern = os.path.join(vs_path, "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64")
                matches = glob.glob(msvc_pattern)
                if matches:
                    return matches[0]
        except Exception:
            pass

    return None


# 為 Windows 自動產生 onnxruntime.lib 匯入函式庫
def ensure_onnxruntime_lib(build_temp, cl_dir):
    lib_path = os.path.join(build_temp, "onnxruntime.lib")
    if not os.path.exists(lib_path):
        def_path = os.path.join(build_temp, "onnxruntime.def")
        with open(def_path, "w") as f:
            f.write("LIBRARY onnxruntime.dll\nEXPORTS\n    OrtGetApiBase\n")
        lib_exe = (
            os.path.join(cl_dir, "lib.exe")
            if (cl_dir and os.path.exists(os.path.join(cl_dir, "lib.exe")))
            else "lib.exe"
        )
        try:
            subprocess.check_call([lib_exe, f"/def:{def_path}", f"/out:{lib_path}", "/machine:x64"])
            print(f"[vONNX Build] 成功匯出產生 onnxruntime.lib 於 {lib_path}")
        except Exception as e:
            print(f"[vONNX Build] 警告: 產生 onnxruntime.lib 失敗: {e}")
    return lib_path


class CUDABuildExt(build_ext):
    def build_extensions(self):
        try:
            import pybind11

            pybind11_inc = pybind11.get_include()
        except ImportError:
            raise RuntimeError("編譯擴充模組需要 pybind11，請先執行: pip install pybind11")

        cl_dir = find_msvc_cl_dir() if os.name == "nt" else None
        if cl_dir:
            print(f"[vONNX Build] 自動定位 MSVC C++ 編譯器: {cl_dir}")

        cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")
        if not os.path.exists(cuda_path):
            cuda_matches = glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*")
            if cuda_matches:
                cuda_path = cuda_matches[-1]

        cuda_inc = os.path.join(cuda_path, "include")
        cuda_lib = os.path.join(cuda_path, "lib", "x64")

        # 尋找 Python venv 中的 onnxruntime 相關目錄
        try:
            import onnxruntime

            ort_dir = os.path.dirname(onnxruntime.__file__)
            ort_capi_dir = os.path.join(ort_dir, "capi")
        except ImportError:
            ort_capi_dir = None

        for ext in self.extensions:
            build_temp = os.path.abspath(self.build_temp)
            os.makedirs(build_temp, exist_ok=True)

            if os.name == "nt":
                ensure_onnxruntime_lib(build_temp, cl_dir)

            sources = ext.sources
            cu_sources = [s for s in sources if s.endswith(".cu")]
            cpp_sources = [s for s in sources if s.endswith(".cpp")]

            include_dirs = [
                os.path.abspath("include"),
                os.path.abspath("include/onnxruntime"),
                cuda_inc,
                pybind11_inc,
            ] + ext.include_dirs

            inc_flags = [f"-I{os.path.abspath(d)}" for d in include_dirs if os.path.exists(d)]

            obj_files = []
            # 1. 使用 nvcc 編譯 .cu 檔案為 .obj / .o
            for cu in cu_sources:
                obj_name = os.path.join(
                    build_temp, os.path.basename(cu) + (".obj" if os.name == "nt" else ".o")
                )
                nvcc_cmd = [
                    "nvcc",
                    "-c",
                    os.path.abspath(cu),
                    "-o",
                    obj_name,
                    "-O3",
                    "--fmad=false",
                    "-std=c++17",
                    "-Xcompiler",
                    "/MD /utf-8 /wd4819" if os.name == "nt" else "-fPIC",
                ]

                if cl_dir:
                    nvcc_cmd.extend(["-ccbin", cl_dir])

                nvcc_cmd.extend(inc_flags)

                print(f"[vONNX Build] 執行 NVCC 編譯 CUDA 核心: {' '.join(nvcc_cmd)}")
                header_mtime = max(os.path.getmtime(p) for p in glob.glob("include/vonnx/*"))
                if not os.path.exists(obj_name) or os.path.getmtime(obj_name) < max(
                    os.path.getmtime(cu), header_mtime, os.path.getmtime(__file__)
                ):
                    subprocess.check_call(nvcc_cmd)
                obj_files.append(obj_name)

            ext.sources = cpp_sources
            ext.include_dirs = include_dirs
            ext.extra_objects.extend(obj_files)

            if os.name == "nt":
                ext.extra_compile_args = ["/O2", "/std:c++17", "/EHsc", "/utf-8", "/wd4819"]
                # Extension DLL has no Win32 resources or private CRT manifest.
                ext.extra_link_args = ["/MANIFEST:NO"]
                ext.library_dirs.append(cuda_lib)
                ext.library_dirs.append(build_temp)
                if ort_capi_dir and os.path.exists(ort_capi_dir):
                    ext.library_dirs.append(ort_capi_dir)
                ext.libraries.extend(["cudart", "onnxruntime"])
            else:
                ext.extra_compile_args = ["-O3", "-std=c++17", "-fPIC"]
                ext.libraries.extend(["cudart", "onnxruntime"])

        super().build_extensions()


setup(
    name="vonnx",
    version="0.4.0",
    author="vONNX contributors",
    description="Resident CUDA/ONNX Runtime YOLOv8 inference with automatic FP16 preparation",
    license="MIT",
    license_files=["LICENSE", "THIRD_PARTY_NOTICES.md"],
    packages=["vonnx"],
    install_requires=["onnx==1.22.0", "onnxruntime-gpu==1.28.0", "numpy"],
    setup_requires=["pybind11"],
    ext_modules=[
        Extension(
            name="vonnx.vonnx_py",
            sources=[
                "bindings/vonnx_pybind.cpp",
                "src/paged_tensor_pool.cu",
                "src/cuda_preprocess.cu",
                "src/cuda_nms.cu",
                "src/onnx_runner.cpp",
                "src/continuous_scheduler.cpp",
                "src/vonnx_engine.cpp",
            ],
            include_dirs=[os.path.abspath("include")],
        )
    ],
    cmdclass={"build_ext": CUDABuildExt},
)
