# vonnx Engine Python High-Level SDK
import os
import sys
import glob

# 在 Windows 上自動向 Python 載入器註冊 ONNX Runtime、CUDA 與 cuDNN 的 DLL 搜尋路徑
if sys.platform == "win32":
    # 1. 註冊 ONNX Runtime capi
    try:
        import onnxruntime
        ort_capi_dir = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
        if os.path.exists(ort_capi_dir):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(ort_capi_dir)
                except Exception:
                    pass
            os.environ["PATH"] = ort_capi_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    # 2. 註冊 CUDA Toolkit bin 目錄
    cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")
    cuda_bin = os.path.join(cuda_path, "bin")
    if os.path.exists(cuda_bin):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(cuda_bin)
            except Exception:
                pass
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

    # 3. 搜尋並註冊系統與 Anaconda 中的 cuDNN (cudnn64_9.dll / cudnn*.dll)
    user_home = os.path.expanduser("~")
    search_dirs = [
        os.path.join(user_home, "anaconda3", "envs", "text_translate", "Lib", "site-packages", "torch", "lib"),
        os.path.join(user_home, "anaconda3", "envs", "py39", "Library", "bin"),
    ] + glob.glob(os.path.join(user_home, "anaconda3", "envs", "*", "Lib", "site-packages", "torch", "lib"))

    for d in search_dirs:
        if os.path.exists(d) and any(f.lower().startswith("cudnn") for f in os.listdir(d)):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

try:
    from .vonnx_py import vONNXEngine, EngineConfig, PixelFormat, BBox, DetectionResult, load_image
except ImportError as e:
    raise ImportError(
        "\n\n[vONNX 錯誤提醒] 無法載入 compiled C++/CUDA 模組 `vonnx.vonnx_py`！\n"
        "這是因為 C++/CUDA 原始碼尚未被編譯為 .pyd / .so 檔案。\n\n"
        "請在您的 venv 環境中執行以下指令進行手動編譯：\n"
        "    pip install --no-build-isolation -e .\n"
        f"原始錯誤訊息: {e}"
    )

__version__ = "0.1.0"
__all__ = ["vONNXEngine", "EngineConfig", "PixelFormat", "BBox", "DetectionResult", "load_image"]
