# vONNX 0.3.0 Windows x64 發布包

## 授權與依賴

引擎使用 MIT LICENSE；第三方聲明見 `THIRD_PARTY_NOTICES.md`。
包內不包含模型、資料集或第三方 runtime DLL。模型／Ultralytics 的授權獨立於本引擎。

此預編譯版本需要 Windows x64、NVIDIA GPU、Visual C++ 2022 x64 runtime、
CUDA 13 runtime (`cudart64_13.dll`)、ONNX Runtime GPU 1.28.0、CUDA 12 cuBLAS 與 cuDNN 9。
Python wheel 僅供 CPython 3.12 x64。不同 Python／CUDA 組合請從原始碼建置。
ORT 的 `onnxruntime.dll`、`onnxruntime_providers_shared.dll`、
`onnxruntime_providers_cuda.dll` 及其依賴必須可載入。
Python 可透過 `VONNX_DLL_DIRS` 指定多個 DLL 目錄（Windows 用分號分隔）。
原生範例也使用 `VONNX_DLL_DIRS` 與 `AddDllDirectory`；不要混用不相容版本。

## Python 安裝與呼叫

```powershell
python -m pip install .\vonnx-0.3.0-cp312-cp312-win_amd64.whl
python -m pip install onnxruntime-gpu==1.28.0 numpy
$env:VONNX_DLL_DIRS = 'C:\runtime\ort;C:\runtime\cuda;C:\runtime\cudnn'
```

```python
from vonnx import EngineConfig, vONNXEngine, PixelFormat, load_image

image = load_image("image.jpg")  # RGB uint8 HWC
config = EngineConfig()
config.model_path = "yolov8s.onnx"  # pose/seg 同樣使用此介面
config.imgsz = 640               # 或 (384, 640)，初始化後固定
config.max_input_height, config.max_input_width = image.shape[:2]
config.concurrency = 32
config.max_batch_size = 8
config.batch_mode = "throughput"
config.max_latency_ms = 2
config.pipeline_depth = 2
engine = vONNXEngine(config)
try:
    futures = [engine.submit_numpy(i, i, image, PixelFormat.RGB24) for i in range(16)]
    for future in futures:
        result = future.result(timeout=30)[0]
        print(result.frame_id, len(result.boxes), result.keypoints, result.masks)
    engine.pause()   # 排空、保留資源
    engine.resume()
finally:
    engine.close()   # 關閉後不可恢复
```

CPU 非同步輸入會複製；GPU 輸入須保持有效直到完成，並傳入 producer stream。
所有原始影像必須在初始化最大輸入尺寸內；路數是未完成請求容量，不是 GPU 隔離。

## C / C++ DLL 呼叫

SDK zip 包含 `vonnx.dll`、`vonnx.lib`、`vonnx_c_api.h` 與 `c_api.cpp`。
在 VS 2022 x64 Native Tools 命令列：

```powershell
cl /std:c++17 /EHsc /MD /I. c_api.cpp vonnx.lib delayimp.lib /Fe:c_api.exe /link /DELAYLOAD:vonnx.dll
.\c_api.exe yolov8s.onnx image.rgb 640 480
```

`image.rgb` 是未壓縮 RGB24，可用 Python `load_image("image.jpg").tofile("image.rgb")` 產生。
`vonnx_predict_v1` 是阻塞呼叫；多個執行緒可以對同一 v2 engine 同時提交以形成批次。
使用 `vonnx_config_init_v2` 設定預設值，`vonnx_create_engine_v2` 建立引擎；
失敗時立即讀取執行緒自己的 `vonnx_last_error()`。
結果使用 `vonnx_result_free` 釋放；結果釋放後其 boxes/keypoints/masks 指標失效。
pixel_format：0=BGR24、1=RGB24、2=RGBA32、3=NV12。
結束時先等待所有呼叫完成再 `vonnx_destroy_engine`，不能與推論同時銷毀。
v2 handle 使用 v1 影像與通用結果結構；舊 bbox-only API 僅支援 Detect。

## 模型與驗證工具

GitHub tag 的原始碼包包含 CLI，wheel 僅安裝推論模組。在原始碼目錄：

```powershell
python -m pip install -r requirements-validation.txt
python -m vonnx_tools prepare --scales s --datasets smoke
python examples/quick_start.py --model models/yolov8s/yolov8s.onnx
```

ONNX 須為 FP32 NCHW YOLOv8 原始輸出、nms=False；支援 Detect/Pose/Seg。
完整驗證／跑分為獨立指令，詳細見 `docs/validation.md`；不會自動開始。
ORT 結尾同步仍保留；多引擎共享 GPU 算力／頻寬，並不提供硬體隔離。
既有跑分的版本與範圍見 `docs/performance.md`，不代表所有硬體都能得到相同增幅。
