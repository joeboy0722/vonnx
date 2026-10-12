# vONNX 0.3

以 C++17、CUDA 與 ONNX Runtime 建立的常駐 YOLOv8 推論引擎，支援 **Detect、Pose、Seg**。
提供非同步請求、動態 ONNX 尺寸初始化、批次排程，以及 Python／C++／版本化 C API。
目前主要驗證環境為 **Windows + RTX 3070 8GB**。

預編譯 Windows x64 SDK／Python 3.12 wheel 見 [GitHub Releases](https://github.com/joeboy0722/vonnx/releases)。
安裝、runtime 依賴與 DLL／Python 呼叫方式見 [發布包使用指南](docs/release.md)。

## 功能

- 每筆請求有自己的結果與錯誤，保留 `stream_id`／`frame_id`；有界佇列滿載時明確拒絕。
- 初始化時指定 `imgsz=640` 或 `(height, width)`，原始影像統一 letterbox 至固定推論尺寸。
- session、CUDA streams、events、pinned staging 與各 batch 的 bindings 常駐重用。
- Detect／Pose／Seg 共用前處理、框解碼與確定性類別感知 NMS；關鍵點與 masks 對應保留的框。
- 支援 RGB24、BGR24、RGBA32、NV12；CPU 非同步輸入由引擎持有副本。
- `pause()` 排空並保留資源；`resume()` 恢復；`close()` 釋放資源，`stop()` 保持終止語意。
- 官方模型下載／匯出、正確性驗證與跑分各有獨立 CLI，結果可續跑並檢查模型／二進位雜湊。

## 建置

需要 Visual Studio 2022 C++ 工具、CUDA Toolkit 13、相容的 cuDNN 9／ONNX Runtime GPU runtime。
本機測試版本：Python 3.12、ORT GPU 1.28.0、Ultralytics 8.4.115、PyTorch 2.13.0+cu126。
詳細依賴與 CMake 建置見 [驗證與建置指南](docs/validation.md)。

全新 clone 可使用 Conda 建立環境；先依 [PyTorch 官方指令](https://pytorch.org/get-started/locally/)
安裝 GPU 版 PyTorch。已有可用環境時不需要重建或替換它。

```powershell
conda create -p .\venv python=3.12 -y
# 安裝 GPU 版 PyTorch 後：
.\venv\python.exe -m pip install -r requirements-validation.txt
.\venv\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\build.ps1
```

本專案的 Conda 環境使用 `venv\python.exe`。若使用一般 Python venv，請改用
`.venv\Scripts\python.exe`，並傳入 `build.ps1 -Python <你的python.exe>`。
建置產出 `vonnx/vonnx_py*.pyd` 與 `vonnx_standalone_sdk/vonnx.dll`；重建前關閉載入 DLL 的程序。
獨立 DLL 使用者仍需準備 CUDA／cuBLAS／cuDNN／ORT runtime 的 DLL 搜尋路徑。
環境、模型、資料集與編譯產物不包含在 Git repository。

## 快速開始

先準備三種任務的 n 模型及小資料集，再執行少量真實影像請求：

```powershell
.\venv\python.exe -m vonnx_tools prepare --scales n --datasets smoke
.\venv\python.exe examples\quick_start.py
```

Python 非同步用法：

```python
from pathlib import Path
from vonnx import EngineConfig, vONNXEngine, PixelFormat, load_image

config = EngineConfig()
config.model_path = "models/yolov8n/yolov8n.onnx"
config.imgsz = 640                 # 也可使用 (384, 640)
config.concurrency = 32
config.max_batch_size = 8
config.batch_mode = "throughput"   # 或 "latency"
config.max_latency_ms = 2
config.pipeline_depth = 2

engine = vONNXEngine(config)
try:
    image_path = next(Path("datasets/coco8/images").rglob("*.jpg"))
    image = load_image(str(image_path))  # RGB uint8、HWC
    future = engine.submit_numpy(1, 42, image, PixelFormat.RGB24)
    result = future.result(timeout=30)[0]
    print(result.frame_id, len(result.boxes))
finally:
    engine.close()
```

同步入口為 `predict_numpy()`。Pose 加入 `keypoints[box_index]`；Seg 預設提供
`masks[box_index]` 的原圖 ROI `x/y/width/height` 與 uint8 `data`，可設定 `full_masks=True`。
GPU 輸入須保留至完成，並提供實際 producer stream handle。
C API v2、舊 v1 相容方式與 C++ 用法見 [架構與 API](docs/architecture.md)。

## 初始化設定

| 設定 | 新介面預設 | 意義 |
|---|---:|---|
| `concurrency` | 32 | 已接受但尚未完成的請求上限，並非 CUDA stream 數 |
| `max_batch_size` | 8 | 每批推論上限 |
| `batch_mode` | `throughput` | `throughput` 或 `latency` |
| `max_latency_ms` | 自動 | 吞吐 2 ms／低延遲 0 ms；是湊批等待，並非總延遲保證 |
| `pipeline_depth` | 自動 | 吞吐 2 槽／低延遲 1 槽；可指定 1 或 2 |
| `gpu_memory_budget_mb` | 6144 | 引擎 buffers 與 ORT arena 預算，非整個裝置的硬限制 |
| `collect_timings` | `False` | 需要詳細 CUDA 階段計時時才開啟 |

推論尺寸、模型與容量在建立後固定，變更時須關閉並建立新引擎。
Python 輸入須為連續 uint8 陣列；NV12 為 `[H*3/2, W]`。尺寸須為正值且符合 stride。
目前以 FP32、NCHW、無內嵌 NMS 的 YOLOv8 原始 ONNX 輸出為支援範圍。

## 驗證與跑分

```powershell
# 需要已建置 DLL、n 模型及小資料集；不下載或執行完整跑分。
.\venv\python.exe -m unittest discover -s tests -p "test_*.py" -v

# s／640 的獨立短測流程：
.\venv\python.exe -m vonnx_tools prepare --scales s --datasets smoke
.\venv\python.exe -m vonnx_tools optimize --phase check
.\venv\python.exe -m vonnx_tools optimize --phase quick
# quick 六組皆勝出後，才允許執行 confirm：
.\venv\python.exe -m vonnx_tools optimize --phase confirm
```

quick 每設定 100 次請求，25 次暖機排空後計分 75 次；confirm 每設定 500 次，25 次排除，重複 3 輪。
完整驗證與跑分維持獨立指令，不會由短測自動啟動。
`run_full.ps1` 預設每案例驗證 500 張、每模型跑分最多 500 次；全部資料須明確指定 `-ValidationLimit 0`。
完整矩陣有數千案例，執行前請閱讀 [驗證與跑分指南](docs/validation.md)。

## 效能與限制

歷史 s／640 三輪確認曾在 16／32 路、batch 8 下比官方湊批快約 45%～51%；
這是指定版本與測試配置的結果，並非所有模型／尺寸的保證。
四模型 1920、FP32、batch 1 的短測，vONNX 合計約 17.76 次推論／秒、官方約 15.72。
各模型共享 GPU 運算資源，獨立引擎不提供硬體分區或延遲隔離；增加路數可能只增加排隊。
ORT 結尾同步仍保留，完全非同步模式的清理崩潰尚未解決。
完整數據、二進位範圍與比較限制見 [效能記錄](docs/performance.md)。

## 目錄

| 目錄 | 用途 |
|---|---|
| `include/`、`src/`、`bindings/` | C++／CUDA 核心、C API 與 Python binding |
| `vonnx/` | Python runtime 入口 |
| `vonnx_tools/` | 模型準備、驗證、跑分與報告 CLI |
| `examples/`、`tests/` | 精簡範例與回歸測試 |
| `tools/` | 建置、下載與可選的診斷工具 |
| `docs/` | API、驗證流程與效能記錄 |
| `models/`、`datasets/`、`runs/` | 本機快取、測試結果與歷史歸檔；不提交 Git |

引擎程式碼採 [MIT](LICENSE)。下載的模型權重與 Ultralytics 套件依其上游授權條款使用。
