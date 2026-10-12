# 常駐資源與 s / 640 短測

新 Python / C++ `EngineConfig` 預設 `concurrency=32`、`max_batch_size=8`、
`batch_mode="throughput"`、`gpu_memory_budget_mb=6144`、`collect_timings=False`。
`max_latency_ms=-1`、`pipeline_depth=0` 表示依模式選用預設：吞吐為 2 ms / 2 槽，
低延遲 (`"latency"`) 為 0 ms / 1 槽。明確指定 0 ms 會保留。
路數限制已接受但尚未完成的請求，與 CUDA streams 數量無關。

```python
from vonnx import EngineConfig, vONNXEngine, PixelFormat, load_image
c = EngineConfig()
c.model_path = "models/yolov8s/yolov8s.onnx"
c.imgsz = 640
c.concurrency = 32
c.max_batch_size = 8
c.batch_mode = "throughput"
c.max_latency_ms = 2
c.pipeline_depth = 2
c.collect_timings = False
e = vONNXEngine(c)
try:
    image = load_image("your_image.jpg")
    futures = [e.submit_numpy(i, i, image, PixelFormat.RGB24) for i in range(32)]
    e.pause()  # 拒絕新請求，排空已接受請求；保留 session / CUDA / buffers。
    results = [f.result(timeout=30)[0] for f in futures]
    print(e.resource_stats.state, e.resource_stats.hot_path_device_allocations)
    e.resume()
finally:
    e.close()  # 釋放資源；之後不能 resume。stop() 也保持不可恢復的終止語意。
```

初始化探測使用 CPU session；解析真實輸出 shape 後，建立唯一的常駐 CUDA session，
配置獨立上傳、運算、下載 streams、每槽緩衝區與各 batch 的 bindings，並在正式運算執行緒暖機所有 batch。
ORT 使用 cuDNN `HEURISTIC` 選擇卷積算法，仍保持 FP32、TF32 關閉；輸出精度另行驗證。
暫停／恢復保留 CPU dispatch threads、CUDA 資源及 ORT 的執行執行緒上下文。
GPU producer pointers 必須保留到 future 完成；CPU 影像在 submit 返回前複製進有界 pinned pool。
上傳／運算／下載 streams 使用 `cudaStreamNonBlocking`；GPU producer stream（含 legacy stream 0）
以事件明確銜接，不依賴 default stream 的隱含同步。請傳入實際 producer stream handle。
不能在引擎 callback 內呼叫 pause / resume / stop / close，也不能同時 destroy 與使用 handle。

預算涵蓋引擎的 device buffers 與 ORT arena 上限；初始化暖機超出預算會失敗。
ORT 的 `gpu_mem_limit` 限制 arena，無法保證限制驅動或第三方 cuDNN 的所有暫時配置。
`resource_stats` 的配置計數只攔截 vONNX 的配置；初始化與目前顯存差額是 device-wide，
其他程式也會影響它。`close()` 後保留的數字是歷史統計，不代表仍占有那些 buffers。

計時關閉時 GPU 階段數字為 0。診斷模式啟用 CUDA events。
`queue_ms` 為 dispatch 前排隊時間（扣除湊批等待），`batch_wait_ms` 為實際湊批等待，
`dispatch_ms` 包含上傳與等待已派送批次取得運算槽的 CPU 時間。
Windows 使用常駐 high-resolution waitable timer；不永久設定 timeBeginPeriod。
後備 timer 可由 `resource_stats.timer_backend` 查詢。

本機 ORT 1.28 在停用 execution-provider 同步後，釋放大型模型 session 時會在 CUDA
host callback 中中止程序，因此保留 ORT 的預設同步。H2D 上傳仍可与上一批運算重疊，
前後處理使用常駐事件與 bindings；此版本不宣稱已達成完全非同步的 ORT 執行。

## C v2 與舊介面

C++ 使用同一組初始化設定；模型路徑、尺寸與容量在建立後固定：

```cpp
vonnx::EngineConfig c;
c.model_path = "models/yolov8s/yolov8s.onnx";
c.input_height = c.input_width = 640;
c.concurrency = 32;
c.max_batch_size = 8;
c.batch_mode = "throughput";
c.max_latency_ms = 2;
c.pipeline_depth = 2;
vonnx::vONNXEngine engine(c);
// FrameTask 必須提供 data_ptr、data_size、width、height、step、format。
auto future = engine.predict_async(frame);
auto results = future.get();
engine.pause();
engine.resume();
engine.close();
```

既有 C 呼叫可繼續使用：

```c
vonnx_config_v1 legacy;
vonnx_config_init(&legacy);
legacy.model_path = "models/yolov8s/yolov8s.onnx";
vonnx_engine_handle_t engine = vonnx_create_engine_v1(&legacy);
/* 原 bbox API 僅適用 Detect；通用 v1 結果可提供 Pose / Seg。 */
vonnx_destroy_engine(engine);
```

```c
vonnx_config_v2 c;
vonnx_config_init_v2(&c);
c.base.model_path = "models/yolov8s/yolov8s.onnx";
c.concurrency = 32;
c.base.max_batch_size = 8;
c.pipeline_depth = 2;
vonnx_engine_handle_t e = vonnx_create_engine_v2(&c);
/* 使用既有 vonnx_image_v1 / vonnx_predict_v1 / vonnx_result_free。 */
vonnx_pause_v2(e);
vonnx_resume_v2(e);
vonnx_close_v2(e);
vonnx_destroy_engine(e);
```

v1 的結構布局與函式保留；`vonnx_config_init()` 維持 batch 16、等待 2 ms。
v1 建立的引擎保留一槽與舊 queue 上限語意。所有失敗由 `vonnx_last_error()` 查詢。
新 v2 資源查詢的 descriptor 需設定 version=2 與 struct_size=sizeof(descriptor)。

## 獨立短測 CLI

```powershell
.\venv\python.exe -m vonnx_tools optimize --phase check
.\venv\python.exe -m vonnx_tools optimize --phase quick
.\venv\python.exe -m vonnx_tools optimize --phase diagnostic
# 六個 quick 吞吐比較全部勝出後，才允許 500 次 × 3 的確認。
.\venv\python.exe -m vonnx_tools optimize --phase confirm
.\venv\python.exe -m vonnx_tools optimize --phase report
```

每個 quick / diagnostic 設定共 100 次請求；前 25 次暖機完成、排空，之後 75 次計分。
confirm 為每設定 500 次，前 25 次排除，475 次計分，重複 3 次。
所有影像均為預載的 COCO8 / COCO8-pose / COCO8-seg；不計下載、解碼、初始化。
正式比較关闭詳細 timing；另測 1 / 2 槽、0 / 2 ms 的診斷，避免計時成本混入主成績。
官方基準有独立常駐 predictor 與 dispatcher，採相同 deadline timer，FP32 / TF32 關閉，
conf=.25、IoU=.7、max_det=300、rect=False、CPU boxes / keypoints / ROI uint8 masks。
每一路最多一筆未完成請求。

每輪（相同 lanes / repeat 的三種模型設定）累計 child wall time 限制 300 秒，
逾時／OOM／程序或請求失敗即停，不降 batch、不延伸完整驗證。
`--list` 顯示設定、`--case ID` 單獨測、`--resume` 略過已通過案例。
結果寫到 `runs/optimization`：逐請求 JSONL、案例 JSON、summary.csv、REPORT.md 與 throughput.png。
吞吐驗收分別使用 16 / 32 個 request lanes，引擎的 admission 容量維持預設 32；每 lane 至多一筆未完成請求。
500 次門檻：三種 s 在 16 / 32 路的 FPS 中位數皆超過官方湊批，且各至少 2 / 3 輪勝出；
不足 5% 標示小幅優勢。單路延遲独立呈現。

`--phase baseline` 只供開發期比較使用，需要 `runs/optimization/baseline/vonnx` 的舊二進位快照。
重新 build 不會產生此快照。完整驗證與完整跑分維持原有獨立 CLI，短測不串接它們。
`run_full.ps1` 預設驗證上限與跑分請求上限均為 500；只有手動指定 `-ValidationLimit 0`
才使用全部 val2017，`-BenchmarkRequests 0` 則恢復純時間限制。請先審閱短測與時間估算，再決定是否執行。

參考：[Windows timer](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-createwaitabletimerexw)、
[ORT CUDA stream / arena / async execution](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

## 1920 異構模型追查

兩個 Detect n＋Detect s＋Pose l 的純 C++ 對照與 Nsight trace 記錄於
`runs/stream_isolation/REPORT.md`。四引擎 batch 1／單槽／FP32，CPU 早已提交工作，
小模型仍在 GPU 上排隊；non-blocking streams、最高 CUDA priority 與停用 ORT 結尾同步
均未提供實質吞吐改善。停用 ORT 同步另有清理時崩潰，未採用於正式 DLL。
獨立 session／stream 不等於跨模型 GPU 延遲隔離；此版本尚未提供 GPU 全局服務預算。

先前 s／640 三輪跑分屬於報告記錄的舊二進位。本次 stream 隔離改動通過 22 項回歸與
新二進位三種 s／640 的小樣本正確性比對，未重跑完整效能矩陣；CLI 會檢查 binary hash
以防誤用舊驗證結果。可重跑的 C++ 對照原始碼在 `tools/diagnostics/multimodel_1920_native.cpp`，
操作說明見 [診斷工具](../tools/diagnostics/README.md)。一次性實驗腳本與 DLL 留在本機
`runs/archive/`、`runs/stream_isolation/diagnostic_dlls`，不發布或放入正式 SDK。
