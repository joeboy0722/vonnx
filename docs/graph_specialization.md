# 初始化圖專化（v0.5.0、預設關閉）

`graph_specialization="auto"` 讓引擎在初始化時固定已知的模型輸入尺寸，提前處理可確定的 shape／常數路徑。來源模型不改寫，產物寫入 `model_cache_dir`。它消除軟體往返，不能提供 GPU 硬體隔離。

```python
config.graph_specialization = "auto"  # 預設 "off"
config.precision = "fp16"             # 也支援 fp32 / auto / 原生 FP16 I/O
config.imgsz = (384, 640)
config.batch_shape_policy = "fixed"
config.max_batch_size = 8
config.cuda_graph = False             # 獨立選用，預設關閉
```

專化只固定引擎已固定的高／寬；動態執行 batch 保留動態，固定執行 batch 才專化 batch。不足固定批次仍補零，僅回傳真實請求。靜態來源模型的尺寸／batch 限制不會被放寬。初始化探測、binding、暖機與實際執行尺寸一致。

快取包含來源與外部權重雜湊、尺寸、batch、精度、處理順序、工具及規則版本；鎖保護、內容雜湊檢查與原子發布避免並行讀取半成品。不可寫、來源不相容或準備失敗明確回報，不靜默退回。原本不需轉換的 FP32／原生 FP16 使用者，啟用此功能也需要既有 `precision.py` 與 ONNX／ORT 準備依賴。

`model_spec` 提供 `graph_specialized`、`source_input_shape`、`execution_input_shape`、`source_model_sha256`、`prepared_model_sha256`、`prepared_model_path`、`cuda_graph`。來源 shape 的 -1 表示動態；公開 shape／dtype 描述不是 GPU 內部累加精度保證。

既有 `output_shapes` 保留每張影像容量的表示（batch 軸為 1）；實際綁定的輸出 batch 軸依執行 batch 設定。準備後 ONNX 的靜態輸入／輸出以檔案本身與 `execution_input_shape` 為準，來源匯出 metadata 保留供追溯。

## CUDA Graph

必須同時設定 `graph_specialization="auto"` 且執行 batch 固定（含 batch 上限 1）。如果目前策略會使用動態執行 batch，上限大於 1 時須明確使用 `batch_shape_policy="fixed"`；不會自動改動 batch 策略。

每個常駐槽位有自己的 graph ID 與固定 I/O 位址，運行期間單一 ORT session 的 Run 保持有序。初始化先用暫時的 CUDA session 探測 shape，再以最終 arena 預算建立常駐 session；探測時關閉 capture，捕捉與暖機在初始化完成。pause／resume 重用原資源。CPU fallback、capture 失敗、預算不足均報錯；不自動關閉 Graph。ORT 結尾同步持續保留。Graph 只涵蓋 ORT 模型，前後處理及必要拷貝仍照常完成。

## C++ 與 C API

C++ 使用同名 `EngineConfig` 欄位，使用前須以更新後標頭重編譯。

```cpp
vonnx_config_v4 config{};
vonnx_config_init_v4(&config);
config.base.base.base.model_path = model_path_utf8;
config.base.base.base.max_batch_size = 8;
config.base.batch_shape_policy = 2;  // fixed
config.base.precision = 2;          // fp16
config.base.precision_python = python_path_utf8;
config.base.model_cache_dir = cache_path_utf8;
config.graph_specialization = 1;    // auto
config.cuda_graph = 0;
auto engine = vonnx_create_engine_v4(&config);
// NULL: vonnx_last_error(); otherwise existing v1 prediction/result functions.
vonnx_model_info_v4 info{};
info.version = 4; info.struct_size = sizeof(info);
vonnx_model_info_get_v4(engine, &info);
```

v1／v2／v3 布局與預設保持不變；既有結果結構不擴張。v4 資訊查詢只新增專化／Graph 與雜湊，精度與模型路徑繼續使用 v3 查詢。

## 生命週期

session、arena、槽位與 CUDA streams／events 在 close 時排空並釋放。ORT Env／logging 採每個載入模組一份的程序生命週期配置，避免關閉最後引擎後 logger 失效，及 Windows／Python DLL 解載順序導致的退出崩潰；不逐引擎新增 Env。pause／resume 不重新準備模型、capture 或初始化。

範例：`examples/graph_specialization.py`。功能與速度是否通過本輪門檻，以 `docs/software_wait.md` 的測試範圍、二進位及限制為準；本版另有 [500張抽樣AP回歸](ap_v050.md)，不代表完整AP驗收。
