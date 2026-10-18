# 效能記錄與測試範圍

測試日期：2026-10-12。Windows 10、RTX 3070 8GB、驅動 591.59、CUDA Toolkit 13、
ORT GPU 1.28.0、Ultralytics 8.4.115、PyTorch 2.13.0+cu126。FP32、TF32 關閉。
下載、解碼、初始化與暖機不計入吞吐；輸出完成以 CPU boxes／keypoints／ROI masks 可用為準。
本機原始 JSON／JSONL、雜湊、CUDA trace 與報告保留在 `runs/`，不與程式碼一起提交。

## s／640 歷史三輪確認

三種 s，batch 上限 8、16／32 請求路數、兩槽、湊批 2 ms、詳細計時關閉。
每輪 500 次，前 25 次完成並排空後計分其餘 475 次。以下為三輪 FPS 中位數。
每種任務使用預載官方 COCO8 對應影像，官方對照為獨立常駐 PyTorch predictor 與等效湊批。

| 模型 | 路數 | vONNX FPS | 官方湊批 FPS | 增幅 | 勝出 |
|---|---:|---:|---:|---:|---:|
| Detect s | 16 | 200.56 | 137.94 | +45.4% | 3/3 |
| Detect s | 32 | 203.38 | 136.64 | +48.8% | 3/3 |
| Pose s | 16 | 191.06 | 128.98 | +48.1% | 3/3 |
| Pose s | 32 | 192.12 | 129.47 | +48.4% | 3/3 |
| Seg s | 16 | 157.74 | 105.64 | +49.3% | 3/3 |
| Seg s | 32 | 157.96 | 104.95 | +50.5% | 3/3 |

上述 Python 二進位 SHA256 為
`0aaccdb95978e2df267e559aa779afa40b710f91a4e7709e845a3c612b2e21b0`。
之後加入 non-blocking streams／producer stream 0 的明確事件銜接；新二進位已重新做小樣本
正確性比對與回歸，沒有重跑這個三輪矩陣，因此上述成績保留為歷史記錄。
本機原始記錄：`runs/optimization/`。

## 四模型／1920 單輪短測

兩個獨立 Detect n 引擎、Detect s、Pose l。1920×1920 推論，輸入為預載的
1920×1080 真實 RGB 影像，batch 1、單槽、湊批 0 ms，conf=.25、IoU=.7、max_det=300。
每一路至多一筆未完成請求；每設定約 8 秒並排空後結束。
vONNX 對照為純 C++ C API v2；官方使用四個獨立 predictor／執行緒／CUDA streams。

| 每模型路數 | vONNX 合計推論/s | 官方合計推論/s | vONNX 平均 ms | 官方平均 ms |
|---|---:|---:|---:|---:|
| 1 | 17.76 | 15.72 | 約225 | 約254 |
| 4 | 17.73 | 15.70 | 約867 | 約975 |

合計為四模型完成次數相加，不能當成每張影像完成全部四任務的 FPS。
vONNX 本輪約快 13%，但這是单輪短測，不是 500 次三輪驗收。兩方沒有 OOM 或請求失敗，
單獨／同時執行的各自輸出一致；這不替代 1920 的官方輸出與完整 AP 驗證。
官方 `model.predict()` 保留內建階段計時，其 Profile 會同步整個 CUDA 裝置；
這裡比較的是原版高階入口，不是去除計時後的 PyTorch forward 上限。

本機記錄：`runs/stream_isolation/native_after/` 與 `runs/multimodel_1920_official/`。
可重跑入口見 [診斷工具](../tools/diagnostics/README.md)。

## 已確認的併發限制

純 C++ 與 Python 的四模型結果相近，排除 Python／GIL 為主要瓶頸。
CUDA trace 中，一個 n 的自身 kernel 累積約24ms，CPU 約14ms已提交最後推論 kernel，
GPU 約200ms才完成它。等待主要發生在 GPU 排程／資源競爭，GPU 使用率高並不證明已達硬體極限。
目前尚未完全拆分 Windows WDDM 排程與 GPU SM／記憶體資源競爭的比例。

non-blocking streams 與本機最高 CUDA priority 均未實質提高此組吞吐。
停用 ORT 結尾同步的試驗未見吞吐改善，並在清理時回傳 `0xC0000409`；該案例標記失敗，未採用於正式引擎。
獨立 session／stream 隔離資料與流程，仍共享 GPU，不能保證各模型維持單獨運行的速度。
本機 trace 與診斷報告：`runs/stream_isolation/`。

## 正確性與尚未完成的驗收

22 項回歸包含请求對應、容量拒絕、pause／resume 排空、兩槽不覆寫、CPU 輸入持有、
GPU default／external producer streams、像素格式、錯誤派送及 C API v1／v2。
三種 s／640 的 COCO8 小樣本、batch 上限 1／2／4／8 比對通過：前處理誤差為0，
框／關鍵點誤差低於1 pixel，Seg mask IoU為1.0；原始輸出使用 atol=1e-4、rtol=1e-3。
vONNX 自身熱路徑 cudaMalloc 計數為0，但沒有攔截第三方 ORT／cuDNN 配置。

目前不是所有模型與尺寸的完整 COCO AP 或效能證明。完整驗證與完整跑分維持独立指令，
不會自動接續短測；歷史 COCO128 報告不作為目前版本的驗收依據。

## FP16 0.4.0 短測（2026-10-18）

Windows／RTX 3070 8GB，三種 s 模型、640×640、batch 上限 8、等待 2 ms、雙槽位。
每設定共 500 次請求，前 25 次暖機排除、475 次計分，重複三輪；下表為 FPS 中位數。
雙方都回傳必要的 CPU 框／關鍵點／ROI masks；主要比較關閉詳細 timing。

| 任務 | 16 路 vONNX / 官方 FP16 | 32 路 vONNX / 官方 FP16 | 每設定勝場 |
|---|---:|---:|---:|
| Detect s | 336.02 / 151.18 | 336.53 / 151.73 | 3/3 |
| Pose s | 310.35 / 140.92 | 310.50 / 141.36 | 3/3 |
| Seg s | 252.10 / 111.60 | 252.31 / 111.86 | 3/3 |

六組皆通過速度門檻，約為官方 PyTorch FP16 湊批版本的 2.20–2.26 倍。
官方輸入事先預載為 BGR，沒有把額外的 RGB→BGR 轉換計入測量；較早的基準已歸檔。
本輪 Python 二進位 SHA256：`55fbcfeab264247356752880a13e4d7a70720cb72ed3439703f1cbcd1f4c8948`。
修改前基準另以保留的 0.3 二進位重新短測，未沿用舊數字。

初版 FP16 曾因 batch 切換重建 cuDNN 計畫而慢於官方。現在 FP16 吞吐模式預設固定
最大執行 batch、不足補零；補零不回傳結果、不計入 FPS，另記錄實際執行 batch 與補齊量。
FP32／低延遲模式維持動態 batch，亦可明確設定 batch_shape_policy。

33 項測試通過，包含三任務真實 FP16 I/O、小量同 ONNX GPU 比對、矩形尺寸、輸出完整性與 AP 門檻邏輯。
三種 s／640 的正式 COCO AP 已通過，最大降幅 0.0385 個百分點，低於 0.5 的門檻。
Detect／Seg 使用完整 5,000 張；Pose 使用全部有效關鍵點影像；各項 AP 與雜湊見 [AP 報告](ap_fp16.md)。
本機明細：`runs/fp16/REPORT.md`；用法見 [FP16 指南](fp16.md)。本版未執行長時間效能跑分。
