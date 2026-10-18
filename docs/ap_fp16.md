# FP16 AP 驗證（2026-10-18）

三種 YOLOv8 s、640×640，相同權重對照官方 PyTorch FP32。
Detect／Seg 使用完整 COCO val2017 5,000 張；Pose 使用有有效關鍵點標註的全部影像。
此範圍全部通過 AP50–95 最多下降 0.5 個百分點的門檻；其他模型與尺寸不包含在此結論。

| 模型 | 影像數 | 指標 | vONNX FP16 AP (%) | 官方 FP32 AP (%) | 下降（百分點） |
|---|---:|---|---:|---:|---:|
| yolov8s | 5000 | bbox | 44.2685 | 44.3024 | 0.0339 |
| yolov8s-pose | 2346 | bbox | 56.0274 | 56.0551 | 0.0277 |
| yolov8s-pose | 2346 | keypoints | 59.9493 | 59.9493 | 0.0000 |
| yolov8s-seg | 5000 | bbox | 44.0125 | 44.0510 | 0.0384 |
| yolov8s-seg | 5000 | segm | 35.6463 | 35.6390 | 0.0000 |

設定：FP32 前後處理、confidence=0.001、IoU=0.7、max_det=300、rect=False、batch=1。
Seg 使用原圖 ROI masks，COCO 評估前重建完整原圖 mask；官方使用 retina_masks=True。
COCOeval 使用標準 maxDets（bbox／segm=100、keypoints=20）；兩方設定一致。

二進位 SHA256：`55fbcfeab264247356752880a13e4d7a70720cb72ed3439703f1cbcd1f4c8948`。
轉換器 SHA256：`efd4b46aa06944e86530189b66cbbcbdeb58914b8edbb5fca05893db512c8d8e`。

機器／依賴版本、權重與來源模型雜湊見 [JSON 明細](ap_fp16.json)。
原始逐影像預測、完整評估 log 與 image IDs 保留於本機 `runs/fp16/ap_full/`。
此次未執行長時間效能跑分。
