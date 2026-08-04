import os
import numpy as np
from vonnx import vONNXEngine, EngineConfig, PixelFormat

def main():
    print("==========================================================")
    print("  vONNX Engine: Python Quick Start 基礎推論範例          ")
    print("==========================================================")

    # 1. 建立引擎組態
    config = EngineConfig()
    config.model_path = "yolov8n.onnx"
    config.max_batch_size = 16
    config.max_latency_ms = 2.0
    config.conf_threshold = 0.25
    config.iou_threshold = 0.45
    config.gpu_id = 0

    if not os.path.exists(config.model_path):
        print(f"請先確保專案目錄下存在模型檔案: {config.model_path}")
        return

    # 2. 實體化 vONNX 推論引擎
    print("初始化 vONNX Engine...")
    engine = vONNXEngine(config)

    # 3. 模擬輸入影像 (1920x1080 3-Channel BGR24 陣列)
    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # 4. 提交推論請求
    print("發送推論請求...")
    results = engine.predict_numpy(stream_id=1, frame_id=101, array=dummy_frame, format=PixelFormat.BGR24)

    # 5. 檢視推論結果
    print(f"推論完成! 傳回串流 ID: {results[0].stream_id}, 幀 ID: {results[0].frame_id}")
    print(f"偵測目標邊界框數量: {len(results[0].boxes)}")
    for i, box in enumerate(results[0].boxes[:5]):
        print(f"  [{i+1}] Class ID: {box.class_id}, Score: {box.score:.4f}, BBox: [{box.x1:.1f}, {box.y1:.1f}, {box.x2:.1f}, {box.y2:.1f}]")

    # 6. 停止引擎
    engine.stop()
    print("vONNX Engine 已安全停止。")

if __name__ == "__main__":
    main()
