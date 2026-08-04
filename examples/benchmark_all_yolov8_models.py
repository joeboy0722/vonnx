import os
import time
import threading
import numpy as np

from vonnx import vONNXEngine, EngineConfig, PixelFormat, load_image

MODEL_NAMES = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
IMAGE_PATH = "bus.jpg"

def export_model_if_needed(model_name):
    onnx_path = f"{model_name}.onnx"
    pt_path = f"{model_name}.pt"
    if not os.path.exists(onnx_path):
        print(f"\n[vONNX 評測] 自動匯出 {model_name}.onnx 模型...")
        from ultralytics import YOLO
        model = YOLO(pt_path)
        model.export(format="onnx", dynamic=True)
    return onnx_path

def test_single_model(model_name, img_rgb, num_streams=32, frames_per_stream=10):
    onnx_path = export_model_if_needed(model_name)
    model_file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    config = EngineConfig()
    config.model_path = onnx_path
    config.max_batch_size = 16
    config.max_latency_ms = 2.0
    config.conf_threshold = 0.25
    config.iou_threshold = 0.45
    config.max_gpu_pages = 64
    config.gpu_id = 0

    engine = vONNXEngine(config)

    single_res = engine.predict_numpy(stream_id=0, frame_id=0, array=img_rgb, format=PixelFormat.RGB24)
    boxes = single_res[0].boxes
    detected_count = len(boxes)
    top_score = boxes[0].score if detected_count > 0 else 0.0

    latencies = []
    lat_lock = threading.Lock()
    completed_counter = [0]
    counter_lock = threading.Lock()

    def stream_worker(stream_id):
        for f_idx in range(frames_per_stream):
            t0 = time.perf_counter()
            try:
                results = engine.predict_numpy(
                    stream_id=stream_id,
                    frame_id=f_idx,
                    array=img_rgb,
                    format=PixelFormat.RGB24
                )
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0

                with lat_lock:
                    latencies.append(elapsed_ms)

                with counter_lock:
                    completed_counter[0] += 1
            except Exception as ex:
                print(f"[{model_name} Stream {stream_id}] 錯誤: {ex}")

            time.sleep(0.001)

    threads = []
    wall_start = time.perf_counter()

    for s in range(num_streams):
        t = threading.Thread(target=stream_worker, args=(s,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    wall_end = time.perf_counter()
    total_time_sec = wall_end - wall_start

    engine.stop()

    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    fps = completed_counter[0] / total_time_sec if total_time_sec > 0 else 0.0

    return {
        "model": model_name,
        "size_mb": model_file_size_mb,
        "detected_count": detected_count,
        "top_score": top_score,
        "fps": fps,
        "avg_lat": avg_lat,
        "p95_lat": p95_lat,
    }

def main():
    print("==========================================================================")
    print("  vONNX Engine: YOLOv8 全系列模型 (n, s, m, l, x) 高併發綜合評測          ")
    print("==========================================================================")

    if not os.path.exists(IMAGE_PATH):
        print(f"請先確保專案目錄下存在測試圖片: {IMAGE_PATH}")
        return

    img_rgb = load_image(IMAGE_PATH)
    print(f"載入測試圖片: {IMAGE_PATH} ({img_rgb.shape[1]}x{img_rgb.shape[0]})")

    results_table = []
    for model_name in MODEL_NAMES:
        print(f"\n[開始評測] 模型: {model_name} ...")
        res = test_single_model(model_name, img_rgb, num_streams=32, frames_per_stream=10)
        results_table.append(res)

    print("\n==========================================================================================")
    print("                         YOLOv8 全系列模型高併發綜合評測總表                             ")
    print("==========================================================================================")
    print(f"{'模型名稱':<10} | {'模型大小(MB)':<12} | {'偵測框數':<8} | {'最高信心度':<10} | {'系統吞吐量(FPS)':<16} | {'平均延遲(ms)':<12} | {'P95延遲(ms)':<12}")
    print("------------------------------------------------------------------------------------------")
    for r in results_table:
        print(f"{r['model']:<10} | {r['size_mb']:<12.1f} | {r['detected_count']:<8} | {r['top_score']:<10.4f} | {r['fps']:<16.2f} | {r['avg_lat']:<12.2f} | {r['p95_lat']:<12.2f}")
    print("==========================================================================================\n")

if __name__ == "__main__":
    main()
