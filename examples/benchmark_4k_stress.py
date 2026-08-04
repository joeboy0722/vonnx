import os
import time
import threading
import numpy as np

from vonnx import vONNXEngine, EngineConfig, PixelFormat

MODEL_PATH = "yolov8n.onnx"

def run_multi_model_4k_stress_test(num_models=5, streams_per_model=10, frames_per_stream=15):
    total_streams = num_models * streams_per_model
    total_frames = total_streams * frames_per_stream

    print("==========================================================================")
    print("  vONNX Engine: 多模型 x 多相機 x 4K 極限高併發壓力測試                  ")
    print("==========================================================================")

    if not os.path.exists(MODEL_PATH):
        print(f"請先確保專案目錄下存在模型檔案: {MODEL_PATH}")
        return

    engines = []
    print("[1/3] 初始化 5 個獨立的 vONNX GPU 推論引擎...")
    for i in range(num_models):
        config = EngineConfig()
        config.model_path = MODEL_PATH
        config.max_batch_size = 16
        config.max_latency_ms = 2.0
        config.conf_threshold = 0.25
        config.iou_threshold = 0.45
        config.max_gpu_pages = 32
        config.gpu_id = 0
        
        eng = vONNXEngine(config)
        engines.append(eng)

    img_w, img_h = 3840, 2160
    frame_bytes = img_w * img_h * 3 / (1024 * 1024)
    print(f"\n[2/3] 生成 4K 測試影像 Buffer (單幀大小: {frame_bytes:.2f} MB)...")
    dummy_4k_frame = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)

    latencies = []
    lat_lock = threading.Lock()
    completed_counter = [0]
    counter_lock = threading.Lock()

    def camera_stream_worker(model_idx, stream_id):
        engine = engines[model_idx]
        global_stream_id = model_idx * streams_per_model + stream_id

        for f_idx in range(frames_per_stream):
            t0 = time.perf_counter()
            try:
                results = engine.predict_numpy(
                    stream_id=global_stream_id,
                    frame_id=f_idx,
                    array=dummy_4k_frame,
                    format=PixelFormat.BGR24
                )
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0

                with lat_lock:
                    latencies.append(elapsed_ms)

                with counter_lock:
                    completed_counter[0] += 1
            except Exception as ex:
                print(f"[Model #{model_idx+1} Stream {stream_id}] 錯誤: {ex}")

            time.sleep(0.001)

    print(f"\n[3/3] 啟動 {total_streams} 路 4K RTSP 非同步相機執行緒...")
    threads = []
    wall_start = time.perf_counter()

    for m_idx in range(num_models):
        for s_idx in range(streams_per_model):
            t = threading.Thread(target=camera_stream_worker, args=(m_idx, s_idx))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    wall_end = time.perf_counter()
    total_time_sec = wall_end - wall_start

    for eng in engines:
        eng.stop()

    if latencies:
        avg_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        throughput_fps = completed_counter[0] / total_time_sec
        processed_data_gb = (completed_counter[0] * img_w * img_h * 3) / (1024**3)
        bandwidth_gbps = processed_data_gb / total_time_sec

        print("\n==========================================================================")
        print("                        4K 極限高併發壓力測試結果                      ")
        print("==========================================================================")
        print(f" 併發模型總數 (Active Models):       {num_models} 個獨立 YOLO 模型")
        print(f" 總併發 4K 串流數 (Concurrent 4K):   {total_streams} 路 Full 4K (3840x2160)")
        print(f" 完成推論總幀數 (Completed Frames):  {completed_counter[0]} 幀")
        print(f" 處理總數據量 (Total Data Volume):   {processed_data_gb:.2f} GB")
        print(f" 總測試耗時 (Total Time Elapsed):    {total_time_sec:.3f} 秒")
        print("--------------------------------------------------------------------------")
        print(f" 系統總吞吐量 (System Throughput):   {throughput_fps:.2f} FPS")
        print(f" 實測資料頻寬 (Processing Speed):    {bandwidth_gbps:.2f} GB/sec")
        print(f" 平均單幀延遲 (Average Latency):     {avg_lat:.2f} ms")
        print(f" P95 延遲 (95th Percentile):         {p95_lat:.2f} ms")
        print("==========================================================================\n")

if __name__ == "__main__":
    run_multi_model_4k_stress_test(num_models=5, streams_per_model=10, frames_per_stream=15)
