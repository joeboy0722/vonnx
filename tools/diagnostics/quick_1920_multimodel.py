"""Bounded four-engine 1920 smoke test; no full suite and no silent fallback."""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vonnx_tools.common import manifest, images, load_rgb, write_json, environment, build_fingerprint

OUTPUT = ROOT / "runs/multimodel_1920"
MODELS = (
    ("detect_n_1", "yolov8n"),
    ("detect_n_2", "yolov8n"),
    ("detect_s", "yolov8s"),
    ("pose_l", "yolov8l-pose"),
)


def worker():
    import cv2
    import numpy as np
    import pynvml
    from vonnx import EngineConfig, vONNXEngine, PixelFormat
    from vonnx_tools.benchmark import sample_stats
    from vonnx_tools.optimize import stats
    from vonnx_tools.validate import box_array

    frames = {}
    for task in ("detect", "pose"):
        frames[task] = [np.ascontiguousarray(cv2.resize(a, (1920, 1080))) for a in load_rgb(images(task))]
    pynvml.nvmlInit()
    device = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory = lambda: pynvml.nvmlDeviceGetMemoryInfo(device).used
    result = dict(
        status="initializing",
        models=[],
        scenarios=[],
        input_shape=[1080, 1920, 3],
        imgsz=[1920, 1920],
        batch=1,
        depth=1,
        concurrency_per_engine=32,
        wait_ms=0,
        collect_timings=False,
        backend=args.backend,
        official_builtin_profiling=args.backend == "official",
        environment=environment(),
        binary_sha256=build_fingerprint(),
        initial_device_bytes=memory(),
    )
    peak = [memory()]
    sampling = threading.Event()

    def sample():
        while not sampling.wait(0.02):
            peak[0] = max(peak[0], memory())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    engines = []

    def save():
        result["gpu_peak_bytes"] = peak[0]
        write_json(OUTPUT / "result.json", result)

    def submit(index, lane, frame):
        task = "pose" if index == 3 else "detect"
        return engines[index].submit_numpy(
            lane, frame, frames[task][frame % len(frames[task])], PixelFormat.RGB24
        )

    try:
        for label, model in MODELS:
            print("Initialize " + label, flush=True)
            result["initializing_model"] = label
            save()
            cfg = EngineConfig()
            cfg.model_path = str(ROOT / manifest(model)["onnx"])
            cfg.imgsz = 1920
            cfg.max_input_width, cfg.max_input_height = 1920, 1080
            cfg.max_batch_size, cfg.pipeline_depth = 1, 1
            cfg.batch_mode, cfg.max_latency_ms = "latency", 0
            cfg.concurrency = 32
            cfg.collect_timings = False
            begin = time.perf_counter()
            if args.backend == "official":
                from official_1920 import Official1920

                engines.append(Official1920(model))
            else:
                engines.append(vONNXEngine(cfg))
            result["models"].append(
                dict(
                    label=label,
                    model=model,
                    hash=manifest(model)["weight_sha256" if args.backend == "official" else "onnx_sha256"],
                    initialization_seconds=time.perf_counter() - begin,
                    resources=stats(engines[-1]),
                    device_bytes=memory(),
                )
            )
            save()
        for i in range(4):
            for n in range(5):
                submit(i, 0, n).result(timeout=30)
        # Same resident sessions: reference with other models idle, then all submitted before waiting.
        alone = [submit(i, 0, 0).result(timeout=30)[0] for i in range(4)]
        pending = [submit(i, 0, 0) for i in range(4)]
        together = [f.result(timeout=30)[0] for f in pending]
        for a, b in zip(alone, together):
            assert (a.stream_id, a.frame_id) == (b.stream_id, b.frame_id)
            np.testing.assert_allclose(box_array(a), box_array(b), atol=1e-4, rtol=1e-3)
            for ka, kb in zip(a.keypoints, b.keypoints):
                np.testing.assert_allclose(
                    [[p.x, p.y, p.confidence] for p in ka],
                    [[p.x, p.y, p.confidence] for p in kb],
                    atol=1e-4,
                    rtol=1e-3,
                )
        result["concurrent_output_matches_idle"] = True

        def measure(indices, lanes, seconds, name):
            logs = []
            stop = threading.Event()
            barrier = threading.Barrier(len(indices) * lanes + 1)
            errors = []
            lock = threading.Lock()

            def lane_work(i, lane):
                local = []
                frame = lane
                barrier.wait()
                while not stop.is_set():
                    begin = time.perf_counter()
                    try:
                        f = submit(i, lane, frame)
                        submitted = time.perf_counter()
                        r = f.result(timeout=30)[0]
                        complete = time.perf_counter()
                        assert (r.stream_id, r.frame_id) == (lane, frame)
                        assert r.batch_size == 1
                        local.append(
                            dict(
                                model=MODELS[i][0],
                                lane=lane,
                                frame=frame,
                                latency_ms=(complete - begin) * 1000,
                                submit_ms=(submitted - begin) * 1000,
                                boxes=len(r.boxes),
                                keypoints=len(r.keypoints),
                            )
                        )
                        frame += lanes
                    except Exception as e:
                        with lock:
                            errors.append(dict(model=MODELS[i][0], error=repr(e)))
                        stop.set()
                        break
                with lock:
                    logs.extend(local)

            threads = [
                threading.Thread(target=lane_work, args=(i, lane)) for i in indices for lane in range(lanes)
            ]
            for thread in threads:
                thread.start()
            begin = time.perf_counter()
            barrier.wait()
            stop.wait(seconds)
            stop.set()
            for thread in threads:
                thread.join()
            elapsed = time.perf_counter() - begin
            summaries = []
            for i in indices:
                rows = [r for r in logs if r["model"] == MODELS[i][0]]
                assert len({(r["lane"], r["frame"]) for r in rows}) == len(rows)
                summaries.append(
                    dict(
                        model=MODELS[i][0],
                        success=len(rows),
                        fps=len(rows) / elapsed,
                        latency=sample_stats([r["latency_ms"] for r in rows]),
                        submit=sample_stats([r["submit_ms"] for r in rows]),
                    )
                )
            scenario = dict(
                name=name,
                lanes_per_model=lanes,
                requested_seconds=seconds,
                seconds_including_drain=elapsed,
                status="failed" if errors else "passed",
                errors=errors,
                per_model=summaries,
                total_fps=len(logs) / elapsed,
            )
            with (OUTPUT / (name + ".jsonl")).open("w", encoding="utf-8") as file:
                for row in logs:
                    file.write(json.dumps(row) + "\n")
            result["scenarios"].append(scenario)
            save()
            print(json.dumps(scenario), flush=True)
            if errors:
                raise RuntimeError("Scenario failed; stopping")

        for i in range(4):
            measure([i], 1, 3, "idle_" + MODELS[i][0])
        for lanes in (1, 4):
            if args.profile and lanes == 1:
                import ctypes

                runtime = ctypes.WinDLL(
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64/cudart64_13.dll"
                )
                assert runtime.cudaProfilerStart() == 0
            measure(range(4), lanes, 8, "concurrent_" + str(lanes))
            if args.profile and lanes == 1:
                assert runtime.cudaProfilerStop() == 0
        result["resources_after"] = [
            dict(model=label, resources=stats(e)) for (label, _), e in zip(MODELS, engines)
        ]
        result["status"] = "passed"
        save()
    except Exception as e:
        result.update(status="failed", error=repr(e))
        save()
        raise
    finally:
        for e in reversed(engines):
            e.close()
        sampling.set()
        sampler.join()
        result["device_bytes_after_close"] = memory()
        save()
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--export-native-inputs",
        action="store_true",
        help="Export real RGB frames for the C++ client; no inference",
    )
    parser.add_argument("--backend", choices=("native", "official"), default="native")
    parser.add_argument("--profile", action="store_true", help="CUDA profiler range around concurrent_1 only")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if args.export_native_inputs:
        import cv2

        inputs = ROOT / "runs/stream_isolation/inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        for task in ("detect", "pose"):
            for index, image in enumerate(load_rgb(images(task))):
                cv2.resize(image, (1920, 1080)).tofile(inputs / f"{task}_{index}.bin")
        print(inputs)
        raise SystemExit(0)
    OUTPUT = Path(args.output)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker()
    else:
        with (OUTPUT / "worker.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    [sys.executable, __file__, "--worker", "--output", str(OUTPUT), "--backend", args.backend]
                    + (["--profile"] if args.profile else []),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    timeout=240,
                )
                if process.returncode:
                    if (OUTPUT / "result.json").exists():
                        r = json.loads((OUTPUT / "result.json").read_text())
                        r.update(status="failed", exitcode=process.returncode)
                        write_json(OUTPUT / "result.json", r)
                    raise SystemExit(process.returncode)
            except subprocess.TimeoutExpired:
                r = (
                    json.loads((OUTPUT / "result.json").read_text())
                    if (OUTPUT / "result.json").exists()
                    else {}
                )
                r.update(status="timeout", error="240 second GPU test wall-time limit")
                write_json(OUTPUT / "result.json", r)
                raise
        print((OUTPUT / "result.json").read_text())
