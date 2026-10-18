"""Detect s/640 INT8 feasibility experiment; no production API changes or long tests.

100 held-out COCO images for evaluation, 100 different images for calibration.
Run from the repository with its existing GPU Python environment.
"""
import argparse
import collections
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/ultralytics"))
from vonnx_tools.common import manifest, sha256, write_json, load_rgb, environment, build_fingerprint


def paths():
    import numpy as np
    photos = sorted((ROOT / "datasets/coco/val2017").glob("*.jpg"))
    if len(photos) != 5000:
        raise ValueError("Requires existing COCO val2017; this script does not download data")
    selected = np.random.default_rng(20261018).permutation(photos)[:200]
    return list(selected[:100]), list(selected[100:])


def prepare(out):
    import numpy as np
    import onnx
    import vonnx  # Register existing runtime DLL directories.
    from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType, CalibrationMethod
    from ultralytics.data.augment import LetterBox
    info = manifest("yolov8s")
    source = ROOT / info["onnx"]
    calibration, evaluation = paths()
    graph = onnx.load(source)
    class Reader(CalibrationDataReader):
        def __init__(self):
            self.iterator = iter(calibration)
        def get_next(self):
            path = next(self.iterator, None)
            if path is None:
                return None
            rgb = load_rgb([path])[0]
            tensor = LetterBox((640, 640), auto=False)(image=rgb).transpose(2, 0, 1)
            return {graph.graph.input[0].name: np.ascontiguousarray(tensor[None], np.float32) / 255.}
    # Keep DFL and final class/box output layers in floating point for this first PTQ trial.
    import re
    protected = [n.name for n in graph.graph.node if "/dfl/" in n.name or
                 re.search(r"/cv[23]\.\d+/cv[23]\.\d+\.2/", n.name)]
    target = out / "yolov8s-qdq.onnx"
    # ModelProto avoids ONNX's path-based shape inference mishandling Unicode paths.
    quantize_static(graph, target, Reader(), quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv"], per_channel=True,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        nodes_to_exclude=protected, calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CUDAExecutionProvider"],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    quantized = onnx.load(target)
    onnx.checker.check_model(quantized)
    write_json(out / "preparation.json", dict(source_sha256=sha256(source), quantized_sha256=sha256(target),
        calibration_ids=[int(p.stem) for p in calibration], evaluation_ids=[int(p.stem) for p in evaluation],
        ops=collections.Counter(n.op_type for n in quantized.graph.node), protected_nodes=protected,
        settings=dict(format="QDQ", activation="S8", weight="S8 per-channel", calibration="MinMax"),
        environment=environment(), binary_sha256=build_fingerprint()))


def profile(out):
    import ctypes
    import numpy as np
    import vonnx
    import onnxruntime as ort
    runtime = {}
    for dll in ("nvinfer_10.dll", "nvinfer_11.dll", "nvonnxparser_10.dll"):
        try:
            ctypes.WinDLL(dll)
            runtime[dll] = "loadable"
        except OSError as error:
            runtime[dll] = str(error)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(out / "cuda_qdq")
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    options.optimized_model_filepath = str(out / "cuda_optimized.onnx")
    session = ort.InferenceSession(str(out / "yolov8s-qdq.onnx"), options,
        providers=[("CUDAExecutionProvider", {"use_tf32": False})])
    output = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, 640, 640), np.float32)})
    events = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
    counts = collections.Counter()
    for event in events:
        args = event.get("args", {})
        if event.get("cat") == "Node" and args.get("provider"):
            counts[(args["provider"], args.get("op_name", "unknown"))] += 1
    # Record actual node placement, rather than treating advertised providers as proof.
    nodes = [dict(provider=p, op=op, count=n) for (p, op), n in sorted(counts.items())]
    integer_convolution = any("CUDA" in p and op in ("QLinearConv", "ConvInteger") for p, op in counts)
    write_json(out / "placement.json", dict(advertised_providers=ort.get_available_providers(),
        active_providers=session.get_providers(), runtime_dlls=runtime, node_placement=nodes,
        true_gpu_integer_convolution_observed=integer_convolution,
        output_finite=all(np.isfinite(v).all() for v in output)))


def make_engine(out, kind, lanes, rgb):
    from vonnx import EngineConfig, vONNXEngine
    config = EngineConfig()
    config.model_path = str(ROOT / manifest("yolov8s")["onnx"] if kind == "fp16" else out / "yolov8s-qdq.onnx")
    config.precision = "fp16" if kind == "fp16" else "auto"
    config.model_cache_dir = str(ROOT / "runs/fp16/cache")
    config.imgsz = 640
    config.concurrency = 32
    config.max_batch_size = 1 if lanes == 1 else 8
    config.batch_mode = "latency" if lanes == 1 else "throughput"
    config.batch_shape_policy = "fixed"
    config.pipeline_depth = 1 if lanes == 1 else 2
    config.max_latency_ms = 0 if lanes == 1 else 2
    config.collect_timings = False
    config.max_input_width = max(a.shape[1] for a in rgb)
    config.max_input_height = max(a.shape[0] for a in rgb)
    config.conf_threshold = .25
    config.iou_threshold = .7
    config.max_det = 300
    return vONNXEngine(config)


def speed(out, kind, lanes):
    import numpy as np
    from vonnx import PixelFormat
    rgb = load_rgb(paths()[1])
    engine = make_engine(out, kind, lanes, rgb)
    results = []
    try:
        for repeat in range(3):
            for i in range(25):
                engine.submit_numpy(0, i, rgb[i], PixelFormat.RGB24).result(timeout=30)
            barrier = threading.Barrier(lanes + 1)
            lock = threading.Lock()
            rows = []
            def work(lane):
                barrier.wait()
                for frame in range(lane, 100, lanes):
                    start = time.perf_counter()
                    r = engine.submit_numpy(lane, frame, rgb[frame], PixelFormat.RGB24).result(timeout=30)[0]
                    elapsed = (time.perf_counter() - start) * 1000
                    if r.frame_id != frame or r.stream_id != lane:
                        raise AssertionError("Request identity mismatch")
                    with lock:
                        rows.append(dict(frame=frame, latency_ms=elapsed, batch=r.batch_size,
                                         physical_batch=r.physical_batch_size))
            with concurrent.futures.ThreadPoolExecutor(max_workers=lanes) as pool:
                workers = [pool.submit(work, lane) for lane in range(lanes)]
                start = time.perf_counter()
                barrier.wait()
                for worker in workers:
                    worker.result(timeout=90)
                duration = time.perf_counter() - start
            assert len(rows) == 100 and len({r['frame'] for r in rows}) == 100
            latencies = [r['latency_ms'] for r in rows]
            results.append(dict(repeat=repeat, fps=100/duration, mean_ms=float(np.mean(latencies)),
                p95_ms=float(np.percentile(latencies,95)), p99_ms=float(np.percentile(latencies,99)), rows=rows))
    finally:
        engine.close()
    write_json(out / f"speed_{kind}_{lanes}.json", dict(kind=kind, lanes=lanes,
        measured_images=100, warmup=25, repeats=results,
        median_fps=float(np.median([r['fps'] for r in results])), binary_sha256=build_fingerprint()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "prepare", "profile", "speed"], default="all")
    parser.add_argument("--kind", choices=["fp16", "qdq"], default="fp16")
    parser.add_argument("--lanes", type=int, choices=[1,16,32], default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/int8_detect")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "all":
        jobs = [["prepare"], ["profile"]] + [["speed", "--kind", k, "--lanes", str(n)]
            for k in ("fp16", "qdq") for n in (1,16,32)]
        for job in jobs:
            label = "_".join(job).replace("--", "")
            print("Starting", label, flush=True)
            try:
                with (out / (label + ".log")).open("w", encoding="utf-8") as log:
                    run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--phase", *job,
                        "--output", str(out)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                if run.returncode:
                    raise RuntimeError(f"Worker exit {run.returncode}; inspect {label}.log")
            except Exception as error:
                write_json(out / (label + ".failure.json"), dict(error=str(error)))
                print(label, "FAILED", error, flush=True)
                break
            print("Completed", label, flush=True)
    elif args.phase == "prepare":
        prepare(out)
    elif args.phase == "profile":
        profile(out)
    else:
        speed(out, args.kind, args.lanes)


if __name__ == "__main__":
    main()
