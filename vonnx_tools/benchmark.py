import collections
import concurrent.futures
import json
import queue
import threading
import time
from pathlib import Path
import numpy as np
from .common import (
    ROOT,
    RESOLUTIONS,
    STREAMS,
    names,
    manifest,
    engine,
    images,
    load_rgb,
    write_json,
    environment,
    build_fingerprint,
)


def request_quota(limit, lanes, lane):
    return limit // lanes + int(lane < limit % lanes) if limit else None


class OfficialEngine:
    """One predictor/dispatch thread per model, explicit batching rather than a shared predictor lock."""

    def __init__(self, model, size, batch):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = YOLO(str(ROOT / manifest(model)["weight"]))
        self.model.to("cuda")
        self.model.model.float()
        self.size = size
        self.batch = batch
        self.queue = queue.Queue(1024)
        self.thread = threading.Thread(target=self.work)
        self.thread.start()

    def submit(self, stream, frame, image):
        future = concurrent.futures.Future()
        self.queue.put_nowait((future, image, time.perf_counter(), stream, frame))
        return future

    def work(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            items = [first]
            deadline = first[2] + 0.002
            while len(items) < self.batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    items.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                dispatch = time.perf_counter()
                result = self.model.predict(
                    [np.ascontiguousarray(i[1][:, :, ::-1]) for i in items],
                    imgsz=self.size,
                    rect=False,
                    conf=0.25,
                    iou=0.7,
                    max_det=300,
                    quantize=None,
                    retina_masks=True,
                    verbose=False,
                )
                for request, r in zip(items, result):
                    # Match native ROI CPU-output semantics; no full mask D2H when returning ROI masks.
                    boxes = r.boxes.data.cpu().numpy()
                    if r.masks is not None:
                        for box, mask in zip(boxes, r.masks.data):
                            x, y, right, bottom = np.ceil(box[:4]).astype(int)
                            mask[y:bottom, x:right].cpu().numpy()
                    if r.keypoints is not None:
                        r.keypoints.data.cpu().numpy()
                    self.torch.cuda.synchronize()
                    request[0].set_result(
                        {
                            "batch_size": len(items),
                            "queue_ms": (dispatch - request[2]) * 1000,
                            "completed": time.perf_counter(),
                        }
                    )
            except Exception as error:
                for request in items:
                    request[0].set_exception(error)

    def stop(self):
        self.queue.put(None)
        self.thread.join()


class NativeEngine:
    def __init__(self, model, size, width, height):
        self.engine = engine(model, size, max_width=width, max_height=height)

    def submit(self, stream, frame, image):
        from vonnx import PixelFormat

        return self.engine.submit_numpy(stream, frame, image, PixelFormat.RGB24)

    def stop(self):
        self.engine.stop()


def cases(sizes, streams, repeats, seconds, warmup):
    result = []
    singles = [[m] for m in names()]
    mixed = [names(scales=[s]) for s in ("n", "s", "m", "l", "x")]
    same = []
    for task in ("detect", "pose", "segment"):
        for scales in [("n", "s"), ("m", "l"), ("x", "n"), ("n", "s", "m", "l", "x")]:
            same.append(names(tasks=[task], scales=scales))
    groups = (
        [("single", g) for g in singles]
        + [("mixed", g) for g in mixed]
        + [("same-task", g) for g in same]
        + [("fanout", g) for g in mixed]
    )
    for kind, models in groups:
        for size in sizes:
            for lanes in streams:
                for backend in (
                    ["native", "pytorch-single", "pytorch-batched"] if kind == "single" else ["native"]
                ):
                    for repeat in range(repeats):
                        identifier = (
                            f"{kind}__" + "+".join(models) + f"__{size}__s{lanes}__{backend}__r{repeat+1}"
                        )
                        result.append(
                            {
                                "id": identifier,
                                "kind": kind,
                                "models": models,
                                "imgsz": size,
                                "streams_per_model": lanes,
                                "backend": backend,
                                "repeat": repeat + 1,
                                "seconds": seconds,
                                "warmup_batches": warmup,
                                "warmup_policy": "cycle_possible_batch_sizes_v1",
                                "max_batch_size": 1 if backend == "pytorch-single" else 16,
                                "max_latency_ms": 2,
                                "mask_mode": "roi",
                                "conf": 0.25,
                                "iou": 0.7,
                                "max_det": 300,
                            }
                        )
    return result


def sample_stats(values):
    if not values:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    return dict(
        zip(
            ("mean_ms", "p50_ms", "p95_ms", "p99_ms"),
            [float(np.mean(values)), *[float(x) for x in np.percentile(values, [50, 95, 99])]],
        )
    )


def run_case(case, output):
    import pynvml

    pynvml.nvmlInit()
    device = pynvml.nvmlDeviceGetHandleByIndex(0)
    baseline = pynvml.nvmlDeviceGetMemoryInfo(device).used
    peak = [baseline]
    sampling = threading.Event()

    def sample():
        while not sampling.wait(0.05):
            peak[0] = max(peak[0], pynvml.nvmlDeviceGetMemoryInfo(device).used)

    sampler = threading.Thread(target=sample)
    sampler.start()
    engines = []
    try:
        paths = images("detect")
        rgb = load_rgb(paths)
        max_width = max(a.shape[1] for a in rgb)
        max_height = max(a.shape[0] for a in rgb)
        for model in case["models"]:
            eng = (
                NativeEngine(model, case["imgsz"], max_width, max_height)
                if case["backend"] == "native"
                else OfficialEngine(model, case["imgsz"], case["max_batch_size"])
            )
            engines.append(eng)
            warmup_limit = min(case["streams_per_model"], case["max_batch_size"])
            for warm in range(case["warmup_batches"]):
                warmup_batch = 1 + warm % warmup_limit
                futures = [eng.submit(lane, warm, rgb[lane % len(rgb)]) for lane in range(warmup_batch)]
                for f in futures:
                    f.result(timeout=120)
        logs = []
        lock = threading.Lock()
        errors = []
        batch_hist = [collections.Counter() for _ in engines]
        nthreads = case["streams_per_model"] * (1 if case["kind"] == "fanout" else len(engines))
        barrier = threading.Barrier(nthreads + 1)
        timing = {}

        def collect(model_index, lane, frame, t0, f):
            try:
                result = f.result(timeout=120)
                end = time.perf_counter()
                if case["backend"] == "native":
                    r = result[0]
                    if r.stream_id != lane or r.frame_id != frame:
                        raise AssertionError("Request identity mismatch")
                    fields = {
                        name: getattr(r, name)
                        for name in (
                            "batch_size",
                            "queue_ms",
                            "h2d_ms",
                            "preprocess_ms",
                            "inference_ms",
                            "postprocess_ms",
                            "d2h_ms",
                        )
                    }
                else:
                    fields = result
                    end = result["completed"]
                row = {
                    "model": case["models"][model_index],
                    "stream_id": lane,
                    "frame_id": frame,
                    "latency_ms": (end - t0) * 1000,
                    **fields,
                }
                with lock:
                    logs.append(row)
                    batch_hist[model_index][fields["batch_size"]] += 1
            except Exception as error:
                with lock:
                    errors.append(
                        {
                            "model": case["models"][model_index],
                            "stream_id": lane,
                            "frame_id": frame,
                            "error": str(error),
                        }
                    )

        def lane(model_index, lane_index):
            barrier.wait()
            frame = 0
            quota = request_quota(case.get("max_requests", 0), case["streams_per_model"], lane_index)
            while time.perf_counter() < timing["deadline"] and (quota is None or frame < quota):
                t0 = time.perf_counter()
                try:
                    f = engines[model_index].submit(lane_index, frame, rgb[(lane_index + frame) % len(rgb)])
                except Exception as error:
                    with lock:
                        errors.append({"model": case["models"][model_index], "error": str(error)})
                    break
                collect(model_index, lane_index, frame, t0, f)
                frame += 1

        def fanout(lane_index):
            barrier.wait()
            frame = 0
            quota = request_quota(case.get("max_requests", 0), case["streams_per_model"], lane_index)
            while time.perf_counter() < timing["deadline"] and (quota is None or frame < quota):
                t0 = time.perf_counter()
                image = rgb[(lane_index + frame) % len(rgb)]
                futures = []
                try:
                    for eng in engines:
                        futures.append(eng.submit(lane_index, frame, image))
                except Exception as error:
                    with lock:
                        errors.append({"error": str(error)})
                    for i, f in enumerate(futures):
                        collect(i, lane_index, frame, t0, f)
                    break
                threads = [
                    threading.Thread(target=collect, args=(i, lane_index, frame, t0, f))
                    for i, f in enumerate(futures)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                with lock:
                    logs.append(
                        {
                            "model": "ALL",
                            "stream_id": lane_index,
                            "frame_id": frame,
                            "latency_ms": (time.perf_counter() - t0) * 1000,
                        }
                    )
                frame += 1

        threads = []
        for i in range(1 if case["kind"] == "fanout" else len(engines)):
            for stream in range(case["streams_per_model"]):
                threads.append(
                    threading.Thread(
                        target=fanout if case["kind"] == "fanout" else lane,
                        args=(stream,) if case["kind"] == "fanout" else (i, stream),
                    )
                )
        for thread in threads:
            thread.start()
        timing["start"] = time.perf_counter()
        timing["deadline"] = timing["start"] + case["seconds"]
        barrier.wait()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - timing["start"]
        summaries = []
        for index, model in enumerate(case["models"]):
            rows = [r for r in logs if r["model"] == model]
            latencies = [r["latency_ms"] for r in rows]
            summaries.append(
                {
                    "model": model,
                    "successful_requests": len(rows),
                    "fps": len(rows) / elapsed,
                    "batch_histogram": {b: n / b for b, n in batch_hist[index].items()},
                    "batch_histogram_requests": dict(batch_hist[index]),
                    **sample_stats(latencies),
                }
            )
        path = Path(output) / (case["id"] + ".requests.jsonl")
        with path.open("w", encoding="utf-8") as f:
            for row in logs:
                f.write(json.dumps({"status": "passed", **row}, ensure_ascii=False) + "\n")
            for row in errors:
                f.write(json.dumps({"status": "failed", **row}, ensure_ascii=False) + "\n")
        return {
            "case": case,
            "status": "passed" if not errors else "failed",
            "elapsed_seconds": elapsed,
            "total_inferences": sum(s["successful_requests"] for s in summaries),
            "total_fps": sum(s["fps"] for s in summaries),
            "per_model": summaries,
            "all_tasks_latency": sample_stats([r["latency_ms"] for r in logs if r["model"] == "ALL"]),
            "errors": errors,
            "failed_requests": len(errors),
            "rejected_requests": sum(
                "queue" in e["error"].lower() and "full" in e["error"].lower() for e in errors
            ),
            "gpu_peak_mib": peak[0] / 1024**2,
            "gpu_baseline_mib": baseline / 1024**2,
            "memory_sampling_ms": 50,
            "models": [manifest(m) for m in case["models"]],
            "environment": environment(),
        }
    finally:
        for eng in engines:
            eng.stop()
        sampling.set()
        sampler.join()
        pynvml.nvmlShutdown()


def benchmark(args):
    from .isolated import isolated

    if args.worker:
        case = json.loads(Path(args.worker).read_text(encoding="utf-8"))
        row = run_case(case, args.output)
        write_json(Path(args.output) / (case["id"] + ".json"), row)
        return
    if (
        args.seconds <= 0
        or not np.isfinite(args.seconds)
        or args.warmup < 1
        or args.repeats < 1
        or args.max_requests < 0
        or any(s <= 0 or s % 32 for s in args.sizes)
        or any(s <= 0 for s in args.streams)
    ):
        raise ValueError("Invalid benchmark parameters")
    matrix = cases(args.sizes, args.streams, args.repeats, args.seconds, args.warmup)
    for case in matrix:
        case["max_requests"] = args.max_requests
        case["measurement_mode"] = "request-limited" if args.max_requests else "duration"
    if args.case:
        matrix = [c for c in matrix if c["id"] == args.case]
    if not matrix:
        raise ValueError("No matching benchmark case")
    if args.list:
        for c in matrix:
            print(c["id"])
        print(
            f'{len(matrix)} cases; measurement time {"upper bound" if args.max_requests else "total"}: {sum(c["seconds"] for c in matrix)/3600:.2f} hours (excludes initialization/warmup)'
        )
        return
    for case in matrix:
        case["unvalidated"] = args.allow_unvalidated
        case["model_hashes"] = {m: manifest(m)["onnx_sha256"] for m in case["models"]}
        case["environment"] = environment()
        case["build_fingerprint"] = build_fingerprint()
        case["validation_scope"] = {}
        if not args.allow_unvalidated:
            for model in case["models"]:
                path = Path(args.validation) / (model + f'-{case["imgsz"]}x{case["imgsz"]}.json')
                checked = json.loads(path.read_text(encoding="utf-8"))
                if (
                    checked.get("status") != "passed"
                    or not checked["case"]["full"]
                    or (checked["case"].get("limit") and not args.allow_sampled_validation)
                    or checked["model"]["onnx_sha256"] != manifest(model)["onnx_sha256"]
                    or checked["case"].get("build_fingerprint") != build_fingerprint()
                ):
                    raise RuntimeError(f'Full validation gate missing/stale: {model} {case["imgsz"]}')
                case["validation_scope"][model] = {
                    "sampled": bool(checked["case"].get("limit")),
                    "images": checked["detail"]["full"]["images"],
                }
    write_json(Path(args.output) / "matrix.json", matrix)
    for case in matrix:
        isolated("benchmark", case, args.output, args.timeout, resume=args.resume)
    from .report import report

    report(args.output)
