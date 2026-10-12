"""Bounded optimization experiments. Never chains into full validation/benchmarks."""

import collections
import concurrent.futures
import csv
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
import numpy as np
from .common import ROOT, manifest, images, load_rgb, write_json, environment, build_fingerprint, sha256
from .benchmark import sample_stats

MODELS = ("yolov8s", "yolov8s-pose", "yolov8s-seg")
STAGES = (
    "queue_ms",
    "batch_wait_ms",
    "dispatch_ms",
    "h2d_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "d2h_ms",
)


class Official:
    """One resident predictor, one dispatcher; native timer and identical batch deadline."""

    def __init__(self, model, batch, wait, rgb):
        import torch
        from ultralytics import YOLO
        from vonnx import DeadlineWaiter

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.model = YOLO(str(ROOT / manifest(model)["weight"]))
        self.model.to("cuda")
        self.model.model.float()
        self.batch, self.wait = batch, wait / 1000
        self.rgb = rgb
        self.initialized = concurrent.futures.Future()
        self.queue = queue.Queue(32)
        self.timer = DeadlineWaiter()
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()
        self.initialized.result(timeout=60)

    def submit(self, lane, frame, image):
        f = concurrent.futures.Future()
        self.queue.put_nowait((f, image, time.perf_counter(), lane, frame))
        self.timer.notify()
        return f

    def work(self):
        try:
            # Cache cuDNN's thread-local plans on the actual dispatcher thread.
            for n in range(1, self.batch + 1):
                self.model.predict(
                    [np.ascontiguousarray(self.rgb[i % len(self.rgb)][:, :, ::-1]) for i in range(n)],
                    imgsz=640,
                    rect=False,
                    half=False,
                    conf=0.25,
                    iou=0.7,
                    max_det=300,
                    retina_masks=True,
                    verbose=False,
                    quantize=None,
                )
                self.torch.cuda.synchronize()
            self.initialized.set_result(None)
        except Exception as e:
            self.initialized.set_exception(e)
            return
        while True:
            first = self.queue.get()
            if first is None:
                return
            items = [first]
            wait_begin = time.perf_counter()
            deadline = first[2] + self.wait
            while len(items) < self.batch:
                try:
                    items.append(self.queue.get_nowait())
                except queue.Empty:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    self.timer.wait(remaining)
            dispatch = time.perf_counter()
            try:
                result = self.model.predict(
                    [np.ascontiguousarray(i[1][:, :, ::-1]) for i in items],
                    imgsz=640,
                    rect=False,
                    half=False,
                    conf=0.25,
                    iou=0.7,
                    max_det=300,
                    retina_masks=True,
                    verbose=False,
                    quantize=None,
                )
                # Materialize all required results before any request is completed.
                cpu_outputs = []
                for r in result:
                    boxes = r.boxes.data.cpu().numpy()
                    keypoints = None
                    masks = []
                    if r.keypoints is not None:
                        keypoints = r.keypoints.data.cpu().numpy()
                    if r.masks is not None:
                        for box, mask in zip(boxes, r.masks.data):
                            x, y, right, bottom = np.ceil(box[:4]).astype(int)
                            data = (mask[y:bottom, x:right] > 0).to(self.torch.uint8).cpu().numpy()
                            masks.append((int(x), int(y), data))
                    cpu_outputs.append(dict(boxes=boxes, keypoints=keypoints, masks=masks))
                self.torch.cuda.synchronize()
                complete = time.perf_counter()
                for (f, _, submitted, lane, frame), payload in zip(items, cpu_outputs):
                    f.set_result(
                        dict(
                            batch_size=len(items),
                            stream_id=lane,
                            frame_id=frame,
                            queue_ms=max(
                                0, dispatch - submitted - min(dispatch - submitted, dispatch - wait_begin)
                            )
                            * 1000,
                            batch_wait_ms=min(dispatch - submitted, dispatch - wait_begin) * 1000,
                            dispatch_ms=0,
                            completed=complete,
                            outputs=payload,
                        )
                    )
            except Exception as e:
                for item in items:
                    item[0].set_exception(e)

    def close(self):
        self.queue.put(None)
        self.thread.join()


def native(case, rgb):
    # The archived binary is imported only inside an isolated baseline worker.
    if case["backend"] == "before":
        sys.path.insert(0, str(ROOT / "runs/optimization/baseline"))
    from vonnx import EngineConfig, vONNXEngine

    c = EngineConfig()
    c.model_path = str(ROOT / manifest(case["model"])["onnx"])
    c.imgsz = 640
    c.max_batch_size = case["batch"]
    c.max_latency_ms = case["wait"]
    c.max_input_width = max(a.shape[1] for a in rgb)
    c.max_input_height = max(a.shape[0] for a in rgb)
    if hasattr(c, "pipeline_depth"):
        c.pipeline_depth = case["depth"]
        c.concurrency = max(32, case["lanes"])
        c.batch_mode = "latency" if case["lanes"] == 1 else "throughput"
        c.collect_timings = case["timings"]
    return vONNXEngine(c)


def stats(engine):
    if not hasattr(engine, "resource_stats"):
        return None
    s = engine.resource_stats
    return {
        k: getattr(s, k)
        for k in (
            "initialization_count",
            "explicit_device_allocations",
            "hot_path_device_allocations",
            "reserved_device_bytes",
            "arena_limit_bytes",
            "initialized_device_bytes",
            "current_device_bytes",
            "pipeline_depth",
            "timer_backend",
            "state",
        )
    }


def run(case, output):
    import pynvml

    info = manifest(case["model"])
    rgb = load_rgb(images(info["task"]))
    pynvml.nvmlInit()
    dev = pynvml.nvmlDeviceGetHandleByIndex(0)
    peak = [pynvml.nvmlDeviceGetMemoryInfo(dev).used]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.02):
            peak[0] = max(peak[0], pynvml.nvmlDeviceGetMemoryInfo(dev).used)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    eng = None
    try:
        eng = (
            native(case, rgb)
            if case["backend"] in ("native", "before")
            else Official(case["model"], case["batch"], case["wait"], rgb)
        )
        from vonnx import PixelFormat

        def submit(lane, frame):
            a = rgb[frame % len(rgb)]
            return (
                eng.submit_numpy(lane, frame, a, PixelFormat.RGB24)
                if case["backend"] in ("native", "before")
                else eng.submit(lane, frame, a)
            )

        # Exactly 25 completed requests; drain before the measurement barrier.
        for start in range(0, 25, case["lanes"]):
            futures = [submit(i, start + i) for i in range(min(case["lanes"], 25 - start))]
            for f in futures:
                f.result(timeout=30)
        before = stats(eng)
        logs, errors = [], []
        barrier = threading.Barrier(case["lanes"] + 1)
        lock = threading.Lock()

        def lane_work(lane):
            barrier.wait()
            count = (case["total"] - 25) // case["lanes"] + int(lane < (case["total"] - 25) % case["lanes"])
            for n in range(count):
                frame = 25 + lane + n * case["lanes"]
                begin = time.perf_counter()
                try:
                    f = submit(lane, frame)
                    r = f.result(timeout=30)
                    end = time.perf_counter()
                    if case["backend"] in ("native", "before"):
                        r = r[0]
                        row = {
                            k: getattr(r, k, None) for k in ("batch_size", "stream_id", "frame_id", *STAGES)
                        }
                    else:
                        row = {k: r.get(k) for k in ("batch_size", "stream_id", "frame_id", *STAGES)}
                    if row["stream_id"] != lane or row["frame_id"] != frame:
                        raise AssertionError("Request identity mismatch")
                    row = {**row, "latency_ms": (end - begin) * 1000, "submitted": begin, "completed": end}
                    with lock:
                        logs.append(row)
                except Exception as e:
                    with lock:
                        errors.append(dict(stream_id=lane, frame_id=frame, error=str(e)))
                    break

        threads = [threading.Thread(target=lane_work, args=(lane,)) for lane in range(case["lanes"])]
        for t in threads:
            t.start()
        begin = time.perf_counter()
        barrier.wait()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - begin
        after = stats(eng)
        success = len(logs) == case["total"] - 25 and not errors
        if len({r["frame_id"] for r in logs}) != len(logs):
            success = False
            errors.append({"error": "Duplicate completion"})
        result = dict(
            case=case,
            status="passed" if success else "failed",
            success=len(logs),
            failures=errors,
            rejected=sum("capacity" in e["error"] or "full" in e["error"] for e in errors),
            seconds=elapsed,
            fps=len(logs) / elapsed,
            latency=sample_stats([r["latency_ms"] for r in logs]),
            stages={k: sample_stats([r[k] for r in logs if r.get(k) is not None]) for k in STAGES},
            batch_request_histogram=dict(collections.Counter(r["batch_size"] for r in logs)),
            batch_histogram={
                str(b): n // b for b, n in collections.Counter(r["batch_size"] for r in logs).items()
            },
            resources_before=before,
            resources_after=after,
            gpu_peak_bytes=peak[0],
            model=info,
            dataset=[str(p.relative_to(ROOT)) for p in images(info["task"])],
            binary_sha256=(
                build_fingerprint()
                if case["backend"] != "before"
                else sha256(next((ROOT / "runs/optimization/baseline/vonnx").glob("*.pyd")))
            ),
            cpu_outputs_retained=True,
            effective_timings=case["timings"] or case["backend"] == "before",
            effective_config=dict(
                concurrency=max(32, case["lanes"]) if case["backend"] == "native" else None,
                queue_capacity=1024 if case["backend"] in ("native", "before") else 32,
                max_input_height=max(a.shape[0] for a in rgb),
                max_input_width=max(a.shape[1] for a in rgb),
                gpu_memory_budget_mb=6144 if case["backend"] == "native" else None,
                max_mask_bytes=64 * 1024 * 1024 if case["backend"] in ("native", "before") else None,
            ),
        )
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        with (path / (case["id"] + ".jsonl")).open("w", encoding="utf-8") as f:
            for row in logs:
                f.write(json.dumps(row) + "\n")
        result["timer_backend"] = (
            before["timer_backend"]
            if before
            else getattr(getattr(eng, "timer", None), "backend", "legacy_condition_variable")
        )
        write_json(path / (case["id"] + ".json"), result)
        return result
    finally:
        if eng:
            eng.close() if hasattr(eng, "close") else eng.stop()
        stopped.set()
        sampler.join()
        pynvml.nvmlShutdown()


def cases(phase):
    out = []
    for model in MODELS:
        for lanes in (1, 16, 32):
            if phase == "confirm" and lanes == 1:
                continue
            if phase == "diagnostic" and lanes != 32:
                continue
            backends = (
                ("before",)
                if phase == "baseline"
                else ("native",) if phase == "diagnostic" else ("native", "official-batched")
            )
            if phase == "quick" and lanes == 1:
                backends += ("official-single",)
            for backend in backends:
                for repeat in range(3 if phase == "confirm" else 1):
                    for depth, wait in (
                        ((d, w) for d in (1, 2) for w in (0, 2))
                        if phase == "diagnostic"
                        else [(1 if lanes == 1 or backend == "before" else 2, 0 if lanes == 1 else 2)]
                    ):
                        c = dict(
                            model=model,
                            lanes=lanes,
                            backend=backend,
                            depth=depth,
                            wait=wait,
                            batch=1 if backend == "official-single" else 8,
                            total=500 if phase == "confirm" else 100,
                            repeat=repeat + 1,
                            timings=phase == "diagnostic",
                            imgsz=[640, 640],
                            conf=0.25,
                            iou=0.7,
                            max_det=300,
                            dtype="float32",
                            mask_mode="roi",
                            provider_sync=True,
                            cudnn_search=(
                                "DEFAULT"
                                if backend == "before"
                                else "HEURISTIC" if backend == "native" else "pytorch_default"
                            ),
                        )
                        c["id"] = f"{phase}__{model}__{backend}__s{lanes}__d{depth}__w{wait}__r{repeat+1}"
                        out.append(c)
    return (
        sorted(out, key=lambda c: (c["repeat"], c["lanes"], MODELS.index(c["model"]), c["backend"]))
        if phase == "confirm"
        else out
    )


def gate(results, phase):
    comparisons = []
    for model in MODELS:
        for lanes in (16, 32):
            rows = [
                r for r in results if r.get("case", {}).get("model") == model and r["case"]["lanes"] == lanes
            ]
            native = {
                r["case"]["repeat"]: r["fps"]
                for r in rows
                if r.get("status") == "passed" and r["case"]["backend"] == "native"
            }
            official = {
                r["case"]["repeat"]: r["fps"]
                for r in rows
                if r.get("status") == "passed"
                and r.get("cpu_outputs_retained", False)
                and r["case"]["backend"] == "official-batched"
            }
            needed = 3 if phase == "confirm" else 1
            ok = len(native) == len(official) == needed
            n, o = (
                (statistics.median(native.values()), statistics.median(official.values()))
                if ok
                else (None, None)
            )
            wins = sum(native[r] > official[r] for r in native if r in official)
            comparisons.append(
                dict(
                    model=model,
                    lanes=lanes,
                    native_fps=n,
                    official_fps=o,
                    gain_percent=(n / o - 1) * 100 if ok else None,
                    wins=wins,
                    passed=bool(ok and n > o and wins >= (2 if phase == "confirm" else 1)),
                )
            )
    return dict(phase=phase, passed=all(c["passed"] for c in comparisons), comparisons=comparisons)


def report(output):
    output = Path(output)
    results = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in output.glob("*__*.json")
        if not p.name.startswith("check__")
    ]
    rows = []
    for r in results:
        c = r["case"]
        rows.append(
            {
                **{
                    k: c[k]
                    for k in (
                        "id",
                        "model",
                        "backend",
                        "lanes",
                        "depth",
                        "wait",
                        "timings",
                        "repeat",
                        "total",
                    )
                },
                "status": r["status"],
                "effective_timings": r.get("effective_timings", c["timings"] or c["backend"] == "before"),
                "fps": r.get("fps"),
                **{k: r.get("latency", {}).get(k) for k in ("mean_ms", "p95_ms", "p99_ms")},
            }
        )
    if rows:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    lines = [
        "# vONNX s / 640 optimization",
        "",
        "100 total: 25 warmup + 75 measured; confirmation 500 total: 25 + 475 × 3.",
        "CPU results, FP32 / TF32 off, ROI masks, no initialization/download/decode in FPS.",
        "",
        "| Case | Status | FPS | Mean ms | P95 ms | P99 ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        vals = [
            r["id"],
            r["status"],
            *[f"{r[k]:.2f}" if r[k] is not None else "—" for k in ("fps", "mean_ms", "p95_ms", "p99_ms")],
        ]
        lines.append("| " + " | ".join(vals) + " |")
    for phase in ("quick", "confirm"):
        subset = [r for r in results if r["case"]["id"].startswith(phase + "__")]
        if subset:
            g = gate(subset, phase)
            write_json(output / (phase + "_gate.json"), g)
            lines += [
                "",
                f"{phase} gate: {'PASS' if g['passed'] else 'NOT MET'}",
                "",
                "| Model | Lanes | Native FPS | Official batched FPS | Gain | Wins |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for c in g["comparisons"]:
                if c["gain_percent"] is not None:
                    label = " (small advantage)" if 0 < c["gain_percent"] < 5 else ""
                    lines.append(
                        f"| {c['model']} | {c['lanes']} | {c['native_fps']:.2f} | {c['official_fps']:.2f} | {c['gain_percent']:+.2f}%{label} | {c['wins']} |"
                    )
                else:
                    lines.append(f"| {c['model']} | {c['lanes']} | — | — | incomplete | — |")
    lines += [
        "",
        "## Before / after",
        "",
        "| Model | Lanes | Before FPS | After FPS | Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for lanes in (1, 16, 32):

            def find(phase, backend):
                return next(
                    (
                        r
                        for r in results
                        if r["case"]["id"].startswith(phase + "__")
                        and r["case"]["backend"] == backend
                        and r["case"]["model"] == model
                        and r["case"]["lanes"] == lanes
                        and r["status"] == "passed"
                    ),
                    None,
                )

            a, b = find("baseline", "before"), find("quick", "native")
            if a and b:
                lines.append(
                    f"| {model} | {lanes} | {a['fps']:.2f} | {b['fps']:.2f} | {(b['fps']/a['fps']-1)*100:+.2f}% |"
                )
    check_results = [json.loads(p.read_text(encoding="utf-8")) for p in output.glob("check__*.json")]
    for r in check_results:
        lines += ["", f"Correctness {r['case']['model']}: {r['status']}"]
    if (output / "environment.json").exists():
        env = json.loads((output / "environment.json").read_text())
        lines += ["", f"Hardware: {env['gpu']}", "", "Versions: `" + json.dumps(env["versions"]) + "`"]
    lines += [
        "",
        "No complete dataset validation or full benchmark is started by this command.",
        "Resource counters cover vONNX cudaMalloc calls; ORT/cuDNN allocations are not intercepted.",
        "CUDA free-memory deltas and NVML peak are device-wide and can include other applications.",
        "Detailed timings are separate diagnostic runs; primary comparisons have timings disabled.",
        "The archived before build always collected GPU timings; disabling that cost is part of the optimization.",
        "Old queue_ms included batch waiting; do not compare it directly with the new separated queue_ms.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if any(r["case"]["id"].startswith("quick__") for r in results):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, model in zip(axes, MODELS):
            for offset, (phase, backend, label) in enumerate(
                (
                    ("baseline", "before", "Before"),
                    ("quick", "native", "After"),
                    ("quick", "official-batched", "Official batched"),
                )
            ):
                vals = [
                    next(
                        (
                            r["fps"]
                            for r in results
                            if r["status"] == "passed"
                            and r["case"]["id"].startswith(phase + "__")
                            and r["case"]["model"] == model
                            and r["case"]["lanes"] == lane
                            and r["case"]["backend"] == backend
                        ),
                        np.nan,
                    )
                    for lane in (1, 16, 32)
                ]
                ax.bar(np.arange(3) + (offset - 1) * 0.25, vals, 0.25, label=label)
            ax.set(
                title=model,
                ylabel="Completed CPU results / second",
                xticks=np.arange(3),
                xticklabels=[1, 16, 32],
                xlabel="Request lanes",
            )
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "throughput.png", dpi=150)
        plt.close(fig)
    confirmation = [r for r in results if r["case"]["id"].startswith("confirm__")]
    if confirmation:
        g = gate(confirmation, "confirm")
        if g["passed"]:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, model in zip(axes, MODELS):
                data = [c for c in g["comparisons"] if c["model"] == model]
                for offset, (key, label) in enumerate(
                    (("native_fps", "vONNX"), ("official_fps", "Official batched"))
                ):
                    ax.bar(np.arange(2) + (offset - 0.5) * 0.35, [c[key] for c in data], 0.35, label=label)
                ax.set(
                    title=model,
                    ylabel="Median CPU results / second",
                    xticks=np.arange(2),
                    xticklabels=[16, 32],
                    xlabel="Request lanes",
                )
                ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output / "confirmation.png", dpi=150)
            plt.close(fig)


def optimize(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        case = json.loads(Path(args.worker).read_text(encoding="utf-8"))
        try:
            if case.get("correctness"):
                from .validate import smoke_case

                r = {
                    "case": case,
                    "status": "passed",
                    "binary_sha256": build_fingerprint(),
                    "correctness": smoke_case(
                        {"model": case["model"], "imgsz": [640, 640], "batch_limits": [1, 2, 4, 8]}
                    ),
                }
                write_json(output / (case["id"] + ".json"), r)
            else:
                r = run(case, output)
            if r["status"] != "passed":
                raise RuntimeError("Request failures; stop this experiment")
        except Exception as e:
            write_json(output / (case["id"] + ".json"), {"case": case, "status": "failed", "error": repr(e)})
            raise
        return
    if args.phase == "report":
        report(output)
        return
    if args.phase in ("quick", "confirm"):
        current = build_fingerprint()
        for model in MODELS:
            p = output / ("check__" + model + ".json")
            if not p.exists():
                raise RuntimeError("Run optimize --phase check before speed comparisons")
            r = json.loads(p.read_text())
            if r.get("status") != "passed" or r.get("binary_sha256") != current:
                raise RuntimeError("Correctness results are missing or stale; run optimize --phase check")
    if args.phase == "confirm":
        report(output)
        p = output / "quick_gate.json"
        if not p.exists() or not json.loads(p.read_text())["passed"]:
            raise RuntimeError("Confirmation requires all six quick throughput comparisons to pass")
        current = build_fingerprint()
        for p in output.glob("quick__*.json"):
            r = json.loads(p.read_text())
            if r.get("status") == "passed" and r.get("binary_sha256") != current:
                raise RuntimeError("Quick results belong to another build; rerun quick before confirmation")
    write_json(output / "environment.json", environment())
    candidates = (
        [dict(id="check__" + m, model=m, lanes=1, repeat=1, correctness=True) for m in MODELS]
        if args.phase == "check"
        else cases(args.phase)
    )
    selected = [c for c in candidates if not args.case or args.case == c["id"]]
    if not selected:
        raise ValueError("Unknown case")
    if args.list:
        for c in selected:
            print(c["id"])
        return
    # A round includes all three models for one lane/repeat; cumulative child wall time <= 5 min.
    round_time = collections.defaultdict(float)
    for c in selected:
        target = output / (c["id"] + ".json")
        if args.resume and target.exists():
            previous = json.loads(target.read_text())
            compatible_outputs = not c.get("backend", "").startswith("official") or previous.get(
                "cpu_outputs_retained", False
            )
            if (
                previous["status"] == "passed"
                and compatible_outputs
                and previous.get("case") == c
                and (c.get("backend") == "before" or previous.get("binary_sha256") == build_fingerprint())
            ):
                continue
        round_id = (c["lanes"], c["repeat"])
        remaining = 300 - round_time[round_id]
        if remaining <= 0:
            raise TimeoutError("Round GPU time budget exhausted; stopping without further GPU work")
        case_path = output / "active_case.json"
        write_json(case_path, c)
        begin = time.perf_counter()
        print("Run " + c["id"], flush=True)
        with (output / (c["id"] + ".log")).open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "vonnx_tools",
                        "optimize",
                        "--worker",
                        str(case_path),
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=min(args.timeout, remaining),
                )
                failed = completed.returncode != 0
                if failed:
                    previous = json.loads(target.read_text()) if target.exists() else {"case": c}
                    write_json(
                        target,
                        {
                            **previous,
                            "status": "failed",
                            "process_exitcode": completed.returncode,
                            "error": "Worker failed, including initialization or cleanup; measurements are invalid",
                        },
                    )
            except subprocess.TimeoutExpired:
                write_json(
                    target, {"case": c, "status": "timeout", "error": "GPU round/case time budget exceeded"}
                )
                failed = True
        round_time[round_id] += time.perf_counter() - begin
        if args.phase != "check":
            report(output)
        if failed:
            raise RuntimeError("Failure/timeout; stopped without reducing batch or continuing GPU work")
    if args.phase != "check":
        report(output)
