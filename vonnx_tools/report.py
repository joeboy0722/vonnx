import csv
import json
from pathlib import Path
from .common import write_json


def report(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(directory)
    rows = []
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".case.json") or path.name in ("matrix.json", "summary.json"):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(row, dict) and "case" in row and "status" in row:
            rows.append(row)
    baselines = {}
    for row in rows:
        c = row["case"]
        if c["kind"] == "single" and c["backend"] == "native" and row["status"] == "passed":
            baselines[(c["models"][0], c["imgsz"], c["streams_per_model"], c["repeat"])] = row["per_model"][0]
    flat = []
    for row in rows:
        c = row["case"]
        for model in row.get("per_model", []):
            entry = {
                "case": c["id"],
                "kind": c["kind"],
                "backend": c["backend"],
                "imgsz": c["imgsz"],
                "streams_per_model": c["streams_per_model"],
                "repeat": c["repeat"],
                "status": row["status"],
                "unvalidated": c.get("unvalidated", False),
                **model,
                "total_fps": row["total_fps"],
                "gpu_peak_mib": row["gpu_peak_mib"],
            }
            base = baselines.get((model["model"], c["imgsz"], c["streams_per_model"], c["repeat"]))
            if base and c["kind"] != "single":
                entry["fps_vs_single"] = model["fps"] / base["fps"] if base["fps"] else None
                entry["p95_vs_single"] = model["p95_ms"] / base["p95_ms"] if base["p95_ms"] else None
            flat.append(entry)
        if not row.get("per_model"):
            flat.append(
                {
                    "case": c["id"],
                    "kind": c["kind"],
                    "backend": c["backend"],
                    "imgsz": c["imgsz"],
                    "streams_per_model": c["streams_per_model"],
                    "repeat": c["repeat"],
                    "status": row["status"],
                    "error": row.get("error", "See case log"),
                }
            )
    write_json(root / "summary.json", {"results": rows, "per_model": flat})
    columns = sorted({k for r in flat for k in r})
    with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    md = [
        "# vONNX YOLOv8 benchmark",
        "",
        f'Cases: {len(rows)}; passed: {sum(r["status"]=="passed" for r in rows)}.',
        "",
        "Latency includes submission, queueing, GPU pipeline and CPU outputs. FPS counts successful model inferences; fanout additionally reports all-model completion latency. GPU memory is device-wide, sampled every 50 ms. Stage timings are shared physical-batch timings, not summed per-request GPU work.",
        "",
        "Validation scope and image counts are recorded in each case's validation_scope. Sampled validation does not certify full COCO val2017 AP.",
        "Request-limited cases are quick finite-workload measurements, not sustained 30-second benchmarks; case metadata records the request cap, duration limit and repeat count.",
        "",
        "| Case | Model | Status | FPS | Mean ms | P95 ms | P99 ms | Peak MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in flat:
        fmt = lambda key: f"{r[key]:.3f}" if isinstance(r.get(key), (int, float)) else "-"
        md.append(
            f'| {r["case"]} | {r.get("model","-")} | {r["status"]} | {fmt("fps")} | {fmt("mean_ms")} | {fmt("p95_ms")} | {fmt("p99_ms")} | {fmt("gpu_peak_mib")} |'
        )
    md += [
        "",
        "All hashes, versions, parameters and errors are retained in summary.json and per-case logs. Unvalidated development runs must not be treated as final benchmark results.",
    ]
    (root / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    for key, label in [("fps", "Successful inferences/s"), ("p95_ms", "P95 latency (ms)")]:
        candidates = [r for r in flat if r.get(key) is not None]
        if candidates:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.scatter(range(len(candidates)), [r[key] for r in candidates], s=8)
            ax.set_xlabel("Case/model index (see summary.csv)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(root / (key + ".png"), dpi=160)
            plt.close(fig)
    print(root / "REPORT.md")
