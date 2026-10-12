import json
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, write_json


def isolated(command, case, output, timeout, extra=(), resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = output / (case["id"] + ".json")
    config = output / (case["id"] + ".case.json")
    if resume and record.exists():
        old = json.loads(record.read_text(encoding="utf-8"))
        if old.get("case") == case:
            return old
    write_json(config, case)
    argv = [
        sys.executable,
        "-m",
        "vonnx_tools",
        command,
        "--worker",
        str(config.resolve()),
        "--output",
        str(output.resolve()),
        *extra,
    ]
    started = time.monotonic()
    with open(output / (case["id"] + ".log"), "w", encoding="utf-8") as log:
        try:
            result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            if result.returncode != 0:
                write_json(
                    record,
                    {
                        "case": case,
                        "status": "failed",
                        "error": f"Worker exit {result.returncode}; see log",
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
            elif not record.exists():
                raise RuntimeError("Worker omitted result")
        except subprocess.TimeoutExpired:
            write_json(
                record,
                {
                    "case": case,
                    "status": "timeout",
                    "error": f"Exceeded {timeout}s",
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
    row = json.loads(record.read_text(encoding="utf-8"))
    print(case["id"], row["status"], flush=True)
    return row
