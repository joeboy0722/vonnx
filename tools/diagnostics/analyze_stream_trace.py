"""Inspect CUDA stream/API correlations in an Nsight Systems SQLite export."""

import sqlite3
import sys

c = sqlite3.connect(sys.argv[1])
c.row_factory = sqlite3.Row
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
for table in tables:
    if any(s in table for s in ("RUNTIME", "KERNEL", "MEMCPY", "MEMSET", "SYNCHRON", "StringIds")):
        print(table, [r[1] for r in c.execute("PRAGMA table_info(" + table + ")")])
for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
    if table in tables:
        print(
            table,
            "stream summary",
            [
                dict(r)
                for r in c.execute(
                    "SELECT streamId,COUNT(*) AS count,SUM(end-start)/1e6 AS total_ms FROM "
                    + table
                    + " GROUP BY streamId"
                )
            ],
        )
if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
    print(
        "API per thread",
        [
            dict(r)
            for r in c.execute(
                'SELECT s.value, r.globalTid,COUNT(*) AS count,AVG(r.end-r.start)/1e6 AS avg_ms FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE "%Synchronize%" OR s.value LIKE "%Memcpy%" GROUP BY s.value,r.globalTid'
            )
        ],
    )
    print(
        "Sync target streams",
        [
            dict(r)
            for r in c.execute(
                "SELECT n.value, a.streamId,a.syncType,COUNT(*) AS count,AVG(a.end-a.start)/1e6 AS avg_ms FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION a JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=a.correlationId JOIN StringIds n ON n.id=r.nameId GROUP BY n.value,a.streamId,a.syncType"
            )
        ],
    )
    print(
        "First kernel per stream",
        [
            dict(r)
            for r in c.execute(
                "SELECT k.streamId,s.value,MIN(k.start) AS start FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName GROUP BY k.streamId ORDER BY k.streamId"
            )
        ],
    )
    for stream in (13, 48, 83, 118):
        kernels = list(
            c.execute(
                "SELECT k.start,k.end,s.value,r.end AS api_end FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId WHERE k.streamId=? ORDER BY k.start",
                (stream,),
            )
        )
        if not kernels:
            continue
        start = kernels[0]["start"]
        score = next((k for k in kernels if "vonnx::score_kernel" in k["value"]), None)
        if score:
            selected = [k for k in kernels if start <= k["start"] < score["start"]]
            print(
                "First run",
                stream,
                "begin_ms",
                start / 1e6,
                "postprocess_start_ms",
                score["start"] / 1e6,
                "last_inference_kernel_end_ms",
                max(k["end"] for k in selected) / 1e6,
                "kernel_active_ms",
                sum(k["end"] - k["start"] for k in selected) / 1e6,
                "last_kernel_name",
                selected[-1]["value"],
                "last_api_end_ms",
                selected[-1]["api_end"] / 1e6,
            )
