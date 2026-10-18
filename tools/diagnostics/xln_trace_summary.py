"""Separate host submission, own-stream kernels and shared-GPU gaps."""
import collections
import json
from pathlib import Path
import sqlite3
import statistics
import sys

file=Path(sys.argv[1])
model_names=sys.argv[2].split(',') if len(sys.argv)>2 else ('yolov8x','yolov8l','yolov8n')
c=sqlite3.connect(file);c.row_factory=sqlite3.Row
kernels=[dict(r) for r in c.execute('SELECT k.start,k.end,k.streamId,k.correlationId,s.value AS name,r.end AS api_end '
    'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName '
    'LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId ORDER BY k.start')]
streams=sorted({k['streamId'] for k in kernels if 'vonnx::letterbox' in k['name']})
summary=[]
for model,stream in zip(model_names,streams):
    same=[k for k in kernels if k['streamId']==stream]
    records=[]
    for score in [k for k in same if 'vonnx::score_kernel' in k['name']]:
        pres=[k for k in same if 'vonnx::letterbox' in k['name'] and k['end']<=score['start']]
        if not pres:continue
        pre=pres[-1]
        selected=[k for k in same if pre['end']<=k['start']<score['start']]
        if not selected:continue
        last=selected[-1]
        record=dict(own_kernel_ms=sum(k['end']-k['start'] for k in selected)/1e6,
                    inference_span_ms=(last['end']-pre['end'])/1e6,
                    last_launch_from_pre_ms=(last['api_end']-pre['end'])/1e6 if last['api_end'] else None,
                    last_kernel_wait_after_launch_ms=(last['end']-last['api_end'])/1e6 if last['api_end'] else None,
                    kernel_count=len(selected))
        record['non_kernel_span_ms']=record['inference_span_ms']-record['own_kernel_ms']
        records.append(record)
    thread=c.execute('SELECT r.globalTid,COUNT(*) AS n FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                     'JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId '
                     'WHERE k.streamId=? GROUP BY r.globalTid ORDER BY n DESC LIMIT 1',(stream,)).fetchone()
    sync=c.execute('SELECT COUNT(*) AS n,SUM(r.end-r.start)/1e6 AS ms FROM CUPTI_ACTIVITY_KIND_RUNTIME r '
                   'JOIN StringIds s ON s.id=r.nameId WHERE r.globalTid=? AND s.value LIKE "cudaEventSynchronize%"',
                   (thread['globalTid'],)).fetchone()
    summary.append(dict(model=model,stream_id=stream,batches=len(records),
                        event_wait_calls=sync['n'], event_wait_calls_per_batch=sync['n']/len(records),
                        event_wait_total_ms=sync['ms'],
                        median={key:statistics.median([r[key] for r in records if r[key] is not None]) if any(r[key] is not None for r in records) else None for key in records[0]},
                        first=records[0],records=records))
apis=[dict(r) for r in c.execute('SELECT s.value AS name,COUNT(*) AS calls,SUM(r.end-r.start)/1e6 AS total_ms '
    'FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId '
    'WHERE s.value LIKE "%Synchronize%" OR s.value LIKE "%Malloc%" OR s.value LIKE "%Free%" '
    'GROUP BY s.value')]
result=dict(streams=summary,apis=apis,
    caveat='Non-kernel span includes other streams and driver scheduling; not entirely removable CPU overhead. Stream mapping follows independent engine initialization order X/L/n.')
file.with_suffix('.summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(streams=[{k:v for k,v in r.items() if k!='records'} for r in summary],apis=apis),indent=2))
