"""Kernel/idle intervals from bounded Nsight traces; no GPU execution."""
import collections
import json
from pathlib import Path
import sqlite3
import statistics

ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'runs/latency_opt/traces'
rows=[]
for file in sorted(folder.glob('*.sqlite')):
    connection=sqlite3.connect(file);connection.row_factory=sqlite3.Row
    kernels=[dict(r) for r in connection.execute('SELECT k.start,k.end,k.streamId,k.correlationId,s.value AS name '
        'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start')]
    scores=[r for r in kernels if 'vonnx::score_kernel' in r['name']]
    intervals=[];categories=collections.Counter()
    for score in scores:
        same=[r for r in kernels if r['streamId']==score['streamId'] and r['start']<score['start']]
        letterboxes=[r for r in same if 'vonnx::letterbox' in r['name']]
        if not letterboxes:continue
        pre=letterboxes[-1]
        selected=[r for r in same if r['start']>=pre['end']]
        if not selected:continue
        span=(score['start']-pre['end'])/1e6
        active=sum(r['end']-r['start'] for r in selected)/1e6
        # Same compute stream: kernels cannot overlap; gap also includes non-kernel
        # stream activity and is an upper bound on CPU submission/replay opportunity.
        intervals.append(dict(span_ms=span,kernel_ms=active,non_kernel_ms=span-active,
                              non_kernel_percent=100*(span-active)/span))
        for kernel in selected:
            name=kernel['name'].lower()
            category='layout' if 'transpose' in name else 'cast' if 'op_cast' in name else \
                     'conv_gemm' if any(s in name for s in ('cudnn','convol','gemm','xmma','winograd')) else 'other'
            categories[category]+=(kernel['end']-kernel['start'])/1e6
    row=dict(case=file.stem,batches=len(intervals),
        median={k:statistics.median(r[k] for r in intervals) for k in intervals[0]},
        kernel_categories_total_ms=categories,intervals=intervals,
        non_kernel_is_upper_bound=True)
    rows.append(row)
    print(row['case'],row['median'])
(folder/'summary.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
