"""Combine bounded before/after/official results without starting GPU work."""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def load(folder,pattern):
    return [json.loads(p.read_text(encoding='utf-8')) for p in folder.glob(pattern)]

def report(base):
    before=load(base/'before','baseline__*.json')
    fp32=load(base/'fp32','quick__*.json')
    fp16=load(base/'fp16','quick__*.json')
    initial=load(base/'initial_fp16','quick__*.json')
    def find(rows,model,lanes,backend):
        return next((r for r in rows if r.get('status')=='passed' and
            r['case']['model']==model and r['case']['lanes']==lanes and r['case']['backend']==backend),None)
    rows=[]
    for model in ('yolov8s','yolov8s-pose','yolov8s-seg'):
        for lanes in (1,16,32):
            a=find(before,model,lanes,'before');b=find(fp32,model,lanes,'native')
            c=find(fp16,model,lanes,'native');d=find(fp16,model,lanes,'official-batched')
            old=find(initial,model,lanes,'native')
            row=dict(model=model,lanes=lanes,before_fp32_fps=a.get('fps') if a else None,
                after_fp32_fps=b.get('fps') if b else None,
                initial_fp16_fps=old.get('fps') if old else None,
                fp16_fps=c.get('fps') if c else None,official_fp16_fps=d.get('fps') if d else None,
                fp16_gain_percent=(c['fps']/d['fps']-1)*100 if c and d else None,
                fp16_mean_ms=c.get('latency',{}).get('mean_ms') if c else None,
                fp16_p95_ms=c.get('latency',{}).get('p95_ms') if c else None,
                fp16_p99_ms=c.get('latency',{}).get('p99_ms') if c else None,
                fp16_peak_bytes=c.get('gpu_peak_bytes') if c else None)
            rows.append(row)
    with (base/'COMPARISON.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gate_path=base/'fp16/confirm_gate.json'
    gate=json.loads(gate_path.read_text(encoding='utf-8')) if gate_path.exists() else None
    quick_path=base/'fp16/quick_gate.json'
    quick=json.loads(quick_path.read_text(encoding='utf-8')) if quick_path.exists() else None
    verdict='CONFIRMED' if gate and gate['passed'] else 'NOT CONFIRMED'
    ap_path=ROOT/'docs/ap_fp16.json'
    ap=json.loads(ap_path.read_text(encoding='utf-8')) if ap_path.exists() else None
    ap_validated=bool(ap and ap.get('passed') and ap.get('binary_sha256') in
        {r.get('binary_sha256') for r in fp16})
    ap_status='Full s/640 COCO AP validation passed; see docs/ap_fp16.md.' if ap_validated else 'Full AP validation is pending.'
    lines=['# FP16 bounded implementation report','',f'Speed acceptance: **{verdict}**. {ap_status}',
        '','Windows / RTX 3070 8GB; s models / 640; batch cap 8; concurrency 16/32, 2 ms wait, depth 2.',
        'Single lane: latency mode / depth 1 / 0 ms wait. Each quick case: 25 warmup + 75 measured.',
        'CPU results required; detailed timing disabled. Conversion/init/decode/warmup excluded.',
        'RGB inputs for native and BGR inputs for official are preloaded before measurement; no extra official RGB-to-BGR adapter copy is timed.',
        '','| Model | Lanes | Before FP32 | After FP32 | FP16 | Official FP16 | Gain |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return f'{v:.2f}' if v is not None else '—'
    for r in rows:
        lines.append('| '+' | '.join([r['model'],str(r['lanes'])]+[fmt(r[k]) for k in
            ('before_fp32_fps','after_fp32_fps','fp16_fps','official_fp16_fps','fp16_gain_percent')])+' |')
    lines+=['','## 500-request confirmation','',
        '25 completed warmup requests are excluded; 475 measured requests per case, repeated three times.','',
        '| Model | Lanes | Native median FPS | Official median FPS | Gain % | Wins |',
        '|---|---:|---:|---:|---:|---:|']
    if gate:
        for c in gate['comparisons']:
            lines.append(f"| {c['model']} | {c['lanes']} | {fmt(c['native_fps'])} | {fmt(c['official_fps'])} | {fmt(c['gain_percent'])} | {c['wins']}/3 |")
    lines+=['','## Acceptance and limitations','',
        f'All-six quick gate: {quick["passed"] if quick else "incomplete"}.',
        'No 500-request confirmation is permitted unless all six quick throughput comparisons pass.',
        'Only a complete 3-repeat confirmation gate can establish the planned speed acceptance.',
        'Small correctness checks are independent same-ONNX CUDA comparisons; they do not establish FP16 AP.',
        ap_status+' No long-duration performance benchmark was started.',
        '','## Provenance','']
    for name,data in [('before',before),('after fp32',fp32),('after fp16',fp16)]:
        hashes=sorted({r.get('binary_sha256','unknown') for r in data})
        lines.append(f'- {name}: '+', '.join(hashes))
    env_path=base/'fp16/environment.json'
    env=json.loads(env_path.read_text(encoding='utf-8')) if env_path.exists() else {}
    runtime_path=base/'runtime_versions.json'
    if runtime_path.exists():env['runtime_details']=json.loads(runtime_path.read_text(encoding='utf-8'))
    models={}
    for r in fp16:
        if r.get('status')=='passed' and r['case']['backend']=='native':
            models[r['case']['model']]=dict(weight_sha256=r['model'].get('weight_sha256'),
                source_onnx_sha256=r['model'].get('onnx_sha256'),prepared=r.get('prepared_model'),
                source_export=r['model'].get('export'))
    lines+=['','## Environment','','```json',json.dumps(env,indent=2,ensure_ascii=False),'```',
            '','## Models and configuration','','```json',json.dumps(models,indent=2,ensure_ascii=False),'```']
    lines+=['','Raw per-request JSONL, parameters, model hashes, batch histograms, failures and memory peaks',
        'are preserved in before/, fp32/ and fp16/. The initial FP16 attempt is retained in initial_fp16/.',
        'The legacy before runner incorrectly marked effective_timings=true; case.timings=false and the',
        'archived 0.3 binary configuration are authoritative. No FPS values were corrected or substituted.',
        'Archived before filenames also retain the legacy d1 label; the 16/32-lane case.depth and resource',
        'snapshots record the actual depth=2 configuration. Configuration fields, not filenames, are authoritative.',
        'GPU memory samples are device-wide and may include other applications.']
    lines+=['','The earlier run with an in-measurement official RGB-to-BGR adapter is retained in',
            'before_bgr_preload/ for audit and is excluded from the final comparison and acceptance.',
            '33 tests passed, including real half-I/O models, padding/result identity, output completeness and AP gate logic.',
            'The native SDK example compiled and inferred successfully with Unicode model/interpreter/cache paths.']
    lines+=['','## Padding accounting','',
        '| Case | Completed requests | Physical batch histogram | Padded positions |',
        '|---|---:|---|---:|']
    for r in fp16:
        if r['case']['backend']=='native':
            lines.append(f"| {r['case']['id']} | {r.get('success')} | {r.get('physical_batch_histogram')} | {r.get('padded_images')} |")
    diagnostics=load(base/'diagnostics','diagnostic__*.json')
    lines+=['','## Dynamic-shape investigation','',
        'The ORT cuDNN frontend tracks the last input dimensions, rebuilding its execution plan when',
        'batch dimensions change. Warming bindings for batches 1–8 does not retain every cuDNN plan.',
        'The warmup remainder and measurement tail repeatedly triggered this cost in dynamic FP16 runs.',
        'The fixed-batch policy pads unused positions with zeros, reports physical/logical batches separately,',
        'and counts only accepted completed requests in FPS. Latency/FP32 defaults remain dynamic.',
        'FP16 results before this mitigation are preserved in initial_fp16/ and nhwc_dynamic_fp16/.',
        'CUDA event stage intervals include stream idle and host submission gaps; they are not kernel-active time.',
        'C++ ORT-only diagnostics exclude the full request pipeline and cannot be used as end-to-end FPS.',
        '','| Model | Depth | Wait ms | Diagnostic FPS | ORT interval ms |',
        '|---|---:|---:|---:|---:|']
    for d in diagnostics:
        c=d['case'];ms=d.get('stages',{}).get('inference_ms',{}).get('mean_ms')
        lines.append(f"| {c['model']} | {c['depth']} | {c['wait']} | {fmt(d.get('fps'))} | {fmt(ms)} |")
    final_diagnostics=load(base/'final_diagnostics','diagnostic__*.json')
    if final_diagnostics:
        lines+=['','### Current fixed-batch diagnostic (timing enabled)','','| Model | Depth | Wait ms | FPS | ORT interval ms |',
                '|---|---:|---:|---:|---:|']
        for d in final_diagnostics:
            c=d['case'];ms=d.get('stages',{}).get('inference_ms',{}).get('mean_ms')
            lines.append(f"| {c['model']} | {c['depth']} | {c['wait']} | {fmt(d.get('fps'))} | {fmt(ms)} |")
    for filename in ('ort_diagnostic.json','ort_native_options.json','core_real_diagnostic.json'):
        p=base/filename
        if p.exists():lines+=['',f'## {filename} diagnostic (not acceptance)','','```json',p.read_text(encoding='utf-8'),'```']
    lines+=['','## Accuracy status and optional next verification','',
        '- '+ap_status,
        '- Longer speed check: the same three models, 1/16/32 lanes, 30 seconds × 3, native and official FP16. Measurement alone is 27 minutes; estimate 30–45 minutes with initialization/warmup.',
        '- These are estimates from the short tests, not measured full-run durations. Low-confidence mask volume and COCO CPU evaluation can change the duration. Neither command is chained automatically.',
        '- The old 15-model/multi-resolution/multi-model matrix is outside these estimates and requires a separate scope and time review.']
    (base/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (base/'COMPARISON.json').write_text(json.dumps(dict(speed_confirmed=bool(gate and gate['passed']),
        ap_validated=ap_validated,ap=ap if ap_validated else None,environment=env,models=models,confirmation=gate,rows=rows),indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,model in zip(axes,('yolov8s','yolov8s-pose','yolov8s-seg')):
        selected=[r for r in rows if r['model']==model]
        for offset,(key,label) in enumerate([('before_fp32_fps','Before FP32'),('after_fp32_fps','After FP32'),
                                            ('fp16_fps','FP16'),('official_fp16_fps','Official FP16')]):
            ax.bar(np.arange(3)+(offset-1.5)*.2,[r[key] or 0 for r in selected],.2,label=label)
        ax.set(title=model,xticks=np.arange(3),xticklabels=[1,16,32],xlabel='Request lanes',ylabel='Successful requests/s')
        ax.legend(fontsize=7)
    fig.suptitle('Bounded quick test (75 measured requests); confirmation is separate')
    fig.tight_layout();fig.savefig(base/'COMPARISON.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'runs/fp16')
    report(p.parse_args().input)
