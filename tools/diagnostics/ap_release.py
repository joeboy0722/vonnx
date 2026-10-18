"""Bounded 500-image release AP gate; no throughput benchmark or publishing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vonnx_tools.common import build_fingerprint, sha256, write_json, environment, manifest
from vonnx_tools.validate import full_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'runs/v050/ap500')
    p.add_argument('--worker', type=Path)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        case = json.loads(args.worker.read_text(encoding='utf-8'))
        started = time.monotonic()
        detail = full_case(case, args.output)
        write_json(args.output / (case['id']+'.json'), dict(case=case, status='passed',
                   detail=detail, environment=environment(), model=manifest(case['model']),
                   elapsed_seconds=time.monotonic()-started))
        return
    rows = []
    for precision in ('fp32', 'fp16'):
        for mode in ('off', 'specialized', 'graph'):
            for model in ('yolov8s', 'yolov8s-pose', 'yolov8s-seg'):
                case = dict(id=f'{model}-{precision}-{mode}', model=model, precision=precision,
                            imgsz=[640,640], limit=500, graph_specialization='off' if mode=='off' else 'auto',
                            cuda_graph=mode=='graph', build_fingerprint=build_fingerprint(),
                            conversion_fingerprint=sha256(ROOT/'vonnx/precision.py'),
                            model_hash=manifest(model)['onnx_sha256'])
                record = args.output/(case['id']+'.json')
                if args.resume and record.exists():
                    row=json.loads(record.read_text(encoding='utf-8'))
                    if row.get('case')==case and row.get('status')=='passed':
                        rows.append(row)
                        print(case['id'], 'passed (resume)', flush=True)
                        continue
                config=args.output/(case['id']+'.case.json')
                write_json(config,case)
                print(case['id'], 'starting 500-image AP', flush=True)
                with (args.output/(case['id']+'.log')).open('w',encoding='utf-8') as log:
                    try:
                        proc=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),
                            '--worker',str(config.resolve()),'--output',str(args.output.resolve())],
                            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1200,
                            env={**os.environ,'PYTHONUTF8':'1','YOLO_CONFIG_DIR':str(ROOT/'runs/ultralytics')})
                        if proc.returncode:
                            raise RuntimeError(f'Worker exit {proc.returncode}; see {log.name}')
                    except Exception as exc:
                        write_json(record,dict(case=case,status='failed',error=str(exc)))
                        raise
                rows.append(json.loads(record.read_text(encoding='utf-8')))
                write_json(args.output/'summary.json',rows)
                print(case['id'], 'passed', flush=True)
    write_json(args.output/'summary.json',rows)
    write_json(args.output/'gate.json',dict(passed=len(rows)==18,cases=len(rows),
               images_per_case=500,scope='COCO val2017 deterministic subset; not full AP',
               binary_sha256=build_fingerprint(),converter_sha256=sha256(ROOT/'vonnx/precision.py')))


if __name__=='__main__':
    main()
