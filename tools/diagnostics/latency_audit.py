"""Bounded FP32/FP16 latency evidence, using the unchanged released binary."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vonnx_tools.common import write_json,build_fingerprint,sha256,manifest,environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    out=ROOT/'runs/latency_audit'
    out.mkdir(parents=True,exist_ok=True)
    if args.worker:
        from vonnx_tools import optimize
        from vonnx import EngineConfig,vONNXEngine
        case=json.loads(args.worker.read_text(encoding='utf-8'))
        original=optimize.native
        def factory(c,rgb):
            if 'batch_mode_override' not in c:return original(c,rgb)
            cfg=EngineConfig()
            cfg.model_path=str(ROOT/manifest(c['model'])['onnx'])
            cfg.precision=c['precision'];cfg.model_cache_dir=str(ROOT/'runs/fp16/cache')
            cfg.imgsz=640;cfg.max_batch_size=c['batch'];cfg.concurrency=32
            cfg.max_latency_ms=c['wait'];cfg.pipeline_depth=c['depth']
            cfg.batch_mode=c['batch_mode_override'];cfg.batch_shape_policy=c['batch_shape_policy']
            cfg.max_input_width=max(im.shape[1] for im in rgb)
            cfg.max_input_height=max(im.shape[0] for im in rgb)
            cfg.collect_timings=True;cfg.conf_threshold=.25;cfg.iou_threshold=.7;cfg.max_det=300
            return vONNXEngine(cfg)
        optimize.native=factory
        optimize.run(case,out)
        return
    cases=[]
    for precision in ('fp32','fp16'):
        for model in optimize_models():
            for lanes in (1,32):
                cases.append(dict(id=f'{precision}_{model}_lanes{lanes}',model=model,precision=precision,
                    lanes=lanes,backend='native',depth=1 if lanes==1 else 2,wait=0 if lanes==1 else 2,
                    batch=8,total=100,repeat=1,timings=True,imgsz=[640,640],conf=.25,iou=.7,max_det=300,
                    batch_shape_policy='auto',conversion_fingerprint=sha256(ROOT/'vonnx/precision.py'),
                    source_model_sha256=manifest(model)['onnx_sha256']))
    for precision,policy in (('fp32','auto'),('fp16','auto'),('fp16','dynamic')):
        case=dict(next(c for c in cases if c['precision']==precision and c['model']=='yolov8s' and c['lanes']==1))
        case.update(id=f'{precision}_yolov8s_single_throughput_{policy}',batch_mode_override='throughput',
                    depth=2,wait=2,batch_shape_policy=policy)
        cases.append(case)
    for model in optimize_models():
        case=dict(next(c for c in cases if c['precision']=='fp32' and c['model']==model and c['lanes']==32))
        case.update(id=f'fp32_{model}_lanes32_fixed',batch_shape_policy='fixed')
        cases.append(case)
    write_json(out/'manifest.json',dict(binary_sha256=build_fingerprint(),environment=environment(),cases=cases))
    for case in cases:
        record=out/(case['id']+'.json')
        if record.exists():
            previous=json.loads(record.read_text(encoding='utf-8'))
            if previous.get('status')=='passed' and previous.get('case')==case and previous.get('binary_sha256')==build_fingerprint():
                print(case['id'],'reuse',flush=True);continue
        file=out/(case['id']+'.case.json');write_json(file,case)
        print('Starting',case['id'],flush=True)
        try:
            with (out/(case['id']+'.log')).open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(file)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            if result.returncode:raise RuntimeError(f'Worker exit {result.returncode}')
            row=json.loads(record.read_text(encoding='utf-8'))
            if row['status']!='passed':raise RuntimeError('Request correctness failed')
        except Exception as error:
            write_json(out/(case['id']+'.failure.json'),dict(error=str(error)))
            raise
        print('Passed',case['id'],flush=True)


def optimize_models():return ('yolov8s','yolov8s-pose','yolov8s-seg')


if __name__=='__main__':main()
