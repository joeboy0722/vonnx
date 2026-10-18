"""Small same-ONNX correctness matrix, no COCO AP or performance matrix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vonnx_tools.common import write_json,build_fingerprint
parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path)
args=parser.parse_args();out=ROOT/'runs/latency_opt/functional';out.mkdir(parents=True,exist_ok=True)
if args.worker:
    from vonnx_tools.precision_check import smoke
    case=json.loads(args.worker.read_text(encoding='utf-8'))
    detail=smoke(case)
    write_json(out/(case['id']+'.json'),dict(case=case,status='passed',detail=detail,binary_sha256=build_fingerprint()))
else:
    cases=[]
    for precision in ('fp32','fp16'):
        for shape in ([640,640],[384,640]):
            cases.append(dict(model='yolov8s',precision=precision,imgsz=shape,batch_limits=[1,2,4,8]))
        for suffix in ('pose','seg'):
            cases.append(dict(model='yolov8s-'+suffix,precision=precision,imgsz=[640,640],batch_limits=[1,4],image_limit=2))
    rows=[]
    for case in cases:
        case['id']=case['model']+'_'+case['precision']+'_'+str(case['imgsz'][0])
        file=out/(case['id']+'.case.json');write_json(file,case)
        print('Correctness',case['id'],flush=True)
        with (out/(case['id']+'.log')).open('w',encoding='utf-8') as log:
            run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(file)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if run.returncode:raise RuntimeError(f"Correctness failed: {case['id']} (exit {run.returncode})")
        rows.append(json.loads((out/(case['id']+'.json')).read_text(encoding='utf-8')))
    write_json(out/'summary.json',rows)
