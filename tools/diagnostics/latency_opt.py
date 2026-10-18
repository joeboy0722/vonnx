"""Bounded latency optimization A/B; never starts AP or long benchmarks."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vonnx_tools.common import write_json,sha256,manifest,environment
OUT=ROOT/'runs/latency_opt'


def worker(case,folder):
    if case['variant']=='baseline':sys.path.insert(0,str(OUT/'baseline'))
    import vonnx
    from vonnx_tools import optimize
    import pynvml
    original_stats=optimize.stats
    samples=[];stop=threading.Event();started=[False]
    dll=ctypes.WinDLL('cudart64_13.dll') if case.get('profile') else None
    def sample_gpu():
        dev=pynvml.nvmlDeviceGetHandleByIndex(0)
        while not stop.wait(.05):
            samples.append(dict(time=time.perf_counter(),sm_mhz=pynvml.nvmlDeviceGetClockInfo(dev,pynvml.NVML_CLOCK_SM),
                memory_mhz=pynvml.nvmlDeviceGetClockInfo(dev,pynvml.NVML_CLOCK_MEM),
                temperature=pynvml.nvmlDeviceGetTemperature(dev,pynvml.NVML_TEMPERATURE_GPU),
                gpu_util=pynvml.nvmlDeviceGetUtilizationRates(dev).gpu,
                memory_used=pynvml.nvmlDeviceGetMemoryInfo(dev).used))
    thread=[None]
    def stats(engine):
        if not started[0]:
            started[0]=True
            thread[0]=threading.Thread(target=sample_gpu,daemon=True);thread[0].start()
            if dll and dll.cudaProfilerStart()!=0:raise RuntimeError('cudaProfilerStart failed')
        else:
            if dll:dll.cudaProfilerStop()
            stop.set()
            if thread[0]:thread[0].join()
        return original_stats(engine)
    optimize.stats=stats
    row=optimize.run(case,folder)
    actual=next(Path(vonnx.__file__).parent.glob('vonnx_py*.pyd'))
    row['binary_sha256']=sha256(actual)
    row['gpu_samples']=samples
    row['timing_schema_version']=2
    # New timestamps are additive diagnostics; legacy public intervals retain their meaning.
    # optimize.run records any attached trace dictionaries directly (new binaries only).
    write_json(folder/(case['id']+'.json'),row)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['baseline','candidate'],default='candidate')
    parser.add_argument('--phase',choices=['quick','confirm','diagnostic'],default='quick')
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--case')
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.worker:
        case=json.loads(args.worker.read_text(encoding='utf-8'))
        worker(case,OUT/case['variant']/case['phase']);return
    if args.phase=='confirm':
        # No automatic confirmation on a promising outlier in just one setting.
        # All four primary quick cases must improve before spending the 3-repeat budget.
        for precision in ('fp32','fp16'):
            for lanes in (16,32):
                name=f'{precision}_s{lanes}_r1.json'
                before=json.loads((OUT/'baseline/quick'/name).read_text(encoding='utf-8'))
                after=json.loads((OUT/'candidate/quick'/name).read_text(encoding='utf-8'))
                if not (after['latency']['mean_ms']<before['latency']['mean_ms'] and
                        after['latency']['p95_ms']<before['latency']['p95_ms'] and after['fps']>=.97*before['fps']):
                    raise RuntimeError('Quick gate not met for all four cases; confirmation not started')
    if not (OUT/'baseline/vonnx').exists():
        if args.variant!='baseline':raise RuntimeError('Capture v0.4.0 baseline first')
        dest=OUT/'baseline/vonnx';dest.mkdir(parents=True)
        for file in (ROOT/'vonnx').iterdir():
            if file.suffix in ('.py','.pyd'):shutil.copy2(file,dest/file.name)
        shutil.copy2(ROOT/'vonnx_standalone_sdk/vonnx.dll',OUT/'baseline/vonnx.dll')
        write_json(OUT/'baseline/provenance.json',dict(binary_sha256=sha256(next(dest.glob('*.pyd'))),
            dll_sha256=sha256(OUT/'baseline/vonnx.dll'),environment=environment()))
    folder=OUT/args.variant/args.phase;folder.mkdir(parents=True,exist_ok=True)
    for precision in ('fp32','fp16'):
        for lanes in ((1,16,32) if args.phase!='confirm' else (16,32)):
            for repeat in range(3 if args.phase=='confirm' else 1):
                name=f'{precision}_s{lanes}_r{repeat+1}'
                if args.case and args.case!=name:continue
                case=dict(id=name,model='yolov8s',precision=precision,variant=args.variant,phase=args.phase,
                    lanes=lanes,backend='native',depth=1 if lanes==1 else 2,wait=0 if lanes==1 else 2,
                    batch=1 if lanes==1 else 8,total=500 if args.phase=='confirm' else 100,repeat=repeat+1,
                    timings=args.phase=='diagnostic',batch_shape_policy='fixed',profile=args.profile,
                    imgsz=[640,640],conf=.25,iou=.7,max_det=300,
                    source_model_sha256=manifest('yolov8s')['onnx_sha256'])
                config=folder/(name+'.case.json');write_json(config,case)
                print(args.variant,args.phase,name,'starting',flush=True)
                with (folder/(name+'.log')).open('w',encoding='utf-8') as log:
                    run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(config)],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                if run.returncode:raise RuntimeError(f'{name} worker exit {run.returncode}; inspect log')
                row=json.loads((folder/(name+'.json')).read_text(encoding='utf-8'))
                if row['status']!='passed':raise RuntimeError(f'{name}: failed request gate')
                print(name,round(row['fps'],2),'FPS',flush=True)


if __name__=='__main__':main()
