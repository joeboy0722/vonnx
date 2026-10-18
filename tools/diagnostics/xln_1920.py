"""Short independent X/L/n resident-model comparison; never starts a full suite."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vonnx_tools.common import manifest, images, load_rgb, write_json, environment, build_fingerprint

MODELS = ("yolov8x", "yolov8l", "yolov8n")


def worker(args):
    import cv2
    import numpy as np
    import pynvml
    from vonnx import EngineConfig, vONNXEngine, PixelFormat
    import vonnx
    import vonnx.precision
    import hashlib
    binary_path=next(Path(vonnx.__file__).parent.glob('vonnx_py*.pyd'))
    from vonnx_tools.benchmark import sample_stats
    from vonnx_tools.validate import box_array

    if args.backend == "official_stream":
        # Diagnostic control only: preserve predictor computation and CPU outputs,
        # remove its timing-only device-wide barriers. Never label as stock official.
        from ultralytics.utils.ops import Profile
        Profile.time = lambda self: time.perf_counter()

    out = Path(args.output)
    frames = [np.ascontiguousarray(cv2.resize(a, (1920, 1080))) for a in load_rgb(images("detect"))]
    pynvml.nvmlInit()
    dev = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory = lambda: pynvml.nvmlDeviceGetMemoryInfo(dev).used
    result = dict(status="initializing", backend=args.backend, precision=args.precision,
                  imgsz=[1920, 1920], raw_shape=[1080, 1920, 3], batch=1, pipeline_depth=1,
                  wait_ms=0, collect_timings=args.profile, warmup_per_model=args.warmup,
                  binary_path=str(binary_path),binary_sha256=hashlib.sha256(binary_path.read_bytes()).hexdigest(),
                  preparation_tool_sha256=hashlib.sha256(Path(vonnx.precision.__file__).read_bytes()).hexdigest(),
                  graph_specialization=args.specialization, fixed_quota=args.fixed_quota,
                  cuda_graph=args.cuda_graph,
                  cuda_environment={key:os.environ.get(key) for key in ('CUDA_DEVICE_MAX_CONNECTIONS','CUDA_DEVICE_MAX_COPY_CONNECTIONS','CUDA_LAUNCH_BLOCKING')},
                  specialization_model=args.specialization_model,
                  request_cap_per_model_per_case=75, models=[], scenarios=[],
                  specialize_n=args.specialize_n,
                  environment=environment(), binary=build_fingerprint(), initial_bytes=memory(),
                  note="Solo keeps other models resident. Common window ends when first producer stops; tail excluded from common FPS.")
    engines = []
    peak = [memory()]
    gpu_samples=[]
    done = threading.Event()

    def monitor():
        while not done.wait(.02):
            peak[0] = max(peak[0], memory())
            gpu_samples.append(dict(time=time.perf_counter(),sm_mhz=pynvml.nvmlDeviceGetClockInfo(dev,pynvml.NVML_CLOCK_SM),
                memory_mhz=pynvml.nvmlDeviceGetClockInfo(dev,pynvml.NVML_CLOCK_MEM),
                temperature=pynvml.nvmlDeviceGetTemperature(dev,pynvml.NVML_TEMPERATURE_GPU),
                utilization=pynvml.nvmlDeviceGetUtilizationRates(dev).gpu))

    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()

    def save():
        result["peak_device_bytes"] = peak[0]
        write_json(out / "result.json", result)

    def submit(i, lane, frame):
        return engines[i].submit_numpy(lane, frame, frames[frame % len(frames)], PixelFormat.RGB24)

    def measure(indices, lanes, seconds, label):
        rows, errors, ended = [], [], []
        lock = threading.Lock()
        barrier = threading.Barrier(len(indices) * lanes + 1)
        issued = {i: 0 for i in indices}
        start = [0.]

        def run(i, lane):
            barrier.wait()
            while True:
                with lock:
                    if issued[i] >= 75 or (not args.fixed_quota and time.perf_counter() - start[0] >= seconds) or errors:
                        ended.append(time.perf_counter())
                        return
                    frame = issued[i]
                    issued[i] += 1
                begin = time.perf_counter()
                try:
                    future = submit(i, lane, frame)
                    submitted = time.perf_counter()
                    r = future.result(timeout=30)[0]
                    finish = time.perf_counter()
                    assert (r.stream_id, r.frame_id, r.batch_size) == (lane, frame, 1)
                    row = dict(model=MODELS[i], lane=lane, frame=frame,
                               start_s=begin-start[0], finish_s=finish-start[0],
                               latency_ms=(finish-begin)*1000, submit_ms=(submitted-begin)*1000,
                               boxes=len(r.boxes))
                    if args.profile:
                        row["diagnostic_trace"] = dict(getattr(r, "diagnostic_trace", {}))
                        row['gpu_inference_ms']=r.inference_ms
                        row['h2d_ms']=r.h2d_ms
                        row['d2h_ms']=r.d2h_ms
                    with lock:
                        rows.append(row)
                except Exception as e:
                    with lock:
                        errors.append(dict(model=MODELS[i], error=repr(e)))
                    return

        threads = [threading.Thread(target=run, args=(i, lane)) for i in indices for lane in range(lanes)]
        for t in threads:
            t.start()
        start[0] = time.perf_counter()
        barrier.wait()
        for t in threads:
            t.join()
        elapsed = time.perf_counter()-start[0]
        common = min(ended)-start[0] if ended else elapsed
        summaries = []
        for i in indices:
            selected = [r for r in rows if r["model"] == MODELS[i]]
            active = [r for r in selected if r["finish_s"] <= common]
            assert len({(r["lane"],r["frame"]) for r in selected}) == len(selected)
            summaries.append(dict(model=MODELS[i], success=len(selected), common_success=len(active),
                                  common_fps=len(active)/common, overall_fps=len(selected)/elapsed,
                                  latency=sample_stats([r["latency_ms"] for r in active]),
                                  submit=sample_stats([r["submit_ms"] for r in active])))
        scenario = dict(name=label, lanes_per_model=lanes, elapsed_s=elapsed, common_window_s=common,
                        gpu_samples=[r for r in gpu_samples if start[0]<=r['time']<=start[0]+elapsed],
                        per_model=summaries, common_total_fps=sum(r["common_fps"] for r in summaries),
                        status="failed" if errors else "passed", errors=errors,
                        count=len(rows), tail_count=sum(r["finish_s"]>common for r in rows))
        with (out/(label+".jsonl")).open("w",encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r)+"\n")
        result["scenarios"].append(scenario)
        save()
        print(json.dumps({k:v for k,v in scenario.items() if k!='gpu_samples'}), flush=True)
        if errors:
            raise RuntimeError(errors)

    try:
        for model in MODELS:
            print("Initialize", args.backend, args.precision, model, flush=True)
            save()
            begin = time.perf_counter()
            if args.backend in ("official", "official_stream"):
                from official_1920 import Official1920
                engine = Official1920(model, args.precision)
            else:
                cfg = EngineConfig()
                cfg.model_path = str(ROOT/manifest(model)["onnx"])
                if args.specialize_n and model == "yolov8n":
                    import onnx
                    graph = onnx.load(cfg.model_path)
                    for dim, size in zip(graph.graph.input[0].type.tensor_type.shape.dim,(1,3,1920,1920)):
                        dim.ClearField("dim_param")
                        dim.dim_value=size
                    target=out/"yolov8n_static_1920.onnx"
                    onnx.save(graph,target)
                    cfg.model_path=str(target.resolve())
                cfg.imgsz = 1920
                cfg.max_input_width, cfg.max_input_height = 1920, 1080
                cfg.max_batch_size, cfg.pipeline_depth = 1, 1
                cfg.batch_mode, cfg.max_latency_ms = "latency", 0
                cfg.concurrency = 32
                cfg.precision = args.precision
                if args.specialization!='off' and (args.specialization_model=='all' or model=='yolov8n'):
                    cfg.graph_specialization=args.specialization
                if args.cuda_graph:cfg.cuda_graph=True
                cfg.model_cache_dir = str(ROOT/"runs/fp16/cache")
                cfg.collect_timings = args.profile
                engine = vONNXEngine(cfg)
            engines.append(engine)
            result["models"].append(dict(model=model, manifest=manifest(model),
                                         init_seconds=time.perf_counter()-begin, device_bytes=memory()))
            if args.backend=='native':
                spec=engine.model_spec
                result['models'][-1]['prepared']=dict(path=spec.prepared_model_path,
                    sha256=hashlib.sha256(Path(spec.prepared_model_path).read_bytes()).hexdigest(),
                    fixed_batch=spec.fixed_batch_size,input_dtype=spec.input_dtype,output_dtypes=spec.output_dtypes)
            save()
        for i in range(3):
            for frame in range(args.warmup):
                submit(i,0,frame).result(timeout=30)
        alone = [submit(i,0,0).result(timeout=30)[0] for i in range(3)]
        futures = [submit(i,0,0) for i in range(3)]
        together = [f.result(timeout=30)[0] for f in futures]
        for i,(a,b) in enumerate(zip(alone,together)):
            np.testing.assert_allclose(box_array(a),box_array(b),atol=1e-4,rtol=1e-3)
            np.savez(out/(MODELS[i]+"_check.npz"),alone=box_array(a),concurrent=box_array(b))
        result["solo_concurrent_outputs_identical"] = True
        if not args.profile and not args.concurrent_only:
            for i in range(3):
                measure([i],1,3,"solo_"+MODELS[i])
        if args.profile:
            import ctypes
            runtime=ctypes.WinDLL("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64/cudart64_13.dll")
            assert runtime.cudaProfilerStart()==0
        for lanes in ((1,) if args.profile else (1,4)):
            measure(range(3),lanes,6,"concurrent_"+str(lanes))
        if args.profile:
            assert runtime.cudaProfilerStop()==0
        result["status"]="passed"
    except Exception as e:
        result.update(status="failed",error=repr(e))
        raise
    finally:
        result["inference_finished"] = True
        save()
        for engine in reversed(engines):
            engine.close()
        done.set()
        sampler.join()
        result["closed"] = True
        result["after_close_bytes"] = memory()
        save()
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=("native","official","official_stream"),required=True)
    p.add_argument("--precision",choices=("fp32","fp16"),required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--worker",action="store_true")
    p.add_argument("--profile",action="store_true")
    p.add_argument("--specialize-n",action="store_true",help="Experimental static-shape copy of n only; source stays unchanged")
    p.add_argument("--concurrent-only",action="store_true")
    p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--specialization",choices=('off','auto'),default='off')
    p.add_argument('--specialization-model',choices=('all','n'),default='all')
    p.add_argument('--cuda-graph',action='store_true')
    p.add_argument('--connections',type=int,choices=(1,8,16,32),help='Diagnostic child-process compute/copy work queues, set before CUDA initialization')
    p.add_argument("--fixed-quota",action='store_true')
    p.add_argument("--binary-root",type=Path)
    args=p.parse_args()
    if args.connections is not None:
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS']=str(args.connections)
        os.environ['CUDA_DEVICE_MAX_COPY_CONNECTIONS']=str(args.connections)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.worker:
        if args.binary_root:sys.path.insert(0,str(args.binary_root.resolve()))
        worker(args)
    else:
        with (out/"worker.log").open("w",encoding="utf-8") as log:
            try:
                run=subprocess.run([sys.executable,__file__,*sys.argv[1:],"--worker"],
                                   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                if run.returncode:
                    status=out/'result.json'
                    r=json.loads(status.read_text(encoding='utf-8')) if status.exists() else {}
                    r.update(status='failed',exitcode=run.returncode)
                    write_json(status,r)
                    raise SystemExit(run.returncode)
            except subprocess.TimeoutExpired:
                write_json(out/"timeout.json",dict(status="timeout",limit_seconds=300))
                raise
        print((out/"result.json").read_text(encoding="utf-8"))
