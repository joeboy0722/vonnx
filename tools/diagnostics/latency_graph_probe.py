"""Isolated ORT CUDA Graph feasibility on two resident slots, no graph rewriting."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vonnx_tools.common import manifest,write_json,sha256
parser=argparse.ArgumentParser();parser.add_argument('--precision',choices=['fp32','fp16'],required=True)
parser.add_argument('--specialize',action='store_true')
parser.add_argument('--output',type=Path,default=ROOT/'runs/latency_opt/graph_probe')
args=parser.parse_args();folder=args.output;folder.mkdir(parents=True,exist_ok=True)
import vonnx
import torch
import numpy as np
import onnxruntime as ort
from vonnx.precision import prepare
source=ROOT/manifest('yolov8s')['onnx']
path=prepare(source,ROOT/'runs/fp16/cache') if args.precision=='fp16' else source
rows=[]
for batch in (1,8):
    if args.precision=='fp32' and batch==8:continue # Trace gap 6.6%, below experiment threshold.
    for graph in (False,True):
        if args.specialize:path=prepare(source,ROOT/'runs/fp16/cache',shape=(640,640),fixed_batch=batch,convert_fp16=args.precision=='fp16')
        row=dict(precision=args.precision,batch=batch,graph=graph,model_sha256=sha256(path),specialized=args.specialize)
        try:
            stream=torch.cuda.Stream()
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
            options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
            session=ort.InferenceSession(str(path),options,providers=[('CUDAExecutionProvider',dict(
                prefer_nhwc=int(args.precision=='fp16'),enable_cuda_graph=int(graph),use_tf32=0,
                user_compute_stream=str(stream.cuda_stream),cudnn_conv_algo_search='HEURISTIC',gpu_mem_limit=3*1024**3))])
            session.disable_fallback()
            slots=[]
            for slot in range(2):
                x=torch.rand((batch,3,640,640),device='cuda',dtype=torch.float32)
                y=torch.empty((batch,84,8400),device='cuda',dtype=torch.float32)
                io=session.io_binding();io.bind_input(session.get_inputs()[0].name,'cuda',0,np.float32,tuple(x.shape),x.data_ptr())
                io.bind_output(session.get_outputs()[0].name,'cuda',0,np.float32,tuple(y.shape),y.data_ptr())
                run=ort.RunOptions()
                if graph:run.add_run_config_entry('gpu_graph_id',str(slot+1))
                torch.cuda.synchronize();session.run_with_iobinding(io,run)
                slots.append((x,y,io,run))
            for _ in range(25):
                for x,y,io,run in slots:session.run_with_iobinding(io,run)
            start=time.perf_counter()
            for i in range(75):
                x,y,io,run=slots[i%2];session.run_with_iobinding(io,run)
            stream.synchronize()
            row.update(status='passed',mean_batch_ms=(time.perf_counter()-start)*1000/75,
                       finite=all(torch.isfinite(s[1]).all().item() for s in slots))
            del slots,io,x,y,session
        except Exception as error:
            row.update(status='failed',error=str(error))
        rows.append(row);print(row,flush=True)
write_json(folder/(args.precision+'.json'),rows)
