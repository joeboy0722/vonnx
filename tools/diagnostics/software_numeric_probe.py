"""One-image X FP16 coordinate diagnosis at 1920; not an AP evaluation."""
from pathlib import Path
import sys
import json
import gc
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vonnx_tools.common import manifest,images,load_rgb,write_json,sha256
import vonnx
import cv2
import numpy as np
import onnxruntime as ort

out=ROOT/'runs/software_wait/numeric';out.mkdir(parents=True,exist_ok=True)
image=np.ascontiguousarray(cv2.resize(load_rgb(images('detect'))[0],(1920,1080)))
cfg=vonnx.EngineConfig();cfg.model_path=str(ROOT/manifest('yolov8x')['onnx'])
cfg.precision='fp16';cfg.graph_specialization='auto';cfg.imgsz=1920
cfg.max_batch_size=1;cfg.batch_mode='latency';cfg.pipeline_depth=1;cfg.concurrency=1
cfg.model_cache_dir=str(ROOT/'runs/fp16/cache');cfg.capture_debug=True
engine=vonnx.vONNXEngine(cfg)
spec=engine.model_spec;path=spec.prepared_model_path
r=engine.submit_numpy(0,0,image,vonnx.PixelFormat.RGB24).result(timeout=30)[0]
feed=np.array(r.debug_input,np.float32).reshape(1,3,1920,1920)
raw=np.array(r.debug_outputs[0],np.float32).reshape(1,84,-1)
own=np.array([[b.x1,b.y1,b.x2,b.y2,b.score,b.class_id] for b in r.boxes],np.float32)
engine.close();del engine;gc.collect()

options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
def run(path,nhwc):
    session=ort.InferenceSession(str(path),options,providers=[('CUDAExecutionProvider',dict(
        use_tf32=False,prefer_nhwc=nhwc,cudnn_conv_algo_search='HEURISTIC'))])
    result=session.run(None,{session.get_inputs()[0].name:feed})[0]
    del session;gc.collect();return result
prepared=run(path,1)
np.testing.assert_allclose(raw,prepared,atol=1e-3,rtol=1e-2)
fp32=run(ROOT/manifest('yolov8x')['onnx'],0)

import torch
from ultralytics import YOLO
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=YOLO(str(ROOT/manifest('yolov8x')['weight']));captured=[]
handle=model.model.model[-1].register_forward_hook(lambda module,args,output:captured.append((output[0] if isinstance(output,tuple) else output).detach().clone()))
official=model.predict(np.ascontiguousarray(image[:,:,::-1]),imgsz=1920,rect=False,half=True,quantize=16,
                       conf=.25,iou=.7,max_det=300,verbose=False)[0]
handle.remove();predictor=model.predictor
assert predictor.model.fp16
stock=official.boxes.data.cpu().numpy().astype(np.float32)
input_stub=torch.empty((1,3,1920,1920),device='cuda',dtype=torch.float32)
def post(prediction):
    return predictor.postprocess(prediction,input_stub,[np.ascontiguousarray(image[:,:,::-1])])[0].boxes.data.cpu().numpy().astype(np.float32)
official_float_post=post(captured[-1].float())
same_onnx_post=post(torch.from_numpy(prepared).cuda().float())
source_float_post=post(torch.from_numpy(fp32).cuda().float())

def comparison(a,b):
    used=set();deltas=[];score=[]
    for box in a:
        candidates=[j for j,w in enumerate(b) if j not in used and int(w[5])==int(box[5])]
        if not candidates:return dict(status='unmatched',count_a=len(a),count_b=len(b))
        j=min(candidates,key=lambda j:np.abs(b[j,:4]-box[:4]).max());used.add(j)
        deltas.append(float(np.abs(box[:4]-b[j,:4]).max()));score.append(float(abs(box[4]-b[j,4])))
    return dict(status='matched' if len(used)==len(b) else 'unmatched',max_coordinate_px=max(deltas,default=0),
                max_score_delta=max(score,default=0),count_a=len(a),count_b=len(b))
result=dict(status='passed',model='yolov8x',imgsz=[1920,1920],images=1,ap_validated=False,
    binary_sha256=sha256(next(Path(vonnx.__file__).parent.glob('*.pyd'))),official_raw_dtype=str(captured[-1].dtype),
    source=manifest('yolov8x'),prepared_sha256=sha256(path),
    independent_prepared_raw_max_error=float(np.abs(raw-prepared).max()),
    native_vs_same_onnx=comparison(own,same_onnx_post),native_vs_official_half=comparison(own,stock),
    native_vs_official_half_float_post=comparison(own,official_float_post),
    native_vs_source_fp32=comparison(own,source_float_post),
    official_half_vs_source_fp32=comparison(stock,source_float_post),
    official_boxes_dtype=str(official.boxes.data.dtype),
    boxes=dict(native=own.tolist(),official_half=stock.tolist(),official_float_post=official_float_post.tolist(),source_fp32=source_float_post.tolist()))
write_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('source','boxes')},indent=2))
