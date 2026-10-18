"""Unreleased opt-in shape preparation. Run with model.onnx image.jpg."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from vonnx import EngineConfig,vONNXEngine,PixelFormat

p=argparse.ArgumentParser()
p.add_argument('model');p.add_argument('image')
p.add_argument('--precision',choices=('auto','fp32','fp16'),default='auto')
p.add_argument('--imgsz',type=int,default=640)
p.add_argument('--cuda-graph',action='store_true')
p.add_argument('--cache-dir',default=str(Path.home()/'.cache/vonnx/models'))
args=p.parse_args()
bgr=cv2.imdecode(np.fromfile(args.image,np.uint8),cv2.IMREAD_COLOR)
if bgr is None:raise ValueError('Cannot decode image')
rgb=np.ascontiguousarray(bgr[:,:,::-1])
cfg=EngineConfig();cfg.model_path=args.model;cfg.precision=args.precision
cfg.graph_specialization='auto';cfg.cuda_graph=args.cuda_graph;cfg.model_cache_dir=args.cache_dir
cfg.imgsz=args.imgsz;cfg.max_batch_size=1;cfg.batch_mode='latency';cfg.pipeline_depth=1
cfg.max_input_height,cfg.max_input_width=rgb.shape[:2]
engine=vONNXEngine(cfg)
try:
    result=engine.submit_numpy(0,1,rgb,PixelFormat.RGB24).result(timeout=30)[0]
    print('boxes:',len(result.boxes),'source:',engine.model_spec.source_input_shape,
          'execution:',engine.model_spec.execution_input_shape,'graph:',engine.model_spec.cuda_graph)
    engine.pause();engine.resume()  # Same session, buffers and prepared model.
finally:engine.close()
