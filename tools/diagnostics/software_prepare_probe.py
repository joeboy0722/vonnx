"""Five one-image preparation-order/EP-placement probes; not a speed benchmark."""
from pathlib import Path
import sys
import json
import gc
import collections
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import vonnx
from vonnx.precision import prepare
from vonnx_tools.common import manifest,images,load_rgb,write_json,sha256
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics.data.augment import LetterBox
out=ROOT/'runs/software_wait/preparation';out.mkdir(parents=True,exist_ok=True)
source=ROOT/manifest('yolov8n')['onnx'];cache=ROOT/'runs/fp16/cache'
image=np.ascontiguousarray(cv2.resize(load_rgb(images('detect'))[0],(1920,1080)))
image=LetterBox((1920,1920),auto=False)(image=image)
feed=np.ascontiguousarray(image.transpose(2,0,1)[None],dtype=np.float32)/255
results=[];references={}
for label,half,shape,order in [('fp32_dynamic',False,None,'before'),('fp32_shape',False,(1920,1920),'before'),
    ('fp16_dynamic',True,None,'before'),('fp16_shape_before',True,(1920,1920),'before'),('fp16_shape_after',True,(1920,1920),'after')]:
    path=source if shape is None and not half else prepare(source,cache,shape=shape,fixed_batch=1 if shape else 0,
                                                         convert_fp16=half,specialization_order=order)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
    options.enable_profiling=True;options.profile_file_prefix=str(out/label)
    session=ort.InferenceSession(str(path),options,providers=[('CUDAExecutionProvider',dict(use_tf32=False,
        prefer_nhwc=int(half),cudnn_conv_algo_search='HEURISTIC'))])
    actual=session.run(None,{session.get_inputs()[0].name:feed})[0]
    profile=Path(session.end_profiling());del session;gc.collect()
    events=json.loads(profile.read_text(encoding='utf-8'))
    cpu=[dict(name=e['name'],op=e['args'].get('op_name'),us=e['dur']) for e in events
         if e.get('cat')=='Node' and e.get('args',{}).get('provider')=='CPUExecutionProvider']
    ops=collections.Counter(e['args'].get('op_name') for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider')=='CUDAExecutionProvider')
    key='fp16' if half else 'fp32'
    if key not in references:references[key]=actual
    difference=float(np.abs(actual-references[key]).max())
    np.testing.assert_allclose(actual,references[key],atol=1e-3 if half else 1e-4,rtol=1e-2 if half else 1e-3)
    row=dict(case=label,model_sha256=sha256(path),path=str(path),cpu_nodes=cpu,cuda_ops=ops,
             max_raw_error_vs_dynamic=difference,profile_path=str(profile))
    results.append(row);print(label,'CPU nodes',len(cpu),'raw error',difference,flush=True)
    write_json(out/'summary.json',results)
write_json(out/'summary.json',results)
