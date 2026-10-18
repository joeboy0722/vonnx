"""Isolated TensorRT INT8/FP16 core feasibility, not a production vONNX backend."""
import argparse
import collections
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.diagnostics.int8_probe import paths
from vonnx_tools.common import manifest, load_rgb, sha256, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['fp16','int8','ort-fp16'],required=True)
    p.add_argument('--output',type=Path,default=ROOT/'runs/int8_detect')
    a=p.parse_args();out=a.output.resolve()
    import vonnx
    import torch
    import numpy as np
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(out/'trt_runtime'))
    import tensorrt_bindings as trt
    import onnx
    logger=trt.Logger(trt.Logger.WARNING)
    stream=torch.cuda.Stream()
    build_start=time.perf_counter()
    engine_path=out/f'trt_{a.kind}.engine'
    if a.kind!='ort-fp16':
        builder=trt.Builder(logger)
        network=builder.create_network(0)
        parser=trt.OnnxParser(network,logger)
        source=ROOT/manifest('yolov8s')['onnx'] if a.kind=='fp16' else out/'yolov8s-qdq.onnx'
        graph=onnx.load(source)
        if a.kind=='int8':
            # TRT explicit quantization uses float bias; remove ORT's INT32 bias Q/DQ.
            initializers={i.name:i for i in graph.graph.initializer}
            from onnx import numpy_helper
            remove=[]
            for node in graph.graph.node:
                if node.op_type=='DequantizeLinear' and node.input[0] in initializers:
                    weight=initializers[node.input[0]]
                    if weight.data_type==onnx.TensorProto.INT32:
                        value=numpy_helper.to_array(weight).astype(np.float32)
                        scale=numpy_helper.to_array(initializers[node.input[1]])
                        zero=numpy_helper.to_array(initializers[node.input[2]]) if len(node.input)>2 else 0
                        graph.graph.initializer.append(numpy_helper.from_array((value-zero)*scale,node.output[0]))
                        remove.append(node)
            for node in remove:graph.graph.node.remove(node)
            onnx.checker.check_model(graph)
        if not parser.parse(graph.SerializeToString()):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config=builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1024*1024*1024)
        config.builder_optimization_level=1
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_flag(trt.BuilderFlag.FP16)
        if a.kind=='int8':config.set_flag(trt.BuilderFlag.INT8)
        profile=builder.create_optimization_profile()
        name=network.get_input(0).name
        profile.set_shape(name,(1,3,640,640),(8,3,640,640),(8,3,640,640))
        config.add_optimization_profile(profile)
        if not engine_path.exists():
            built=builder.build_serialized_network(network,config)
            if built is None:raise RuntimeError('TensorRT build failed')
            engine_path.write_bytes(bytes(built))
        runtime=trt.Runtime(logger)
        engine=runtime.deserialize_cuda_engine(engine_path.read_bytes())
        context=engine.create_execution_context()
        input_name=next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                        if engine.get_tensor_mode(engine.get_tensor_name(i))==trt.TensorIOMode.INPUT)
        output_name=next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                         if engine.get_tensor_mode(engine.get_tensor_name(i))==trt.TensorIOMode.OUTPUT)
        context.set_input_shape(input_name,(8,3,640,640))
        inspector=engine.create_engine_inspector();inspector.execution_context=context
        layers=json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        write_json(out/f'trt_{a.kind}_layers.json',layers)
        def infer(x,y):
            context.set_input_shape(input_name,tuple(x.shape))
            context.set_tensor_address(input_name,x.data_ptr())
            context.set_tensor_address(output_name,y.data_ptr())
            if not context.execute_async_v3(stream.cuda_stream):raise RuntimeError('TRT enqueue failed')
        model_hash=sha256(engine_path)
    else:
        import onnxruntime as ort
        from vonnx.precision import prepare
        prepared=prepare(ROOT/manifest('yolov8s')['onnx'],ROOT/'runs/fp16/cache')
        options=ort.SessionOptions();options.log_severity_level=3;options.intra_op_num_threads=1
        session=ort.InferenceSession(str(prepared),options,providers=[('CUDAExecutionProvider',dict(
            use_tf32=False,prefer_nhwc=1,user_compute_stream=str(stream.cuda_stream),
            cudnn_conv_algo_search='HEURISTIC',gpu_mem_limit=2*1024**3))])
        bindings={}
        def infer(x,y):
            key=(tuple(x.shape),x.data_ptr(),y.data_ptr())
            if key not in bindings:
                io=session.io_binding()
                io.bind_input(session.get_inputs()[0].name,'cuda',0,np.float32,tuple(x.shape),x.data_ptr())
                io.bind_output(session.get_outputs()[0].name,'cuda',0,np.float32,tuple(y.shape),y.data_ptr())
                bindings[key]=io
            session.run_with_iobinding(bindings[key])
        model_hash=sha256(prepared)
    build_seconds=time.perf_counter()-build_start
    from ultralytics.data.augment import LetterBox
    eval_paths=paths()[1]
    photos=load_rgb(eval_paths)
    tensors=[np.ascontiguousarray(LetterBox((640,640),auto=False)(image=im).transpose(2,0,1),np.float32)/255. for im in photos]
    measured=[]
    # Identical 100 real images, GPU-resident tensors, CPU enqueue + GPU completion and GPU event intervals.
    for batch in (1,8):
        x=torch.empty((batch,3,640,640),device='cuda',dtype=torch.float32)
        y=torch.empty((batch,84,8400),device='cuda',dtype=torch.float32)
        gpu_batches=[]
        for offset in range(0,100,batch):
            v=np.zeros((batch,3,640,640),np.float32)
            for j,t in enumerate(tensors[offset:offset+batch]):v[j]=t
            gpu_batches.append(torch.from_numpy(v).to('cuda'))
        for _ in range(25):
            with torch.cuda.stream(stream):
                x.copy_(gpu_batches[0]);infer(x,y)
            stream.synchronize()
        rounds=[]
        for repeat in range(3):
            intervals=[];wall_start=time.perf_counter()
            for data in gpu_batches:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    x.copy_(data)
                    start.record(stream);infer(x,y);end.record(stream)
                end.synchronize()
                intervals.append(start.elapsed_time(end))
            wall=time.perf_counter()-wall_start
            rounds.append(dict(fps_core_wall=100/wall,fps_gpu_interval=100/(sum(intervals)/1000),
                               mean_batch_gpu_ms=float(np.mean(intervals))))
        measured.append(dict(batch=batch,actual_images=100,physical_images=len(gpu_batches)*batch,rounds=rounds))
        del gpu_batches
    # Small held-out AP, same postprocessing on CPU for all three backends.
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    # Synchronous prototype: raw preloaded RGB -> CPU boxes, same code for all backends.
    # This is distinct from the native engine's asynchronous queue/pipeline benchmark.
    prototype=[]
    for batch in (1,8):
        x=torch.empty((batch,3,640,640),device='cuda',dtype=torch.float32)
        y=torch.empty((batch,84,8400),device='cuda',dtype=torch.float32)
        rounds=[]
        for repeat in range(3):
            latencies=[];start_all=time.perf_counter()
            for offset in range(0,100,batch):
                begin=time.perf_counter()
                selected=photos[offset:offset+batch]
                host=np.zeros((batch,3,640,640),np.float32)
                for j,im in enumerate(selected):
                    host[j]=LetterBox((640,640),auto=False)(image=im).transpose(2,0,1)/np.float32(255.)
                with torch.cuda.stream(stream):
                    x.copy_(torch.from_numpy(host),non_blocking=False);infer(x,y)
                    boxes=non_max_suppression(y,conf_thres=.25,iou_thres=.7,nc=80,max_det=300)
                    for im,b in zip(selected,boxes):
                        scale_boxes((640,640),b[:,:4],im.shape[:2])
                        b.cpu().numpy()
                stream.synchronize()
                latencies.append((time.perf_counter()-begin)*1000)
            rounds.append(dict(fps=100/(time.perf_counter()-start_all),
                mean_batch_latency_ms=float(np.mean(latencies)),p95_batch_latency_ms=float(np.percentile(latencies,95))))
        prototype.append(dict(batch=batch,rounds=rounds))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO(str(ROOT/'datasets/coco/annotations/instances_val2017.json'))
    categories=[c['id'] for c in gt.dataset['categories']]
    records=[]
    x=torch.empty((1,3,640,640),device='cuda',dtype=torch.float32)
    y=torch.empty((1,84,8400),device='cuda',dtype=torch.float32)
    # ORT bindings hold pointers, so reset when replacing the batch-1 buffers.
    if a.kind=='ort-fp16':bindings.clear()
    for path,image,tensor in zip(eval_paths,photos,tensors):
        with torch.cuda.stream(stream):
            x.copy_(torch.from_numpy(tensor[None]),non_blocking=False);infer(x,y)
        stream.synchronize()
        if not torch.isfinite(y).all().item():raise ValueError('Nonfinite output')
        boxes=non_max_suppression(y.cpu(),conf_thres=.001,iou_thres=.7,nc=80,max_det=300)[0]
        scale_boxes((640,640),boxes[:,:4],image.shape[:2])
        for box in boxes.numpy():
            records.append(dict(image_id=int(path.stem),category_id=categories[int(box[5])],
                bbox=[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])],score=float(box[4])))
    evaluation=COCOeval(gt,gt.loadRes(records),'bbox')
    evaluation.params.imgIds=[int(p.stem) for p in eval_paths]
    evaluation.evaluate();evaluation.accumulate();evaluation.summarize()
    write_json(out/f'trt_{a.kind}_predictions.json',records)
    write_json(out/f'trt_{a.kind}.json',dict(kind=a.kind,trt_version=trt.__version__,
        builder_optimization_level=1,workspace_mb=1024,build_seconds=build_seconds,model_sha256=model_hash,
        measured=measured,prototype=prototype,ap50_95=float(evaluation.stats[0]),ap_images=100,
        timing_scope='GPU-resident core; excludes preprocessing/H2D/D2H/NMS, not end-to-end vONNX FPS'))


if __name__=='__main__':main()
