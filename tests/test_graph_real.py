"""Small real-model Graph replay/slot/lifecycle checks, no AP or speed loop."""
import unittest
import numpy as np
from vonnx import EngineConfig,vONNXEngine,PixelFormat
from vonnx_tools.common import ROOT,manifest,images,load_rgb
from vonnx_tools.validate import box_array,mask_array

class RealGraphTests(unittest.TestCase):
    def test_c_api_v4_graph_external_gpu_producer(self):
        import ctypes as C
        import sys
        import torch
        import vonnx.precision
        from test_c_api import ConfigV2,Image,Result
        if not (ROOT/'models/yolov8s/yolov8s.onnx').exists() or not images('detect'):
            self.skipTest('Prepared s model and COCO8 images required')
        class V3(C.Structure):
            _fields_=[('version',C.c_uint32),('struct_size',C.c_uint32),('base',ConfigV2),
                      ('precision',C.c_int),('batch_shape_policy',C.c_int)]+[(k,C.c_char_p) for k in
                      ('model_cache_dir','precision_python','precision_tool')]
        class V4(C.Structure):
            _fields_=[('version',C.c_uint32),('struct_size',C.c_uint32),('base',V3),
                      ('graph_specialization',C.c_int),('cuda_graph',C.c_int)]
        dll=C.CDLL(str(ROOT/'vonnx_standalone_sdk/vonnx.dll'))
        dll.vonnx_config_init_v4.argtypes=[C.POINTER(V4)]
        dll.vonnx_create_engine_v4.argtypes=[C.POINTER(V4)];dll.vonnx_create_engine_v4.restype=C.c_void_p
        dll.vonnx_predict_v1.argtypes=[C.c_void_p,C.POINTER(Image),C.POINTER(C.POINTER(Result))]
        dll.vonnx_result_free.argtypes=[C.POINTER(Result)];dll.vonnx_destroy_engine.argtypes=[C.c_void_p]
        dll.vonnx_last_error.restype=C.c_char_p
        rgb=load_rgb(images('detect'))[0]
        cfg=V4();dll.vonnx_config_init_v4(C.byref(cfg));cfg.graph_specialization=cfg.cuda_graph=1
        cfg.base.precision=2;cfg.base.base.base.model_path=str(ROOT/manifest('yolov8s')['onnx']).encode('utf-8')
        cfg.base.base.base.input_height=cfg.base.base.base.input_width=640;cfg.base.base.base.max_batch_size=1
        cfg.base.base.base.max_input_height,cfg.base.base.base.max_input_width=rgb.shape[:2]
        cfg.base.precision_python=sys.executable.encode('utf-8');cfg.base.precision_tool=vonnx.precision.__file__.encode('utf-8')
        cfg.base.model_cache_dir=str(ROOT/'runs/fp16/cache').encode('utf-8')
        handle=dll.vonnx_create_engine_v4(C.byref(cfg));self.assertTrue(handle,dll.vonnx_last_error())
        def infer(desc):
            out=C.POINTER(Result)()
            self.assertEqual(dll.vonnx_predict_v1(handle,C.byref(desc),C.byref(out)),0,dll.vonnx_last_error())
            try:
                self.assertEqual(out.contents.frame_id,desc.frame_id)
                return np.array([[b.x1,b.y1,b.x2,b.y2,b.score,b.class_id]
                                 for b in out.contents.boxes[:out.contents.count]],np.float32).reshape(-1,6)
            finally:dll.vonnx_result_free(out)
        try:
            desc=Image(1,C.sizeof(Image),rgb.ctypes.data,rgb.nbytes,rgb.shape[1],rgb.shape[0],rgb.strides[0],1,0,0,4,10)
            expected=infer(desc);self.assertGreater(len(expected),0)
            gpu=torch.zeros(rgb.shape,dtype=torch.uint8,device='cuda');torch.cuda.current_stream().synchronize()
            producer=torch.cuda.Stream()
            with torch.cuda.stream(producer):
                if hasattr(torch.cuda,'_sleep'):torch.cuda._sleep(2_000_000)
                gpu.copy_(torch.from_numpy(rgb),non_blocking=True)
            desc.data=gpu.data_ptr();desc.is_gpu=1;desc.user_stream=producer.cuda_stream;desc.frame_id=11
            np.testing.assert_allclose(infer(desc),expected,atol=1e-4,rtol=0)
        finally:dll.vonnx_destroy_engine(handle)

    def test_tasks_replay_different_inputs_close_inflight_and_empty(self):
        for name,task in [('yolov8s','detect'),('yolov8s-pose','pose'),('yolov8s-seg','segment')]:
            if not (ROOT/'models'/name/(name+'.onnx')).exists() or not images(task):
                self.skipTest('Prepared s models and COCO8 images required')
            with self.subTest(model=name):
                frames=load_rgb(images(task))[:2]
                def config(graph,conf=.25):
                    c=EngineConfig();c.model_path=str(ROOT/manifest(name)['onnx'])
                    c.precision='fp16';c.graph_specialization='auto';c.cuda_graph=graph
                    c.model_cache_dir=str(ROOT/'runs/fp16/cache');c.imgsz=640
                    c.max_batch_size=4;c.batch_shape_policy='fixed';c.pipeline_depth=2
                    c.concurrency=16;c.conf_threshold=conf
                    c.max_input_height=max(a.shape[0] for a in frames);c.max_input_width=max(a.shape[1] for a in frames)
                    return c
                def submit(e,count=8):
                    return [e.submit_numpy(i%4,100+i,frames[i%2],PixelFormat.RGB24) for i in range(count)]
                baseline=vONNXEngine(config(False))
                try:reference=[f.result(timeout=30)[0] for f in submit(baseline)]
                finally:baseline.close()
                e=vONNXEngine(config(True))
                try:
                    self.assertTrue(e.model_spec.cuda_graph)
                    futures=submit(e)
                    actual=[f.result(timeout=30)[0] for f in futures]
                    before=e.resource_stats;e.pause();e.resume()
                    self.assertEqual(e.resource_stats.initialization_count,1)
                    self.assertEqual(e.resource_stats.explicit_device_allocations,before.explicit_device_allocations)
                    for frame,(a,b) in enumerate(zip(actual,reference)):
                        self.assertEqual((a.stream_id,a.frame_id),(b.stream_id,b.frame_id))
                        aa,bb=box_array(a),box_array(b)
                        self.assertEqual(len(aa),len(bb))
                        np.testing.assert_array_equal(aa[:,5],bb[:,5])
                        np.testing.assert_allclose(aa[:,:4],bb[:,:4],atol=1,rtol=0)
                        np.testing.assert_allclose(aa[:,4],bb[:,4],atol=1e-3,rtol=0)
                        for ka,kb in zip(a.keypoints,b.keypoints):
                            np.testing.assert_allclose([[p.x,p.y] for p in ka],[[p.x,p.y] for p in kb],atol=1,rtol=0)
                            np.testing.assert_allclose([p.confidence for p in ka],[p.confidence for p in kb],atol=1e-3,rtol=0)
                        for ma,mb in zip(a.masks,b.masks):
                            am=mask_array(ma,frames[frame%2].shape[:2])>0;bm=mask_array(mb,frames[frame%2].shape[:2])>0
                            union=np.logical_or(am,bm).sum()
                            self.assertGreaterEqual(np.logical_and(am,bm).sum()/union if union else 1,.99)
                    pending=submit(e,4);e.close()
                    self.assertEqual([f.result(timeout=5)[0].frame_id for f in pending],list(range(100,104)))
                    with self.assertRaises(RuntimeError):submit(e,1)
                finally:e.close()
                empty=vONNXEngine(config(True,1.0))
                try:
                    self.assertTrue(all(not f.result(timeout=30)[0].boxes for f in submit(empty,2)))
                finally:empty.close()

if __name__=='__main__':unittest.main()
