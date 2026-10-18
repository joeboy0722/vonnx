"""Opt-in shape preparation, fixed/dynamic batches and independent engines."""
import concurrent.futures
import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
import numpy as np
import onnx
from test_native import fixture
from vonnx import EngineConfig, vONNXEngine, PixelFormat
from vonnx.precision import prepare


class SpecializationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.path=self.root/'source.onnx';fixture(self.path)
    def tearDown(self):self.temp.cleanup()
    def config(self,**kwargs):
        c=EngineConfig();c.model_path=str(self.path);c.model_cache_dir=str(self.root/'cache')
        c.imgsz=(32,64);c.max_input_height=32;c.max_input_width=64
        c.max_batch_size=4;c.graph_specialization='auto';c.iou_threshold=.5
        for k,v in kwargs.items():setattr(c,k,v)
        return c
    def test_static_dynamic_and_task_matrix(self):
        for task in ('detect','pose','segment'):
            fixture(self.path,task)
            for precision,policy,batch in [('fp32','dynamic',0),('fp32','fixed',4),('fp16','auto',4)]:
                e=vONNXEngine(self.config(precision=precision,batch_shape_policy=policy))
                try:
                    s=e.model_spec
                    self.assertTrue(s.graph_specialized)
                    self.assertEqual(s.source_input_shape,[-1,3,-1,-1])
                    self.assertEqual(s.execution_input_shape,[batch or -1,3,32,64])
                    self.assertEqual(s.prepared_model_sha256,hashlib.sha256(Path(s.prepared_model_path).read_bytes()).hexdigest())
                    image=np.zeros((32,64,3),np.uint8)
                    fs=[e.submit_numpy(i,90+i,image,PixelFormat.RGB24) for i in range(7)]
                    results=[f.result(timeout=10)[0] for f in fs]
                    self.assertEqual([r.frame_id for r in results],list(range(90,97)))
                    self.assertTrue(all(len(r.boxes)==2 for r in results))
                    if batch:self.assertTrue(all(r.physical_batch_size==batch for r in results))
                    if task=='pose':self.assertTrue(all(len(r.keypoints)==2 for r in results))
                    if task=='segment':self.assertTrue(all(len(r.masks)==2 for r in results))
                    before=e.resource_stats;e.pause();e.resume()
                    self.assertEqual(e.resource_stats.initialization_count,1)
                    self.assertEqual(e.resource_stats.explicit_device_allocations,before.explicit_device_allocations)
                finally:e.close()
    def test_cache_and_source_integrity(self):
        original=self.path.read_bytes();cache=self.root/'cache'
        def work(_):return prepare(self.path,cache,shape=(32,64),fixed_batch=4,convert_fp16=False)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            paths=list(pool.map(work,range(2)))
        self.assertEqual(paths[0],paths[1]);self.assertEqual(self.path.read_bytes(),original)
        modified=paths[0].stat().st_mtime_ns;work(0)
        self.assertEqual(modified,paths[0].stat().st_mtime_ns)
        paths[0].write_bytes(b'corrupt');onnx.checker.check_model(str(work(0)))
        self.assertNotEqual(work(0),prepare(self.path,cache,shape=(64,64),fixed_batch=4,convert_fp16=False))
        self.assertNotEqual(work(0),prepare(self.path,cache,shape=(32,64),fixed_batch=2,convert_fp16=False))
    def test_native_half_io_and_static_rejection(self):
        from onnxruntime.transformers.float16 import convert_float_to_float16
        from onnxruntime.transformers.onnx_model import OnnxModel
        model=convert_float_to_float16(onnx.load(self.path),keep_io_types=False)
        OnnxModel(model).topological_sort();half=self.root/'half.onnx';onnx.save(model,half)
        e=vONNXEngine(self.config(model_path=str(half),precision='auto',max_batch_size=1))
        try:
            self.assertTrue(e.model_spec.graph_specialized)
            self.assertEqual(e.model_spec.input_dtype,10)
            self.assertEqual(e.submit_numpy(1,3,np.zeros((32,64,3),np.uint8),PixelFormat.RGB24).result(timeout=5)[0].frame_id,3)
        finally:e.close()
        fixture(self.path,static=True)
        with self.assertRaisesRegex((ValueError,RuntimeError),'static ONNX shape'):
            vONNXEngine(self.config(max_batch_size=1))
        with self.assertRaisesRegex((ValueError,RuntimeError),'static batch'):
            vONNXEngine(self.config(imgsz=32,batch_shape_policy='fixed'))
        with self.assertRaisesRegex(ValueError,'graph_specialization'):
            vONNXEngine(self.config(graph_specialization='bad'))
        with self.assertRaisesRegex(ValueError,'requires'):
            vONNXEngine(self.config(cuda_graph=True,graph_specialization='off'))
        with self.assertRaisesRegex(ValueError,'fixed execution batch'):
            vONNXEngine(self.config(cuda_graph=True,batch_shape_policy='dynamic',max_batch_size=4))
    def test_independent_host_delay_and_pause(self):
        x=vONNXEngine(self.config(max_batch_size=1))
        n=vONNXEngine(self.config(max_batch_size=1))
        entered=threading.Event();release=threading.Event();errors=[]
        image=np.zeros((32,64,3),np.uint8)
        def delayed_x():
            try:
                entered.set();release.wait(5)
                x.submit_numpy(0,1,image,PixelFormat.RGB24).result(timeout=5)
            except Exception as error:errors.append(error)
        t=threading.Thread(target=delayed_x);t.start()
        try:
            self.assertTrue(entered.wait(1));x.pause()
            # X has not submitted and is paused; N must complete independently.
            r=n.submit_numpy(2,7,image,PixelFormat.RGB24).result(timeout=2)[0]
            self.assertEqual((r.stream_id,r.frame_id),(2,7));self.assertFalse(release.is_set())
            x.resume();release.set();t.join(5);self.assertFalse(t.is_alive());self.assertFalse(errors)
        finally:release.set();t.join(5);x.close();n.close()
    def test_c_api_v4_fixed_batch_and_query(self):
        import ctypes as C
        import sys
        import vonnx.precision
        from test_c_api import ConfigV2,Image,Result
        from vonnx_tools.common import ROOT
        class V3(C.Structure):
            _fields_=[('version',C.c_uint32),('struct_size',C.c_uint32),('base',ConfigV2),
                      ('precision',C.c_int),('batch_shape_policy',C.c_int)]+[(k,C.c_char_p) for k in
                      ('model_cache_dir','precision_python','precision_tool')]
        class V4(C.Structure):
            _fields_=[('version',C.c_uint32),('struct_size',C.c_uint32),('base',V3),('graph_specialization',C.c_int),('cuda_graph',C.c_int)]
        class Info(C.Structure):
            _fields_=[('version',C.c_uint32),('struct_size',C.c_uint32),('graph_specialized',C.c_int),
                      ('source_input_shape',C.c_int64*4),('execution_input_shape',C.c_int64*4),
                      ('source_model_sha256',C.c_char*65),('prepared_model_sha256',C.c_char*65),('cuda_graph',C.c_int)]
        dll=C.CDLL(str(ROOT/'vonnx_standalone_sdk/vonnx.dll'))
        dll.vonnx_config_init_v4.argtypes=[C.POINTER(V4)]
        dll.vonnx_create_engine_v4.argtypes=[C.POINTER(V4)];dll.vonnx_create_engine_v4.restype=C.c_void_p
        dll.vonnx_model_info_get_v4.argtypes=[C.c_void_p,C.POINTER(Info)]
        dll.vonnx_destroy_engine.argtypes=[C.c_void_p];dll.vonnx_last_error.restype=C.c_char_p
        dll.vonnx_predict_v1.argtypes=[C.c_void_p,C.POINTER(Image),C.POINTER(C.POINTER(Result))]
        dll.vonnx_result_free.argtypes=[C.POINTER(Result)]
        cfg=V4();dll.vonnx_config_init_v4(C.byref(cfg));self.assertEqual(cfg.graph_specialization,0)
        cfg.graph_specialization=1;cfg.base.precision=2
        cfg.base.base.base.model_path=str(self.path).encode('utf-8')
        cfg.base.base.base.input_height=cfg.base.base.base.max_input_height=32
        cfg.base.base.base.input_width=cfg.base.base.base.max_input_width=64
        cfg.base.base.base.max_batch_size=4
        cfg.base.model_cache_dir=str(self.root/'cache').encode('utf-8')
        cfg.base.precision_python=sys.executable.encode('utf-8');cfg.base.precision_tool=vonnx.precision.__file__.encode('utf-8')
        handle=dll.vonnx_create_engine_v4(C.byref(cfg));self.assertTrue(handle,dll.vonnx_last_error())
        try:
            info=Info(4,C.sizeof(Info));self.assertEqual(dll.vonnx_model_info_get_v4(handle,C.byref(info)),0)
            self.assertEqual(list(info.source_input_shape),[-1,3,-1,-1])
            self.assertEqual(list(info.execution_input_shape),[4,3,32,64]);self.assertEqual(len(info.prepared_model_sha256),64)
            image=np.zeros((32,64,3),np.uint8)
            desc=Image(1,C.sizeof(Image),image.ctypes.data,image.nbytes,64,32,192,1,0,0,7,11)
            output=C.POINTER(Result)()
            self.assertEqual(dll.vonnx_predict_v1(handle,C.byref(desc),C.byref(output)),0,dll.vonnx_last_error())
            self.assertEqual(output.contents.frame_id,11);dll.vonnx_result_free(output)
        finally:dll.vonnx_destroy_engine(handle)
        cfg.graph_specialization=2;self.assertFalse(dll.vonnx_create_engine_v4(C.byref(cfg)))
    def test_ort_cuda_session_after_last_engine_close(self):
        import onnxruntime as ort
        from onnx import helper,numpy_helper
        model=onnx.load(self.path);model.graph.node[-1].output[0]='expanded'
        model.graph.initializer.append(numpy_helper.from_array(np.ones((1,3,1,1),np.float32),'conv_weight'))
        model.graph.node.extend([helper.make_node('Conv',['images','conv_weight'],['conv']),
            helper.make_node('ReduceMean',['conv'],['mean'],keepdims=0),
            helper.make_node('Mul',['expanded','mean'],['output0'])])
        onnx.checker.check_model(model);onnx.save(model,self.path)
        e=vONNXEngine(self.config(max_batch_size=1,precision='fp32'))
        path=e.model_spec.prepared_model_path
        e.close();del e
        session=ort.InferenceSession(path,providers=['CUDAExecutionProvider'])
        out=session.run(None,{session.get_inputs()[0].name:np.ones((1,3,32,64),np.float32)})
        self.assertTrue(np.isfinite(out[0]).all())
    def test_graph_rejects_runtime_cpu_control_flow(self):
        from onnx import helper,numpy_helper,TensorProto
        model=onnx.load(self.path);model.graph.node[-1].output[0]='expanded'
        model.graph.initializer.append(numpy_helper.from_array(np.array(0,np.float32),'zero_float'))
        then=helper.make_graph([helper.make_node('Identity',['expanded'],['then_output'])],'then',[],
                              [helper.make_tensor_value_info('then_output',TensorProto.FLOAT,[1,5,4])])
        other=helper.make_graph([helper.make_node('Mul',['expanded','zero_float'],['else_output'])],'else',[],
                               [helper.make_tensor_value_info('else_output',TensorProto.FLOAT,[1,5,4])])
        model.graph.node.extend([helper.make_node('ReduceMean',['images'],['mean'],keepdims=0),
            helper.make_node('Greater',['mean','zero_float'],['condition']),
            helper.make_node('If',['condition'],['output0'],then_branch=then,else_branch=other)])
        onnx.checker.check_model(model);onnx.save(model,self.path)
        e=vONNXEngine(self.config(max_batch_size=1))
        try:self.assertEqual(len(e.submit_numpy(0,0,np.ones((32,64,3),np.uint8),PixelFormat.RGB24).result(timeout=5)[0].boxes),2)
        finally:e.close()
        with self.assertRaisesRegex(RuntimeError,'(?i)(cpu|cuda|graph|fallback|capture)'):
            vONNXEngine(self.config(max_batch_size=1,cuda_graph=True))

if __name__=='__main__':unittest.main()
