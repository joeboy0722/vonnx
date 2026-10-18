"""Bounded FP16 fixtures, conversion/cache integrity and native dtype coverage."""
import concurrent.futures
import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel
from test_native import fixture
from vonnx import EngineConfig, vONNXEngine, PixelFormat
from vonnx.precision import prepare

class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.path=self.root/'source.onnx';fixture(self.path)
        self.cache=self.root/'cache'
    def tearDown(self):self.temp.cleanup()
    def config(self,path=None,precision='auto',**settings):
        c=EngineConfig();c.model_path=str(path or self.path);c.precision=precision
        c.model_cache_dir=str(self.cache);c.imgsz=(32,64);c.max_batch_size=8
        c.max_input_height=32;c.max_input_width=64
        c.iou_threshold=.5
        for k,v in settings.items():setattr(c,k,v)
        return c
    def check_engine(self,path,precision,source,converted):
        e=vONNXEngine(self.config(path,precision))
        try:
            self.assertEqual(e.model_spec.source_precision,source)
            self.assertEqual(e.model_spec.converted,converted)
            image=np.zeros((32,64,3),np.uint8)
            futures=[e.submit_numpy(i,100+i,image,PixelFormat.RGB24) for i in range(16)]
            results=[f.result(timeout=10)[0] for f in futures]
            self.assertEqual([r.frame_id for r in results],list(range(100,116)))
            self.assertTrue(all(len(r.boxes)==2 for r in results))
            if e.model_spec.task=='pose':self.assertTrue(all(len(r.keypoints)==len(r.boxes) for r in results))
            if e.model_spec.task=='segment':self.assertTrue(all(len(r.masks)==len(r.boxes) for r in results))
            self.assertTrue(all(r.physical_batch_size>=r.batch_size for r in results))
            self.assertTrue(all(r.physical_batch_size==(e.model_spec.fixed_batch_size or r.batch_size) for r in results))
            before=e.resource_stats
            for _ in range(2):e.pause();e.resume()
            self.assertEqual(e.resource_stats.explicit_device_allocations,before.explicit_device_allocations)
            self.assertEqual(e.resource_stats.initialization_count,1)
        finally:e.close()
        with self.assertRaises(RuntimeError):e.submit_numpy(0,0,image,PixelFormat.RGB24)
    def test_selection_matrix_and_native_half_io(self):
        self.check_engine(self.path,'auto','fp32',False)
        self.check_engine(self.path,'fp32','fp32',False)
        self.check_engine(self.path,'fp16','fp32',True)
        model=convert_float_to_float16(onnx.load(self.path),keep_io_types=False)
        OnnxModel(model).topological_sort()
        half=self.root/'half.onnx';onnx.save(model,half)
        for mode in ('auto','fp16'):self.check_engine(half,mode,'fp16',False)
        with self.assertRaisesRegex(ValueError,'cannot be requested as FP32'):
            vONNXEngine(self.config(half,'fp32'))
        mixed=prepare(self.path,self.cache)
        self.check_engine(mixed,'auto','mixed',False)
        self.check_engine(mixed,'fp16','mixed',True)
        with self.assertRaisesRegex(ValueError,'cannot be requested as FP32'):
            vONNXEngine(self.config(mixed,'fp32'))
    def test_cache_integrity_invalidation_and_concurrent_conversion(self):
        original=self.path.read_bytes()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            paths=list(pool.map(lambda _:prepare(self.path,self.cache),range(2)))
        self.assertEqual(paths[0],paths[1]);self.assertEqual(self.path.read_bytes(),original)
        modified=paths[0].stat().st_mtime_ns
        self.assertEqual(prepare(self.path,self.cache).stat().st_mtime_ns,modified)
        paths[0].write_bytes(b'corrupt')
        onnx.checker.check_model(str(prepare(self.path,self.cache)))
        model=onnx.load(self.path);model.doc_string='changed';onnx.save(model,self.path)
        self.assertNotEqual(prepare(self.path,self.cache),paths[0])
        blocked=self.root/'not-a-directory';blocked.write_text('x')
        with self.assertRaises(OSError):prepare(self.path,blocked)
    def test_concurrent_engine_initialization_and_empty_results(self):
        model=onnx.load(self.path)
        for i,tensor in enumerate(model.graph.initializer):
            if tensor.name=='prediction':
                values=onnx.numpy_helper.to_array(tensor).copy();values[:,4]=0
                model.graph.initializer[i].CopyFrom(onnx.numpy_helper.from_array(values,'prediction'))
        onnx.save(model,self.path)
        def work(_):
            e=vONNXEngine(self.config(precision='fp16',max_batch_size=2))
            try:
                r=e.submit_numpy(0,42,np.zeros((32,64,3),np.uint8),PixelFormat.RGB24).result(timeout=10)[0]
                return r.frame_id,len(r.boxes)
            finally:e.close()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            self.assertEqual(list(pool.map(work,range(2))),[(42,0),(42,0)])
    def test_partial_padding_and_dynamic_latency_policy(self):
        for policy,mode,expected in [('auto','throughput',8),('dynamic','throughput',1),
                                      ('auto','latency',1),('fixed','latency',8)]:
            e=vONNXEngine(self.config(precision='fp16',batch_shape_policy=policy,batch_mode=mode))
            try:
                r=e.submit_numpy(7,11,np.zeros((32,64,3),np.uint8),PixelFormat.RGB24).result(timeout=10)[0]
                self.assertEqual((r.stream_id,r.frame_id,r.batch_size,r.physical_batch_size),(7,11,1,expected))
                self.assertEqual(len(r.boxes),2)
            finally:e.close()
        e=vONNXEngine(self.config(precision='fp32'))
        try:
            self.assertEqual(e.model_spec.fixed_batch_size,0)
            r=e.submit_numpy(0,1,np.zeros((32,64,3),np.uint8),PixelFormat.RGB24).result(timeout=10)[0]
            self.assertEqual((r.batch_size,r.physical_batch_size),(1,1))
        finally:e.close()
    def test_default_cache_location(self):
        import os
        from unittest.mock import patch
        c=self.config(precision='fp16');c.model_cache_dir=''
        with patch.dict(os.environ,{'LOCALAPPDATA':str(self.root/'local')}):
            e=vONNXEngine(c)
            try:
                self.assertEqual(Path(e.model_spec.prepared_model_path).parent,self.root/'local/vonnx/models')
            finally:e.close()
    def test_external_weights_and_subgraph_detection(self):
        model=onnx.load(self.path)
        external=self.root/'external.onnx'
        onnx.save_model(model,external,save_as_external_data=True,all_tensors_to_one_file=True,
                        location='weights.bin',size_threshold=0)
        first=prepare(external,self.cache)
        with (self.root/'weights.bin').open('ab') as f:f.write(b'changed')
        self.assertNotEqual(prepare(external,self.cache),first)
        # FP16 hidden in a subgraph must reject fp32 even with FP32 model I/O.
        model=onnx.load(self.path)
        half=onnx.helper.make_tensor('half',onnx.TensorProto.FLOAT16,[1],[1.])
        branch=onnx.helper.make_graph([onnx.helper.make_node('Constant',[],['branch_value'],value=half)],
            'branch',[],[onnx.helper.make_tensor_value_info('branch_value',onnx.TensorProto.FLOAT16,[1])])
        model.graph.initializer.append(onnx.numpy_helper.from_array(np.array(True),'condition'))
        model.graph.node.append(onnx.helper.make_node('If',['condition'],['unused'],then_branch=branch,else_branch=branch))
        onnx.save(model,self.path)
        with self.assertRaisesRegex(ValueError,'cannot be requested as FP32'):
            vONNXEngine(self.config(precision='fp32'))
    def test_pose_seg_half_outputs_and_invalid_model(self):
        for task in ('pose','segment'):
            path=self.root/(task+'.onnx');fixture(path,task)
            model=convert_float_to_float16(onnx.load(path),keep_io_types=False)
            OnnxModel(model).topological_sort();onnx.save(model,path)
            self.check_engine(path,'auto','fp16',False)
        self.path.write_bytes(b'invalid')
        with self.assertRaises(ValueError):vONNXEngine(self.config())
        with self.assertRaises(ValueError):vONNXEngine(self.config(precision='wrong'))
    def test_c_api_v3_conversion_and_precision_query(self):
        import ctypes as C
        import sys
        import vonnx.precision
        from test_c_api import ConfigV2, Image, Result
        from vonnx_tools.common import ROOT
        class ConfigV3(C.Structure):
            _fields_=[('version',C.c_uint32),('struct_size',C.c_uint32),('base',ConfigV2),
                ('precision',C.c_int),('batch_shape_policy',C.c_int)]+[(k,C.c_char_p) for k in ('model_cache_dir','precision_python','precision_tool')]
        class Info(C.Structure):
            _fields_=[('version',C.c_uint32),('struct_size',C.c_uint32)]+[(k,C.c_int) for k in
                ('source_precision','effective_precision','converted','input_dtype','output_count')]+[
                ('output_dtypes',C.c_int*2),('fixed_batch_size',C.c_int),('prepared_model_path',C.c_char*8192)]
        dll=C.CDLL(str(ROOT/'vonnx_standalone_sdk/vonnx.dll'))
        dll.vonnx_config_init_v3.argtypes=[C.POINTER(ConfigV3)]
        dll.vonnx_create_engine_v3.argtypes=[C.POINTER(ConfigV3)];dll.vonnx_create_engine_v3.restype=C.c_void_p
        dll.vonnx_model_info_get_v3.argtypes=[C.c_void_p,C.POINTER(Info)]
        dll.vonnx_destroy_engine.argtypes=[C.c_void_p]
        dll.vonnx_predict_v1.argtypes=[C.c_void_p,C.POINTER(Image),C.POINTER(C.POINTER(Result))]
        dll.vonnx_result_free.argtypes=[C.POINTER(Result)]
        dll.vonnx_result_physical_batch_v3.argtypes=[C.POINTER(Result)]
        dll.vonnx_last_error.restype=C.c_char_p
        c=ConfigV3();dll.vonnx_config_init_v3(C.byref(c))
        c.base.base.model_path=str(self.path).encode('utf-8')
        c.base.base.input_height=c.base.base.input_width=32
        c.base.base.max_input_height=c.base.base.max_input_width=64
        c.base.base.max_batch_size=2;c.precision=2
        c.model_cache_dir=str(self.cache).encode('utf-8')
        c.precision_python=sys.executable.encode('utf-8');c.precision_tool=vonnx.precision.__file__.encode('utf-8')
        handle=dll.vonnx_create_engine_v3(C.byref(c))
        self.assertTrue(handle,dll.vonnx_last_error())
        try:
            info=Info();info.version=3;info.struct_size=C.sizeof(info)
            self.assertEqual(dll.vonnx_model_info_get_v3(handle,C.byref(info)),0)
            self.assertEqual((info.source_precision,info.effective_precision,info.converted),(1,3,1))
            self.assertEqual(info.fixed_batch_size,2)
            image=np.zeros((32,32,3),np.uint8)
            desc=Image(1,C.sizeof(Image),image.ctypes.data,image.nbytes,32,32,96,1,0,0,4,12)
            result=C.POINTER(Result)()
            self.assertEqual(dll.vonnx_predict_v1(handle,C.byref(desc),C.byref(result)),0)
            self.assertEqual(result.contents.frame_id,12)
            self.assertEqual(dll.vonnx_result_physical_batch_v3(result),2)
            dll.vonnx_result_free(result)
        finally:dll.vonnx_destroy_engine(handle)

if __name__=='__main__':unittest.main()
