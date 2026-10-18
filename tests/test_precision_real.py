"""Small optional official-s checks for externally prepared FP16 I/O models."""
import tempfile
import unittest
from pathlib import Path

class RealHalfIOTests(unittest.TestCase):
    def test_detect_pose_seg_native_half_io(self):
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
        from onnxruntime.transformers.onnx_model import OnnxModel
        from vonnx_tools.common import ROOT, images
        from vonnx_tools.precision_check import smoke
        models=('yolov8s','yolov8s-pose','yolov8s-seg')
        if not all((ROOT/'models'/m/(m+'.onnx')).exists() for m in models) or not all(images(t) for t in ('detect','pose','segment')):
            self.skipTest('Optional real half-I/O test needs prepared s models and COCO8 images')
        with tempfile.TemporaryDirectory() as directory:
            for name in models:
                with self.subTest(model=name):
                    graph=convert_float_to_float16(onnx.load(ROOT/'models'/name/(name+'.onnx')),keep_io_types=False)
                    OnnxModel(graph).topological_sort();onnx.checker.check_model(graph)
                    path=Path(directory)/(name+'.onnx');onnx.save(graph,path)
                    detail=smoke(dict(model=name,model_path=str(path),precision='auto',
                        imgsz=[640,640],batch_limits=[1,4],image_limit=2))
                    self.assertEqual(detail['prepared_model']['input_dtype'],10)
                    self.assertTrue(all(t==10 for t in detail['prepared_model']['output_dtypes']))
                    self.assertEqual(len(detail['comparisons']),4)

if __name__=='__main__':unittest.main()
