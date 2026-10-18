import unittest
from vonnx_tools.validate import ap_comparison

class PrecisionMetricTests(unittest.TestCase):
    def test_missing_required_outputs_cannot_pass_correctness(self):
        import numpy as np
        from types import SimpleNamespace as NS
        from vonnx_tools.validate import compare
        class Data:
            def cpu(self):return self
            def numpy(self):return np.array([[0,0,10,10,.9,0]],np.float32)
        actual=NS(boxes=[NS(x1=0,y1=0,x2=10,y2=10,score=.9,class_id=0)],keypoints=[],masks=[])
        with self.assertRaisesRegex(AssertionError,'Keypoint count'):
            compare(actual,NS(boxes=NS(data=Data()),keypoints=object(),masks=None),(32,32))
        with self.assertRaisesRegex(AssertionError,'Mask count'):
            compare(actual,NS(boxes=NS(data=Data()),keypoints=None,masks=object()),(32,32))
    def test_fp16_acceptance_is_a_drop_budget(self):
        self.assertAlmostEqual(ap_comparison(.496,.5,'fp16')['drop_percentage_points'],.4)
        self.assertAlmostEqual(ap_comparison(.495,.5,'fp16')['drop_percentage_points'],.5)
        self.assertEqual(ap_comparison(.51,.5,'fp16')['drop_percentage_points'],0.)
        with self.assertRaisesRegex(AssertionError,'drop'):
            ap_comparison(.494,.5,'fp16')
        # FP32 engine/reference correctness still requires agreement in either direction.
        with self.assertRaisesRegex(AssertionError,'difference'):
            ap_comparison(.51,.5,'fp32')
        for value in (float('nan'),float('inf'),-1.):
            with self.assertRaisesRegex(AssertionError,'Invalid AP'):
                ap_comparison(value,.5,'fp16')

if __name__=='__main__':unittest.main()
