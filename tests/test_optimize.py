import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from vonnx_tools.optimize import cases, gate, optimize


class OptimizeTests(unittest.TestCase):
    def test_shared_quota_keeps_producers_eligible_and_exact_ids(self):
        import collections
        import time
        import numpy as np
        from vonnx_tools import optimize as module
        from vonnx_tools.common import ROOT
        counts=collections.Counter()
        class Future:
            def __init__(self,lane,frame):self.lane,self.frame=lane,frame
            def result(self,timeout=None):
                if self.frame>=25:
                    time.sleep(.0001 if self.lane==0 else .01)
                    counts[self.lane]+=1
                return [SimpleNamespace(stream_id=self.lane,frame_id=self.frame,batch_size=1,physical_batch_size=1)]
        class Engine:
            closed=False
            def submit_numpy(self,lane,frame,*args):return Future(lane,frame)
            def close(self):self.closed=True
        engine=Engine()
        nvml=SimpleNamespace(nvmlInit=lambda:None,nvmlShutdown=lambda:None,
            nvmlDeviceGetHandleByIndex=lambda i:0,nvmlDeviceGetMemoryInfo=lambda d:SimpleNamespace(used=1))
        case=dict(id='quota',backend='native',model='fixture',lanes=32,total=100,precision='fp32',timings=False)
        with tempfile.TemporaryDirectory() as temp,patch.dict('sys.modules',{'pynvml':nvml}),\
             patch.object(module,'manifest',return_value={'task':'detect'}),\
             patch.object(module,'images',return_value=[ROOT/'fixture.jpg']),\
             patch.object(module,'load_rgb',return_value=[np.zeros((32,32,3),np.uint8)]),\
             patch.object(module,'native',return_value=engine),patch.object(module,'build_fingerprint',return_value='fixture'):
            result=module.run(case,Path(temp))
            data=[json.loads(s) for s in (Path(temp)/'quota.jsonl').read_text().splitlines()]
        self.assertEqual(result['status'],'passed');self.assertEqual(result['success'],75)
        self.assertEqual(sorted(r['frame_id'] for r in data),list(range(25,100)))
        self.assertGreater(max(counts.values()),3)  # Fast producers are not capped at ceil(75/32).
        self.assertEqual(result['request_issuance'],'shared_quota_v2');self.assertTrue(engine.closed)
    def test_worker_failure_stops_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            case = cases("baseline")[0]
            args = SimpleNamespace(
                output=directory,
                worker=None,
                phase="baseline",
                case=case["id"],
                resume=False,
                list=False,
                timeout=1,
            )
            with patch("vonnx_tools.optimize.environment", return_value={}), patch(
                "vonnx_tools.optimize.report"
            ), patch("vonnx_tools.optimize.subprocess.run", return_value=SimpleNamespace(returncode=1)):
                with self.assertRaisesRegex(RuntimeError, "Failure/timeout"):
                    optimize(args)
            result = json.loads((Path(directory) / (case["id"] + ".json")).read_text())
            self.assertEqual(result["status"], "failed")

    def test_budget_and_confirmation_gate(self):
        quick = cases("quick")
        self.assertEqual(len(quick), 21)
        self.assertTrue(all(c["total"] == 100 and c["timings"] is False for c in quick))
        self.assertEqual(len(cases("confirm")), 36)
        results = [
            {
                "case": c,
                "status": "passed",
                "cpu_outputs_retained": True,
                "fps": 110 if c["backend"] == "native" else 100,
            }
            for c in cases("confirm")
        ]
        self.assertTrue(gate(results, "confirm")["passed"])
        # A single-lane result cannot replace throughput acceptance.
        results.append(
            {"case": {"model": "yolov8s", "lanes": 1, "repeat": 1, "backend": "native"}, "status": "timeout"}
        )
        self.assertTrue(gate(results, "confirm")["passed"])
        row = next(r for r in results if r["case"]["backend"] == "native" and r["case"]["lanes"] == 16)
        row["status"] = "timeout"
        self.assertFalse(gate(results, "confirm")["passed"])

    def test_each_combination_requires_two_wins_and_median(self):
        results = [
            {
                "case": c,
                "status": "passed",
                "cpu_outputs_retained": True,
                "fps": (110 if c["repeat"] == 1 else 90) if c["backend"] == "native" else 100,
            }
            for c in cases("confirm")
        ]
        self.assertFalse(gate(results, "confirm")["passed"])


if __name__ == "__main__":
    unittest.main()
