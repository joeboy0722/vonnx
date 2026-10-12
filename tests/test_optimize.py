import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from vonnx_tools.optimize import cases, gate, optimize


class OptimizeTests(unittest.TestCase):
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
