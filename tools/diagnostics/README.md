# Optional diagnostics

These commands are independent of the regular tests and never start the full benchmark matrix.
Prepare Detect n/s and Pose l models plus COCO8/COCO8-pose first. Each Python worker has a 240-second timeout;
failure is recorded without reducing resolution, batch size or model count.

## Four models at 1920

From the repository root:

```powershell
.\venv\python.exe tools\diagnostics\quick_1920_multimodel.py --output runs\multimodel_1920
.\venv\python.exe tools\diagnostics\quick_1920_multimodel.py --backend official --output runs\multimodel_1920_official
```

Two independent Detect n engines, Detect s and Pose l; FP32, batch 1, one slot, 1920x1920 inference.
Preloaded 1920x1080 RGB images, one/four request lanes per model, 3-second idle-model controls and
8-second concurrent measurements. Results include full CPU boxes/keypoints, JSONL request logs,
latency percentiles, submission time and device-wide memory peaks.
The stock official predictor retains its intrinsic profiling/synchronization behavior.

## Native C++ client

Export input frames without inference:

```powershell
.\venv\python.exe tools\diagnostics\quick_1920_multimodel.py --export-native-inputs
```

In a VS 2022 x64 Developer Command Prompt, from the repository root:

```bat
if not exist build mkdir build
cl /nologo /utf-8 /O2 /MD /std:c++17 /EHsc /Iinclude tools\diagnostics\multimodel_1920_native.cpp /Febuild\multimodel_1920_native.exe /Fobuild\multimodel_1920_native.obj
build\multimodel_1920_native.exe "%CD%" "%CD%\runs\native_1920"
```

The executable uses the standalone C API v2 DLL, independent producer threads and CPU results;
there is no embedded Python runtime. It requires the built SDK and its CUDA/cuDNN runtime dependencies.
The C++ executable runs directly; use an external process timeout when automating it.
Experimental async/priority DLL modes were removed from this supported client.

## CUDA trace inspection

`quick_1920_multimodel.py --worker --profile` marks the one-lane concurrent scenario using CUDA profiler APIs.
Configure Nsight Systems with `--capture-range=cudaProfilerApi --capture-range-end=stop` and export its report
to SQLite. `analyze_stream_trace.py <trace.sqlite>` prints CUDA streams, APIs and synchronization targets.
Nsight Systems may require an ASCII working/output path on Windows; the repository itself supports Chinese paths.

Historical measurements and limitations: [performance notes](../../docs/performance.md).

## FP16 bounded report

`python tools/diagnostics/fp16_report.py --input runs/fp16` combines preserved
before/after/official short-test JSON into CSV, JSON and Markdown without GPU work.
FP16 setup, correctness and short-test commands: [FP16 guide](../../docs/fp16.md).

`python tools/diagnostics/ap_report.py` checks all three full s/640 FP16 AP records,
current binary/converter fingerprints and the 0.5 percentage point decline gate,
then exports the compact public `docs/ap_fp16.md` / JSON report without GPU work.

## Bounded latency optimization

`latency_opt.py --variant baseline` archives the original Python module / native DLL and
runs six Detect s/640 cases (FP32/FP16, 1/16/32 lanes). Preserve this archive before rebuilding.
`--variant candidate` runs the same cases after changes. Both use fixed batch 8 for throughput
and batch 1 for single-lane latency; each case has 25 completed warmups and 75 measurements.
`--phase diagnostic` enables additive schema-v2 host timestamps and CUDA intervals.
NVML clock, temperature, utilization and memory samples cover measurement only.
`--phase confirm` is guarded by all four primary quick cases improving; it never starts AP.

`latency_trace.py` captures four archived-baseline Nsight traces, using a temporary ASCII
junction for Nsight's Windows path limitation. `latency_trace_summary.py` summarizes the
GPU kernel intervals and non-kernel upper bounds. `latency_graph_probe.py --precision fp16`
tests unmodified ONNX capture eligibility in an isolated process; failed capture is retained,
without adding an unsupported production setting. `latency_functional.py` runs small real-model
same-ONNX comparisons; `latency_opt_report.py` renders the resulting report using CPU only.

## Cross-model software waits (v0.5.0)

`software_wait.py --phase functional` runs small same-ONNX specialization checks.
`--phase main` compares archived release/current binaries and the candidate on independent
X/L/n engines at 1920, fixed quotas and bounded saturation, 25 warmups / at most 75 measured
requests per model and case. `--phase single` covers s/640 at 1/16/32 lanes; `--graph`
selects the optional Graph candidate for single-model or functional checks. `--repeat 2`
and `3` remain 100-request short runs, never 500. Successful matching fingerprints resume.

`software_prepare_probe.py` separates FP32/FP16 and specialization order with one-image
EP profiles. `software_numeric_probe.py` isolates the 1920 X coordinate difference against
the same ONNX, official half and source FP32. `xln_trace.py` captures bounded Nsight ranges;
`xln_trace_summary.py` correlates CPU submissions, own-stream kernels and event waits.
All outputs stay under runs/. No command starts AP, a release, or Git operations.

## v0.5.0 sampled AP release gate

`ap_release.py --resume` runs only AP: three s models at 640, FP32/FP16,
off/specialized/Graph, 500 deterministic COCO val2017 images per configuration.
Pose samples from images with valid keypoint annotations. Each configuration is
isolated with a 20-minute timeout; an error stops the gate. It never starts benchmarks.
`ap_release_report.py` checks all 18 records, subset IDs, prepared modes and binary/tool
fingerprints, then exports `docs/ap_v050.md` and JSON. This is sampled regression,
not full-dataset AP. Publishing is a separate explicitly invoked operation.
