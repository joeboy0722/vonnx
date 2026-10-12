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
