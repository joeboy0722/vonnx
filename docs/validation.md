# Validation and benchmark guide

## Environment and build

Development target: Windows, RTX 3070 8GB, Python 3.12, CUDA 13, ONNX Runtime GPU 1.28.0, Ultralytics 8.4.115, Torch 2.13.0+cu126. `requirements-validation.txt` records the tested Python packages; install them in the project environment, without replacing an already-working CUDA Torch installation.

```powershell
.\venv\python.exe -m pip install -r requirements-validation.txt
.\build.ps1
.\venv\python.exe -m unittest discover -s tests -p 'test_*.py'
```

The build wrapper discovers VS 2022, sets its compiler environment, builds the extension and standalone DLL. Close Python processes that have loaded the extension before rebuilding: Windows locks loaded DLLs. Output DLL/header: `vonnx_standalone_sdk/vonnx.dll` and `include/vonnx_c_api.h`. CUDA/cuDNN DLLs must also be available at runtime; Python registers project-environment DLL paths and keeps their directory handles alive. Additional directories can be supplied through `VONNX_DLL_DIRS`.

CMake is also supported with `ONNXRUNTIME_ROOT_DIR` pointing at matching headers/import libraries. `test_scheduler` is a CPU scheduling regression, linked against CUDA only for the public header types. Debug assertions remain enabled for that target in release builds.

## Models and initialization

```powershell
.\venv\python.exe -m vonnx_tools prepare --datasets smoke
```

This prepares 15 official YOLOv8 models: Detect/Pose/Seg each n/s/m/l/x. Each weight is checked against the SHA256 digest in the official Ultralytics GitHub release. Existing root-level weights are reused only when the official digest matches. ONNX export uses opset 17, FP32, dynamic spatial/batch dimensions, `nms=False`, `simplify=False`.

Generated files are cached under `models/<model>/`; manifests record the official URL, SHA256, output descriptors, export parameters and environment. Small official COCO8 datasets go under `datasets/`. For unusually slow GitHub downloads, `tools/download_ranges.py <model> ... --workers 16` offers exact byte-range downloads and requires a matching official SHA256 before promoting the file into the cache.

`imgsz` is an integer square size or `(height,width)`; positive stride multiples are required. An engine keeps this inference size throughout its lifetime. Raw source sizes can differ. `max_input_width/height` bound source buffers; raise these explicitly for 4K sources. Python accepts contiguous uint8 HWC arrays, and NV12 accepts `[H*3/2,W]`. Packed array views with negative or nonstandard strides must be copied using `numpy.ascontiguousarray`.

Static ONNX requires the exact spatial dimensions, a static batch of one and `max_batch_size=1`. Unknown tasks, types, ranks, channel metadata and embedded-NMS models are rejected. No fallback class count or anchor count is guessed.

## Quick correctness gates

```powershell
.\venv\python.exe -m vonnx_tools validate --tasks detect --output runs\smoke_detect
.\venv\python.exe -m vonnx_tools validate --tasks pose segment --output runs\smoke_pose_seg
```

Run Detect first and proceed only when its command succeeds. The default quick matrix covers all selected models, 320/640/960 and 384x640, with batch limits 1/2/4 and eight small-dataset images. Each case runs in a child process so GPU allocations are released between cases. Validation stops at the first failure; inspect its `.log` before continuing.

Checks compare CUDA preprocessing against official LetterBox (max error <=1e-4), raw tensors against the identical ONNX and physical batch (`atol=1e-4`, `rtol=1e-3`), then outputs against official FP32 PyTorch. Box/keypoint coordinate error must be <=1 original pixel, confidence error <=1e-3 and binary mask IoU >=0.99. TF32 is disabled on both paths. Threshold-adjacent ambiguities are explicitly recorded rather than changing acceptance tolerances.

The unit tests cover scheduler callbacks/errors/draining/capacity, deterministic NMS chains, async host ownership, pixel formats, input rejection, static batch, C API descriptors/results, default/external CUDA stream dependencies and concurrent model output consistency. The real multi-model test requires prepared n Detect/Pose/Seg weights and COCO8-pose images.

## Full validation and performance: start after this chat

```powershell
.\run_full.ps1 -Resume
```

This script **does not run automatically**. It prepares all models and downloads official COCO val2017 images plus annotations, runs complete validation, then benchmarks. Validation failure stops the script before benchmarking. The download includes the official train/val annotation archive but no train2017 image archive.

Equivalent independent commands:

```powershell
.\venv\python.exe -m vonnx_tools prepare --datasets full
.\venv\python.exe -m vonnx_tools validate --full --output runs\validation --timeout 21600 --resume
.\venv\python.exe -m vonnx_tools benchmark --validation runs\validation --output runs\benchmark --resume
```

Full validation measures bbox AP and, where applicable, COCO keypoint/mask AP at confidence 0.001. All models run at 320/640/960. Differences from the same-setting official baseline must be <=0.5 percentage points. Pose uses images with valid keypoint annotations. Debug comparisons also run before each dataset evaluation. `--limit` is useful for development but cannot produce a complete-validation certificate.

Benchmarking refuses missing/stale/partial validation certificates. Certificates and resume records include model/build fingerprints. `--allow-unvalidated` exists only for development and permanently marks those case records as unvalidated.

The wrapper defaults to 500 validation images and 500 measured requests per model/case. Use `./run_full.ps1 -Resume -ValidationLimit 500` for sampled COCO validation. This retains compatible successful full-set validations, uses a fixed random seed (20261011) for the remaining model/size cases, and accepts the resulting subset certificates for benchmarking. Each benchmark records the actual validation scope and image count. These results do not certify full COCO val2017 AP; the original 0.5 percentage-point comparison threshold still applies to the sampled AP. Explicitly pass `-ValidationLimit 0` to require the complete dataset, and `-BenchmarkRequests 0` to use duration-only benchmarking.

For a quick benchmark, add `-QuickBenchmark`: only 640 resolution, 1/16 request lanes, one repeat, and at most 100 measured requests per model/case (30 seconds remains a safety ceiling). All 15 models, the original 17 multi-model groups, same-image fanout and both official single-model baselines remain included: 134 cases instead of 3015. The 30 warmup rounds remain excluded from measurement. Quick results go to `runs/benchmark_quick` and are marked request-limited; initialization, warmup and sampled correctness checks still take time. The independent benchmark CLI accepts `--max-requests 500` or another nonnegative request cap. This cap counts inference submissions, not distinct images: benchmark inputs still cycle through preloaded COCO8 images.

Default matrix:

| Group | Matrix |
|---|---|
| Single model | 15 models x 3 sizes x 5 stream counts x native/PyTorch single/PyTorch batched x 3 repeats |
| Same-task models | Each task: n+s, m+l, x+n, n+s+m+l+x; 3 sizes x 5 stream counts x 3 repeats |
| Mixed tasks | Detect+Pose+Seg for each n/s/m/l/x; 3 sizes x 5 stream counts x 3 repeats |
| Same-image fanout | Same mixed groups and settings; also measures all three tasks completing |

Stream counts are 1/4/8/16/32 per model. Max batch is 16 (one for the explicit PyTorch-single baseline), batch deadline 2ms, at least 30 warmup batches, 30 measured seconds, three repeats. **3015 cases; measured intervals alone total about 25.1 hours**, before downloads, validation, warmup and startup. OOM cases may finish early. This is a matrix-size calculation, not an execution ETA.

Warmup cycles through batch sizes from 1 to `min(streams_per_model, max_batch_size)` before measurement. Downloads are separate from these measured intervals. If GitHub is slow, `venv\python.exe tools\download_ranges.py yolov8x-seg --workers 16` retries the official asset using bounded parallel range requests and prints transfer speeds; the official release SHA256 must match before the weight is accepted. The official Hugging Face YOLOv8 repository currently contains Detect weights only, and those may differ from the pinned release; a faster transfer alone does not make an alternative weight interchangeable.

Cases retain their requested parameters; OOM, initialization errors and timeouts are recorded without reducing the workload. Benchmark workers preload real images and exclude file decoding, model loading and warmup from the timer. One request per lane may be outstanding. Mixed/same-task groups use independent per-model lanes; fanout explicitly synchronizes a shared image cohort.

Latency is submission to CPU output availability. Native GPU stage timings use CUDA events and are **physical-batch elapsed timings shared by every result in that batch**; do not sum them over individual requests. CPU result packing and Python return overhead remain visible in end-to-end latency. FPS counts successful model inferences, not unique camera frames. Fanout separately measures all-task completion latency. GPU memory is device-wide and sampled at 50ms intervals, so short peaks and other GPU applications must be considered when interpreting it.

Outputs: per-case `.json`, `.log`, `.requests.jsonl`, `summary.csv`, `summary.json`, `REPORT.md`, FPS and P95 plots. Per-model multi-model degradation is compared with its matching single-model native baseline. Cases run in isolated processes; timeout kills the worker and subsequent cases continue. `--resume` preserves completed success/failure/timeout records with identical fingerprints. Re-run a failed case without `--resume` after fixing its cause.

List or select a case:

```powershell
.\venv\python.exe -m vonnx_tools benchmark --list
.\venv\python.exe -m vonnx_tools benchmark --case single__yolov8n__320__s1__native__r1
.\venv\python.exe -m vonnx_tools report --input runs\benchmark
```

## C ABI and lifetime

Use `vonnx_config_init`, set model path/imgsz/input limits, then `vonnx_create_engine_v1`. Initialize `vonnx_image_v1` with `version=1`, `struct_size=sizeof(...)`, a full buffer size and row stride; call `vonnx_predict_v1`. Its result owns box/keypoint/mask storage until `vonnx_result_free`. Array ordering is identical across boxes and their task data. `vonnx_last_error` is thread-local; negative return or a null engine is a failure.

Legacy `vonnx_predict` remains bbox-only and rejects Pose/Seg. C descriptors support pitched input; Python intentionally requires contiguous arrays. Host async submission copies input; GPU buffers must remain valid and unmodified until completion. `user_stream` describes the producer stream; the engine inserts an event dependency before reading the frame. Do not destroy an engine concurrently with API calls or call stop/destroy from its callback.

Stop rejects new work and drains accepted requests. Restarting a stopped engine is an error; create a new engine for another imgsz. Callback exceptions are isolated. NV12 assumes contiguous planes with equal pitch and full-range BT.601; convert other colorimetry/ranges before submitting.
