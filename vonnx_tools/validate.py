import json
import time
from pathlib import Path
import numpy as np
from .common import (
    ROOT,
    names,
    manifest,
    engine,
    images,
    load_rgb,
    write_json,
    environment,
    build_fingerprint,
)


def box_array(result):
    return np.array(
        [[b.x1, b.y1, b.x2, b.y2, b.score, b.class_id] for b in result.boxes], dtype=np.float32
    ).reshape(-1, 6)


def mask_array(mask, shape):
    out = np.zeros(shape, dtype=np.uint8)
    out[mask.y : mask.y + mask.height, mask.x : mask.x + mask.width] = mask.data
    return out


def box_iou(a, b):
    intersection = np.maximum(0, np.minimum(a[2:4], b[2:4]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:4] - a[:2]).prod() + np.maximum(0, b[2:4] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.0


def compare(actual, official, shape):
    got = box_array(actual)
    want = official.boxes.data.cpu().numpy()[:, :6]
    used = set()
    ambiguous = []
    max_coord = 0.0
    min_mask_iou = 1.0
    max_keypoint = 0.0

    def threshold_ambiguity(box, collection, side):
        if abs(float(box[4]) - 0.25) < 1e-4:
            return {"side": side, "reason": "confidence", "box": box.tolist()}
        for other in collection:
            overlap = box_iou(box, other)
            if int(box[5]) == int(other[5]) and abs(overlap - 0.7) < 1e-4:
                return {
                    "side": side,
                    "reason": "nms_iou",
                    "iou": overlap,
                    "box": box.tolist(),
                    "other_box": other.tolist(),
                }
        return None

    for i, b in enumerate(got):
        candidates = [j for j, w in enumerate(want) if j not in used and int(w[5]) == int(b[5])]
        if not candidates:
            ambiguity = threshold_ambiguity(b, want, "engine")
            if ambiguity:
                ambiguous.append(ambiguity)
                continue
            raise AssertionError("Unmatched engine class/count")
        j = min(candidates, key=lambda j: np.abs(want[j, :4] - b[:4]).max())
        w = want[j]
        delta = float(np.abs(w[:4] - b[:4]).max())
        if delta > 1:
            ambiguity = threshold_ambiguity(b, want, "engine")
            if ambiguity:
                ambiguous.append(ambiguity)
                continue
            raise AssertionError(f"BBox coordinate error {delta:.6f}px")
        if abs(float(w[4] - b[4])) > 1e-3:
            raise AssertionError("Score mismatch")
        max_coord = max(max_coord, delta)
        used.add(j)
        if actual.keypoints:
            kp = np.array([[p.x, p.y, p.confidence] for p in actual.keypoints[i]])
            ref = official.keypoints.data[j].cpu().numpy()
            delta = float(np.abs(kp[:, :2] - ref[:, :2]).max())
            max_keypoint = max(max_keypoint, delta)
            if delta > 1 or np.abs(kp[:, 2] - ref[:, 2]).max() > 1e-3:
                raise AssertionError(f"Keypoint mismatch {delta:.6f}px")
        if actual.masks:
            mask = mask_array(actual.masks[i], shape)
            ref = official.masks.data[j].cpu().numpy() > 0
            union = np.logical_or(mask, ref).sum()
            iou = float(np.logical_and(mask, ref).sum() / union) if union else 1.0
            min_mask_iou = min(min_mask_iou, iou)
            if iou < 0.99:
                raise AssertionError(f"Mask IoU {iou:.6f} below 0.99")
    for j, w in enumerate(want):
        if j not in used:
            ambiguity = threshold_ambiguity(w, got, "official")
            if ambiguity:
                ambiguous.append(ambiguity)
            else:
                raise AssertionError("Unmatched official class/count")
    return {
        "max_bbox_error_px": max_coord,
        "max_keypoint_error_px": max_keypoint,
        "min_mask_iou": min_mask_iou,
        "ambiguities": ambiguous,
    }


def smoke_case(case):
    import torch
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from vonnx import PixelFormat

    info = manifest(case["model"])
    shape = tuple(case["imgsz"])
    paths = images(info["task"])
    rgb = load_rgb(paths)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    official = YOLO(str(ROOT / info["weight"]))
    official.to("cuda")
    official.model.float()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(ROOT / info["onnx"]),
        options,
        providers=[("CUDAExecutionProvider", {"use_tf32": False, "cudnn_conv_algo_search": "DEFAULT"})],
    )
    summaries = []
    max_pre = 0.0
    max_raw = 0.0
    for batch in case.get("batch_limits", (1, 2, 4)):
        eng = engine(case["model"], shape, batch=batch, debug=True)
        try:
            for start in range(0, len(rgb), batch):
                inputs = rgb[start : start + batch]
                futures = [eng.submit_numpy(j, start + j, a, PixelFormat.RGB24) for j, a in enumerate(inputs)]
                results = [f.result(timeout=120)[0] for f in futures]
                tensors = [
                    LetterBox(shape, auto=False)(image=np.ascontiguousarray(a[:, :, ::-1]))[:, :, ::-1]
                    .transpose(2, 0, 1)
                    .astype(np.float32)
                    / 255.0
                    for a in inputs
                ]
                # Compare identical physical batches, not batch-N versus batch-1 convolution algorithms.
                grouped = {}
                for index, r in enumerate(results):
                    grouped.setdefault(r.batch_size, []).append(index)
                raw_by_index = {}
                for actual_batch, indices in grouped.items():
                    if len(indices) % actual_batch:
                        raise AssertionError("Unexpected physical batch grouping")
                    for offset in range(0, len(indices), actual_batch):
                        selected = indices[offset : offset + actual_batch]
                        raw = session.run(
                            None, {session.get_inputs()[0].name: np.stack([tensors[i] for i in selected])}
                        )
                        for pos, index in enumerate(selected):
                            raw_by_index[index] = [v[pos : pos + 1] for v in raw]

                for j, (a, result) in enumerate(zip(inputs, results)):
                    if result.stream_id != j or result.frame_id != start + j:
                        raise AssertionError("Request/result ID mismatch")
                    tensor = (
                        LetterBox(shape, auto=False)(image=np.ascontiguousarray(a[:, :, ::-1]))[:, :, ::-1]
                        .transpose(2, 0, 1)
                        .astype(np.float32)
                        / 255.0
                    )
                    own = np.asarray(result.debug_input, np.float32).reshape(3, *shape)
                    delta = float(np.abs(own - tensor).max())
                    max_pre = max(max_pre, delta)
                    if delta > 1e-4:
                        raise AssertionError(f"Preprocess error {delta:.6g}")
                    raw = raw_by_index[j]
                    for k, ref in enumerate(raw):
                        out = np.asarray(result.debug_outputs[k], np.float32).reshape(ref.shape)
                        max_raw = max(max_raw, float(np.abs(out - ref).max()))
                        np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-3)
                    ref = official.predict(
                        np.ascontiguousarray(a[:, :, ::-1]),
                        imgsz=shape,
                        rect=False,
                        conf=0.25,
                        iou=0.7,
                        max_det=300,
                        quantize=None,
                        retina_masks=True,
                        verbose=False,
                    )[0]
                    summaries.append(compare(result, ref, a.shape[:2]))
            # Partial batch drains on stop and remains readable.
            pending = eng.submit_numpy(99, 99, rgb[0], PixelFormat.RGB24)
            eng.stop()
            assert pending.result()[0].frame_id == 99
            try:
                eng.submit_numpy(0, 0, rgb[0], PixelFormat.RGB24)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Stopped engine accepted request")
            try:
                eng.start()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Stopped engine restarted")
        finally:
            eng.stop()
    return {
        "images": len(paths),
        "batch_limits": list(case.get("batch_limits", (1, 2, 4))),
        "max_preprocess_error": max_pre,
        "max_raw_output_error": max_raw,
        "comparisons": summaries,
    }


def coco_records(result, image_id, task, categories, shape, official=False):
    from pycocotools import mask as mask_utils

    records = []
    boxes = result.boxes.data.cpu().numpy()[:, :6] if official else box_array(result)
    for i, b in enumerate(boxes):
        record = {
            "image_id": image_id,
            "category_id": categories[int(b[5])],
            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
            "score": float(b[4]),
        }
        if task == "pose":
            kp = (
                result.keypoints.data[i].cpu().numpy()
                if official
                else np.array([[p.x, p.y, p.confidence] for p in result.keypoints[i]])
            )
            kp = kp.copy()
            kp[:, 2] = 1
            record["keypoints"] = kp.flatten().tolist()
        if task == "segment":
            mask = (
                result.masks.data[i].cpu().numpy().astype(np.uint8)
                if official
                else mask_array(result.masks[i], shape)
            )
            encoded = mask_utils.encode(np.asfortranarray(mask))
            encoded["counts"] = encoded["counts"].decode("ascii")
            record["segmentation"] = encoded
        records.append(record)
    return records


def full_case(case, output):
    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from vonnx import PixelFormat

    info = manifest(case["model"])
    task = info["task"]
    shape = tuple(case["imgsz"])
    paths = images(task, True)
    if len(paths) != 5000:
        raise RuntimeError("Full validation requires all 5000 COCO val2017 images")
    annotation = (
        ROOT
        / "datasets/coco/annotations"
        / ("person_keypoints_val2017.json" if task == "pose" else "instances_val2017.json")
    )
    gt = COCO(str(annotation))
    categories = [c["id"] for c in gt.dataset["categories"]]
    if task == "pose":
        categories = [1]
        paths = [p for p in paths if any(a.get("num_keypoints", 0) > 0 for a in gt.imgToAnns[int(p.stem)])]
    limited = case.get("limit", 0)
    if limited:
        # Use the same deterministic subset at every resolution and model scale.
        paths = sorted(np.random.default_rng(20261011).permutation(paths)[:limited])
    eng = engine(case["model"], shape, conf=0.001)
    official = YOLO(str(ROOT / info["weight"]))
    official.to("cuda")
    official.model.float()
    own_records = []
    ref_records = []
    image_ids = []
    started = time.perf_counter()
    print(f"COCO validation: {len(paths)} images; subset={bool(limited)}", flush=True)
    try:
        for index, path in enumerate(paths):
            a = load_rgb([path])[0]
            image_id = int(path.stem)
            image_ids.append(image_id)
            own = eng.predict_numpy(0, index, a, PixelFormat.RGB24)[0]
            ref = official.predict(
                np.ascontiguousarray(a[:, :, ::-1]),
                imgsz=shape,
                rect=False,
                conf=0.001,
                iou=0.7,
                max_det=300,
                quantize=None,
                retina_masks=True,
                verbose=False,
            )[0]
            own_records.extend(coco_records(own, image_id, task, categories, a.shape[:2]))
            ref_records.extend(coco_records(ref, image_id, task, categories, a.shape[:2], True))
            if (index + 1) % 100 == 0 or index + 1 == len(paths):
                print(
                    f"  images {index + 1}/{len(paths)}; elapsed {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    finally:
        eng.stop()
    metrics = {}
    for kind in ["bbox"] + (["keypoints"] if task == "pose" else ["segm"] if task == "segment" else []):
        scores = []
        for label, records in [("engine", own_records), ("official", ref_records)]:
            print(f"Evaluating {label} {kind} AP", flush=True)
            if not records:
                raise AssertionError("No predictions on full validation set")
            write_json(Path(output) / (case["id"] + f"-{label}-predictions.json"), records)
            evaluation = COCOeval(
                gt,
                gt.loadRes(
                    [{k: v for k, v in r.items() if k != "bbox"} for r in records]
                    if kind == "segm"
                    else records
                ),
                kind,
            )
            evaluation.params.imgIds = image_ids
            evaluation.evaluate()
            evaluation.accumulate()
            evaluation.summarize()
            scores.append(float(evaluation.stats[0]))
        metrics[kind] = {
            "engine_ap": scores[0],
            "official_ap": scores[1],
            "difference_percentage_points": abs(scores[0] - scores[1]) * 100,
        }
        if abs(scores[0] - scores[1]) > 0.005:
            raise AssertionError(f"{kind} mAP difference exceeds 0.5 percentage points")
    return {
        "images": len(paths),
        "metrics": metrics,
        "limited": bool(limited),
        "sample_seed": 20261011 if limited else None,
        "image_ids": image_ids,
    }


def validate(args):
    from .isolated import isolated

    if args.limit < 0 or any(s <= 0 or s % 32 for s in args.sizes):
        raise ValueError("Invalid limit/imgsz")
    if args.worker:
        case = json.loads(Path(args.worker).read_text(encoding="utf-8"))
        info = manifest(case["model"])
        detail = smoke_case(case)
        if case["full"]:
            detail["full"] = full_case(case, args.output)
        write_json(
            Path(args.output) / (case["id"] + ".json"),
            {"case": case, "status": "passed", "model": info, "environment": environment(), "detail": detail},
        )
        return
    cases = []
    for model in names(args.tasks, args.scales):
        for size in [*[[s, s] for s in args.sizes], *([[384, 640]] if not args.full else [])]:
            cases.append(
                {
                    "id": model + "-" + str(size[0]) + "x" + str(size[1]),
                    "model": model,
                    "imgsz": size,
                    "full": args.full,
                    "limit": args.limit,
                    "model_hash": manifest(model)["onnx_sha256"],
                    "build_fingerprint": build_fingerprint(),
                }
            )
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    if not cases:
        raise ValueError("No matching validation cases")
    rows = []
    for case in cases:
        record = Path(args.output) / (case["id"] + ".json")
        if args.resume and case["full"] and case["limit"] and record.exists():
            old = json.loads(record.read_text(encoding="utf-8"))
            previous = old.get("case", {})
            if (
                old.get("status") == "passed"
                and previous.get("limit") == 0
                and {k: v for k, v in previous.items() if k != "limit"}
                == {k: v for k, v in case.items() if k != "limit"}
            ):
                print(case["id"], "passed (reuse full-set validation)", flush=True)
                rows.append(old)
                continue
        row = isolated("validate", case, args.output, args.timeout, resume=args.resume)
        rows.append(row)
        if row["status"] != "passed":
            write_json(Path(args.output) / "summary.json", rows)
            raise RuntimeError("Validation gate failed; inspect case log before continuing")
    write_json(Path(args.output) / "summary.json", rows)
