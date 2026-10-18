"""Export a compact, reproducible AP release gate; never performs inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"yolov8s": {"bbox"}, "yolov8s-pose": {"bbox", "keypoints"},
          "yolov8s-seg": {"bbox", "segm"}}


def report(folder, output):
    rows = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if len(rows) != 3 or {r["case"]["model"] for r in rows} != set(MODELS):
        raise ValueError("Require exactly the three s-model validations")
    binary = hashlib.sha256(next((ROOT / "vonnx").glob("vonnx_py*.pyd")).read_bytes()).hexdigest()
    converter = hashlib.sha256((ROOT / "vonnx/precision.py").read_bytes()).hexdigest()
    exported = []
    for row in rows:
        case = row["case"]
        full = row["detail"]["full"]
        model = case["model"]
        if (row["status"] != "passed" or case["precision"] != "fp16"
                or case["imgsz"] != [640, 640] or not case["full"] or case["limit"]
                or full["limited"] or case["build_fingerprint"] != binary
                or case["conversion_fingerprint"] != converter
                or set(full["metrics"]) != MODELS[model]):
            raise ValueError(f"Invalid/stale full AP gate: {model}")
        ids = full["image_ids"]
        if full["images"] != len(set(ids)) or not (0 < len(ids) <= 5000):
            raise ValueError("Invalid evaluation image IDs")
        if model != "yolov8s-pose" and len(ids) != 5000:
            raise ValueError("Detect/Seg must evaluate all 5000 images")
        for metric in full["metrics"].values():
            actual, reference = metric["engine_ap"], metric["official_ap"]
            if (not all(math.isfinite(v) and 0 <= v <= 1 for v in (actual, reference))
                    or reference - actual > .005 + 1e-12):
                raise ValueError(f"AP decline exceeds 0.5pp: {model}")
        exported.append(dict(model=model, images=full["images"], metrics=full["metrics"],
            image_ids_sha256=hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
            weight_sha256=row["model"]["weight_sha256"],
            source_onnx_sha256=row["model"]["onnx_sha256"],
            prepared_model={k: v for k, v in row["detail"]["prepared_model"].items() if k != "path"}))
    result = dict(passed=True, dataset="COCO val2017; pose images with num_keypoints > 0",
        native_precision="FP16 with protected FP32 decoder", reference_precision="official PyTorch FP32",
        imgsz=[640, 640], batch=1, confidence=.001, iou=.7, max_det=300, rect=False,
        allowed_drop_percentage_points=.5, binary_sha256=binary, converter_sha256=converter,
        environment=rows[0]["environment"], models=exported)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    lines = ["# FP16 AP 驗證（2026-10-18）", "",
        "三種 YOLOv8 s、640×640，相同權重對照官方 PyTorch FP32。",
        "Detect／Seg 使用完整 COCO val2017 5,000 張；Pose 使用有有效關鍵點標註的全部影像。",
        "此範圍全部通過 AP50–95 最多下降 0.5 個百分點的門檻；其他模型與尺寸不包含在此結論。", "",
        "| 模型 | 影像數 | 指標 | vONNX FP16 AP (%) | 官方 FP32 AP (%) | 下降（百分點） |",
        "|---|---:|---|---:|---:|---:|"]
    for row in exported:
        for kind, m in row["metrics"].items():
            lines.append(f"| {row['model']} | {row['images']} | {kind} | {m['engine_ap']*100:.4f} | "
                         f"{m['official_ap']*100:.4f} | {max(0,m['official_ap']-m['engine_ap'])*100:.4f} |")
    lines += ["", "設定：FP32 前後處理、confidence=0.001、IoU=0.7、max_det=300、rect=False、batch=1。",
        "Seg 使用原圖 ROI masks，COCO 評估前重建完整原圖 mask；官方使用 retina_masks=True。",
        "COCOeval 使用標準 maxDets（bbox／segm=100、keypoints=20）；兩方設定一致。",
        "", f"二進位 SHA256：`{binary}`。", f"轉換器 SHA256：`{converter}`。", "",
        "機器／依賴版本、權重與來源模型雜湊見 [JSON 明細](ap_fp16.json)。",
        "原始逐影像預測、完整評估 log 與 image IDs 保留於本機 `runs/fp16/ap_full/`。",
        "此次未執行長時間效能跑分。"]
    output.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({r['model']: r['metrics'] for r in exported}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "runs/fp16/ap_full")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/ap_fp16")
    args = parser.parse_args()
    report(args.input, args.output)
