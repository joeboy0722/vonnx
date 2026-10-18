"""Run a few asynchronous requests using a prepared official model and real image."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vonnx import EngineConfig, vONNXEngine, PixelFormat, load_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolov8n/yolov8n.onnx")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--precision", choices=["auto","fp32","fp16"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--requests", type=int, default=4, choices=range(1, 33))
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    candidates = sorted((ROOT / "datasets/coco8/images").rglob("*.jpg"))
    image_path = args.image or (candidates[0] if candidates else None)
    if not args.model.is_file() or image_path is None:
        parser.error("Prepare models/images first: python -m vonnx_tools prepare --scales n --datasets smoke")
    image = load_image(str(image_path))
    config = EngineConfig()
    config.model_path = str(args.model)
    config.precision=args.precision
    if args.cache_dir:config.model_cache_dir=str(args.cache_dir)
    config.imgsz = args.imgsz
    config.max_batch_size = args.batch
    config.max_input_height, config.max_input_width = image.shape[:2]
    engine = vONNXEngine(config)
    try:
        spec=engine.model_spec
        print(f"Precision: {spec.source_precision} -> {spec.effective_precision}; converted={spec.converted}")
        pending = [engine.submit_numpy(lane, lane, image, PixelFormat.RGB24) for lane in range(args.requests)]
        for future in pending:
            result = future.result(timeout=30)[0]
            print(
                f"request={result.frame_id} batch={result.batch_size} physical={result.physical_batch_size} boxes={len(result.boxes)} keypoints={len(result.keypoints)} masks={len(result.masks)}"
            )
        engine.pause()
        engine.resume()
        print("Resource state:", engine.resource_stats.state)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
