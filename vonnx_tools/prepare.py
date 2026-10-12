from pathlib import Path
import importlib.metadata as metadata
import json
import urllib.request
import zipfile
import shutil
import time
from .common import ROOT, VERSIONS, names, sha256, write_json, environment


def download_archive(url, partial):
    """Resume interrupted downloads without disabling TLS certificate checks."""
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        started = last_report = time.monotonic()
        received = 0
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                resumed = response.status == 206
                content_range = response.headers.get("Content-Range", "")
                if resumed and not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError("Download server returned an incorrect range")
                if not resumed:
                    offset = 0
                length = response.headers.get("Content-Length")
                expected = int(length) if length else None
                with partial.open("ab" if resumed else "wb") as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
                        received += len(block)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            total = f"/{(offset + expected) / 1e6:.1f}" if expected else ""
                            print(
                                f"  {(offset + received) / 1e6:.1f}{total} MB; {received / (now - started) / 1e6:.2f} MB/s",
                                flush=True,
                            )
                            last_report = now
                if expected is not None and received != expected:
                    raise RuntimeError(f"Incomplete download: {received}/{expected} bytes")
            print(f"Downloaded {partial.stat().st_size / 1e6:.1f} MB", flush=True)
            return
        except Exception as error:
            print(f"Download attempt {attempt + 1}/3 failed: {error}", flush=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def fetch_zip(url, destination, marker):
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / (url.rsplit("/", 1)[-1])
    if not archive.exists():
        partial = archive.with_suffix(".download")
        print("Download", url, flush=True)
        download_archive(url, partial)
        with zipfile.ZipFile(partial) as z:
            bad = z.testzip()
            if bad:
                partial.unlink()
                raise RuntimeError(f"Archive CRC check failed: {bad}; retry preparation")
        partial.replace(archive)
    with zipfile.ZipFile(archive) as z:
        root = destination.resolve()
        for item in z.infolist():
            if not (destination / item.filename).resolve().is_relative_to(root):
                raise ValueError("Unsafe archive path")
        z.extractall(destination)
    if not marker.exists():
        raise RuntimeError(f"Unexpected dataset layout: {url}")


def prepare(args):
    for pkg, version in VERSIONS.items():
        if metadata.version(pkg) != version:
            raise RuntimeError(f"Expected {pkg}=={version}; install requirements-validation.txt")
    import onnx
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    release = json.load(
        urllib.request.urlopen("https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0")
    )
    assets = {a["name"]: a for a in release["assets"]}
    for name in names(args.tasks, args.scales):
        folder = ROOT / "models" / name
        folder.mkdir(parents=True, exist_ok=True)
        weight = folder / (name + ".pt")
        asset = assets[name + ".pt"]
        digest = asset.get("digest", "").removeprefix("sha256:")
        if not digest:
            raise RuntimeError("Official asset lacks SHA256 digest")
        cached = ROOT / (name + ".pt")
        if not weight.exists() and cached.exists() and sha256(cached) == digest:
            shutil.copyfile(cached, weight)
        if not weight.exists():
            staging = folder / ".download"
            staging.mkdir(exist_ok=True)
            downloaded = staging / weight.name
            if downloaded.exists() and sha256(downloaded) != digest:
                downloaded.unlink()
            attempt_download_asset(str(downloaded), release="v8.4.0")
            if sha256(downloaded) != digest:
                raise RuntimeError(f"Official weight SHA256 mismatch: {name}")
            downloaded.replace(weight)
            staging.rmdir()
        if sha256(weight) != digest:
            raise RuntimeError(f"Official weight SHA256 mismatch: {name}")
        onnx_path = weight.with_suffix(".onnx")
        if not onnx_path.exists() or args.force:
            exported = YOLO(str(weight)).export(
                format="onnx",
                imgsz=640,
                batch=1,
                dynamic=True,
                nms=False,
                half=False,
                opset=17,
                simplify=False,
                device="cpu",
            )
            onnx_path = Path(exported)
        graph = onnx.load(str(onnx_path), load_external_data=False)
        onnx.checker.check_model(graph)
        props = {p.key: p.value for p in graph.metadata_props}
        shape = lambda value: [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]
        obj = {
            "model": name,
            "task": props["task"],
            "weight": str(weight.relative_to(ROOT)),
            "onnx": str(onnx_path.relative_to(ROOT)),
            "weight_sha256": sha256(weight),
            "onnx_sha256": sha256(onnx_path),
            "metadata": props,
            "outputs": [
                {"name": v.name, "shape": shape(v), "type": v.type.tensor_type.elem_type}
                for v in graph.graph.output
            ],
            "opset": 17,
            "source": asset["browser_download_url"],
            "official_sha256": digest,
            "export": {
                "imgsz": 640,
                "batch": 1,
                "dynamic": True,
                "nms": False,
                "half": False,
                "simplify": False,
            },
            "environment": environment(),
        }
        write_json(folder / "manifest.json", obj)
        print("Prepared", name, flush=True)
    if args.datasets in ("smoke", "full"):
        for dataset in ("coco8", "coco8-pose", "coco8-seg"):
            fetch_zip(
                f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{dataset}.zip",
                ROOT / "datasets",
                ROOT / "datasets" / dataset / "images/train",
            )
    if args.datasets == "full":
        fetch_zip(
            "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
            ROOT / "datasets/coco",
            ROOT / "datasets/coco/val2017",
        )
        fetch_zip(
            "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
            ROOT / "datasets/coco",
            ROOT / "datasets/coco/annotations/instances_val2017.json",
        )
    write_json(ROOT / "models/environment.json", environment())
