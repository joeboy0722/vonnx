"""Optional bounded range downloader for official assets; SHA256 is mandatory."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vonnx_tools.common import ROOT, sha256


def download(asset, workers=16):
    path = ROOT / "models" / Path(asset["name"]).stem / asset["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = asset["digest"].removeprefix("sha256:")
    if path.exists() and sha256(path) == digest:
        return
    total = asset["size"]
    began = time.perf_counter()
    print(f"Download {asset['name']}: {total / 1e6:.1f} MB, {workers} connections", flush=True)
    chunk = (total + workers - 1) // workers
    folder = path.parent / "ranges"
    folder.mkdir(exist_ok=True)

    def part(index):
        start = index * chunk
        end = min(total - 1, start + chunk - 1)
        piece = folder / f"{index}.part"
        if start >= total:
            return
        if piece.exists() and piece.stat().st_size == end - start + 1:
            return
        for attempt in range(3):
            started = time.perf_counter()
            try:
                request = urllib.request.Request(
                    asset["browser_download_url"], headers={"Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    expected = f"bytes {start}-{end}/{total}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Server did not honor exact range")
                    with piece.open("wb") as out:
                        shutil.copyfileobj(response, out, 1024 * 1024)
                if piece.stat().st_size != end - start + 1:
                    raise RuntimeError("Short range response")
                elapsed = time.perf_counter() - started
                print(
                    f"{asset['name']} part {index + 1}/{workers}: {piece.stat().st_size / elapsed / 1e6:.2f} MB/s",
                    flush=True,
                )
                return
            except Exception as error:
                print(f"{asset['name']} part {index + 1}: attempt {attempt + 1} failed: {error}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(part, range(workers)))
    assembled = path.with_suffix(".verified-download")
    with assembled.open("wb") as out:
        for index in range(workers):
            piece = folder / f"{index}.part"
            if piece.exists():
                with piece.open("rb") as f:
                    shutil.copyfileobj(f, out)
    if sha256(assembled) != digest:
        raise RuntimeError("Official SHA256 mismatch")
    try:
        assembled.replace(path)
    except PermissionError:
        print("Verified download retained; another writer holds target:", assembled, flush=True)
    for piece in folder.glob("*.part"):
        piece.unlink()
    folder.rmdir()
    elapsed = time.perf_counter() - began
    print(
        f"Verified official weight {asset['name']}: {elapsed:.1f}s, {total / elapsed / 1e6:.2f} MB/s (including resumed parts)",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    release = json.load(
        urllib.request.urlopen("https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0")
    )
    assets = {a["name"]: a for a in release["assets"]}
    for model in args.models:
        download(assets[model + ".pt"], args.workers)
