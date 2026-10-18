"""Package already built Windows binaries; does not build or run benchmarks."""
from pathlib import Path
import hashlib
import shutil
import zipfile
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist'
OUT.mkdir(exist_ok=True)
version=re.search(r'__version__ = "([^"]+)"',(ROOT/'vonnx/__init__.py').read_text(encoding='utf-8')).group(1)
wheel = OUT / f'vonnx-{version}-cp312-cp312-win_amd64.whl'
with zipfile.ZipFile(wheel) as package:
    assert package.read('vonnx/vonnx_py.cp312-win_amd64.pyd') == (ROOT / 'vonnx/vonnx_py.cp312-win_amd64.pyd').read_bytes()
    assert package.read('vonnx/__init__.py') == (ROOT / 'vonnx/__init__.py').read_bytes()
sdk = OUT / f'vonnx-{version}-windows-x64-sdk.zip'
with zipfile.ZipFile(sdk, 'w', zipfile.ZIP_DEFLATED) as package:
    for source, target in [
        ('vonnx_standalone_sdk/vonnx.dll', 'vonnx.dll'),
        ('build/temp.win-amd64-cpython-312/Release/vonnx.lib', 'vonnx.lib'),
        ('include/vonnx_c_api.h', 'vonnx_c_api.h'),
        ('vonnx/precision.py', 'precision.py'),
        ('examples/c_api.cpp', 'c_api.cpp'),
        ('examples/quick_start.py', 'quick_start.py'),
        ('LICENSE', 'LICENSE'), ('THIRD_PARTY_NOTICES.md', 'THIRD_PARTY_NOTICES.md'),
        ('docs/fp16.md', 'FP16.md'),
        ('docs/fp16.md', 'USAGE.md'),
        ('docs/ap_fp16.md', 'AP_RESULTS.md'),
        ('docs/ap_fp16.json', 'AP_RESULTS.json'),
    ]:
        package.write(ROOT / source, target)
shutil.copyfile(ROOT / 'docs/fp16.md', OUT / 'USAGE.md')
shutil.copyfile(ROOT / 'docs/ap_fp16.json', OUT / 'AP_RESULTS.json')
files = [wheel, sdk, OUT / 'USAGE.md', OUT / 'AP_RESULTS.json']
(OUT / 'SHA256SUMS.txt').write_text(''.join(
    f'{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.name}\n' for file in files
), encoding='utf-8')
for file in files:
    print(file.name, file.stat().st_size)
