"""Package already built Windows binaries; does not build or run benchmarks."""
from pathlib import Path
import hashlib
import shutil
import zipfile
import re
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist'
OUT.mkdir(exist_ok=True)
version=re.search(r'__version__ = "([^"]+)"',(ROOT/'vonnx/__init__.py').read_text(encoding='utf-8')).group(1)
gate=json.loads((ROOT/'docs/ap_v050.json').read_text(encoding='utf-8'))
assert gate['passed'] and len(gate['models']) == 18, 'Incomplete AP release gate'
for source, fingerprint in [
    ('vonnx/vonnx_py.cp312-win_amd64.pyd', 'binary_sha256'),
    ('vonnx/precision.py', 'converter_sha256'),
    ('vonnx_standalone_sdk/vonnx.dll', 'sdk_dll_sha256'),
]:
    assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == gate[fingerprint], 'AP gate is stale: '+source
wheel = OUT / f'vonnx-{version}-cp312-cp312-win_amd64.whl'
with zipfile.ZipFile(wheel) as package:
    assert package.read('vonnx/vonnx_py.cp312-win_amd64.pyd') == (ROOT / 'vonnx/vonnx_py.cp312-win_amd64.pyd').read_bytes()
    assert package.read('vonnx/__init__.py') == (ROOT / 'vonnx/__init__.py').read_bytes()
    assert package.read('vonnx/precision.py') == (ROOT / 'vonnx/precision.py').read_bytes()
sdk = OUT / f'vonnx-{version}-windows-x64-sdk.zip'
with zipfile.ZipFile(sdk, 'w', zipfile.ZIP_DEFLATED) as package:
    for source, target in [
        ('vonnx_standalone_sdk/vonnx.dll', 'vonnx.dll'),
        ('build/temp.win-amd64-cpython-312/Release/vonnx.lib', 'vonnx.lib'),
        ('include/vonnx_c_api.h', 'vonnx_c_api.h'),
        ('vonnx/precision.py', 'precision.py'),
        ('examples/c_api.cpp', 'c_api.cpp'),
        ('examples/quick_start.py', 'quick_start.py'),
        ('examples/graph_specialization.py', 'graph_specialization.py'),
        ('LICENSE', 'LICENSE'), ('THIRD_PARTY_NOTICES.md', 'THIRD_PARTY_NOTICES.md'),
        ('docs/fp16.md', 'fp16.md'),
        ('docs/release.md', 'USAGE.md'),
        ('docs/graph_specialization.md', 'graph_specialization.md'),
        ('docs/software_wait.md', 'software_wait.md'),
        ('docs/ap_v050.md', 'ap_v050.md'),
        ('docs/ap_v050.json', 'ap_v050.json'),
        ('docs/ap_v050.md', 'AP_RESULTS.md'),
        ('docs/ap_v050.json', 'AP_RESULTS.json'),
    ]:
        package.write(ROOT / source, target)
shutil.copyfile(ROOT / 'docs/release.md', OUT / 'USAGE.md')
shutil.copyfile(ROOT / 'docs/ap_v050.json', OUT / 'AP_RESULTS.json')
files = [wheel, sdk, OUT / 'USAGE.md', OUT / 'AP_RESULTS.json']
(OUT / 'SHA256SUMS.txt').write_text(''.join(
    f'{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.name}\n' for file in files
), encoding='utf-8')
for file in files:
    print(file.name, file.stat().st_size)
