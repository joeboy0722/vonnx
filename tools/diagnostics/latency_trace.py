"""Four bounded Nsight traces of the archived baseline; outputs stay in runs/."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import _winapi

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vonnx_tools.common import write_json

out=ROOT/'runs/latency_opt/traces';out.mkdir(parents=True,exist_ok=True)
nsys=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe')
if not nsys.exists():raise RuntimeError('Existing Nsight Systems CLI not found')
for precision in ('fp32','fp16'):
    for lanes in (1,32):
        label=f'{precision}_s{lanes}_r1'
        source=ROOT/f'runs/latency_opt/baseline/quick/{label}.case.json'
        case=json.loads(source.read_text(encoding='utf-8'));case.update(phase='diagnostic',timings=True,profile=True)
        folder=ROOT/'runs/latency_opt/baseline/diagnostic';folder.mkdir(exist_ok=True)
        file=folder/(label+'.case.json');write_json(file,case)
        with tempfile.TemporaryDirectory(prefix='vonnx_trace_') as temp:
            stem=Path(temp)/label
            # Nsight CLI cannot parse non-ASCII application arguments on this host.
            junction=Path(temp)/'workspace'
            _winapi.CreateJunction(str(ROOT),str(junction))
            command=[str(nsys),'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',
                '-o',str(stem),str(junction/'venv/python.exe'),str(junction/'tools/diagnostics/latency_opt.py'),
                '--worker',str(junction/file.relative_to(ROOT))]
            print('Trace',label,flush=True)
            with (out/(label+'.log')).open('w',encoding='utf-8') as log:
                run=subprocess.run(command,cwd=temp,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                if run.returncode:raise RuntimeError(f'nsys exit {run.returncode}: {label}')
                report=stem.with_suffix('.nsys-rep')
                if not report.exists():raise RuntimeError('Nsight omitted report')
                export=subprocess.run([str(nsys),'export','--type=sqlite','--force-overwrite=true','-o',str(stem.with_suffix('.sqlite')),str(report)],
                                      stdout=log,stderr=subprocess.STDOUT,timeout=60)
                if export.returncode:raise RuntimeError('Nsight export failed')
            for suffix in ('.nsys-rep','.sqlite'):shutil.copy2(stem.with_suffix(suffix),out/(label+suffix))
            junction.rmdir()  # Remove only the junction, never recurse into the checkout.
        print('Captured',label,flush=True)
