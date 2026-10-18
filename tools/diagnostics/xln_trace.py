"""Bounded Nsight capture, with an ASCII junction for the Windows Nsight CLI."""
import argparse
import _winapi
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument("--precision",choices=("fp32","fp16"),default="fp16")
p.add_argument("--specialize-n",action="store_true")
p.add_argument('--specialization',choices=('off','auto'),default='off')
p.add_argument('--output',type=Path,default=ROOT/'runs/xln_1920/traces')
p.add_argument('--binary-root',type=Path)
p.add_argument('--label')
p.add_argument('--case-file',type=Path,help='Capture a bounded latency_opt worker instead of the X/L/n case')
args=p.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
nsys=Path("C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe")
label=args.label or ("native_"+args.precision+("_static_n" if args.specialize_n else "")+('_auto' if args.specialization=='auto' else ''))
with tempfile.TemporaryDirectory(prefix="vonnx_xln_") as temp:
    stem=Path(temp)/label
    junction=Path(temp)/"workspace"
    _winapi.CreateJunction(str(ROOT),str(junction))
    try:
        command=[str(nsys),"profile","--trace=cuda,nvtx","--sample=none","--cpuctxsw=none",
                 "--capture-range=cudaProfilerApi","--capture-range-end=stop","--force-overwrite=true",
                 "-o",str(stem),str(junction/"venv/python.exe"),str(junction/"tools/diagnostics/xln_1920.py"),
                 "--worker","--backend","native","--precision",args.precision,"--profile",
                 '--specialization',args.specialization,'--warmup','25',
                 "--output",str(junction/out.parent.relative_to(ROOT)/("trace_"+label))]
        if args.specialize_n: command.append("--specialize-n")
        if args.binary_root:command.extend(['--binary-root',str(junction/args.binary_root.resolve().relative_to(ROOT))])
        if args.case_file:
            command=command[:command.index(str(junction/'venv/python.exe'))]+[
                str(junction/'venv/python.exe'),str(junction/'tools/diagnostics/latency_opt.py'),
                '--worker',str(junction/args.case_file.resolve().relative_to(ROOT))]
        with (out/(label+".log")).open("w",encoding="utf-8") as log:
            run=subprocess.run(command,cwd=temp,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            if run.returncode:raise RuntimeError(f"Capture failed: {run.returncode}")
            run=subprocess.run([str(nsys),"export","--type=sqlite","--force-overwrite=true",
                                "-o",str(stem.with_suffix(".sqlite")),str(stem.with_suffix(".nsys-rep"))],
                               stdout=log,stderr=subprocess.STDOUT,timeout=60)
            if run.returncode:raise RuntimeError("Export failed")
        for suffix in (".sqlite",".nsys-rep"):
            shutil.copy2(stem.with_suffix(suffix),out/(label+suffix))
    finally:
        junction.rmdir()
print(out/label)
