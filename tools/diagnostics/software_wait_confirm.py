"""Explicit short confirmation only: 100 requests, never 500 or AP."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/software_wait';OUT.mkdir(exist_ok=True)
script=ROOT/'tools/diagnostics/software_wait.py'
for repeat in (2,3):
    variants=('current','candidate') if repeat==2 else ('candidate','current')
    for variant in variants:
        label=f'main_{variant}_fp16_r{repeat}'
        print(label,flush=True)
        with (OUT/(label+'.log')).open('w',encoding='utf-8') as log:
            run=subprocess.run([sys.executable,str(script),'--phase','main','--variant',variant,
                '--precision','fp16','--repeat',str(repeat)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if run.returncode:raise RuntimeError(label+' failed')
    for variant in variants:
        label=f'single_{variant}_r{repeat}';print(label,flush=True)
        with (OUT/(label+'.log')).open('w',encoding='utf-8') as log:
            run=subprocess.run([sys.executable,str(script),'--phase','single','--variant',variant,
                '--repeat',str(repeat)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if run.returncode:raise RuntimeError(label+' failed')
    label=f'single_graph_r{repeat}';print(label,flush=True)
    with (OUT/(label+'.log')).open('w',encoding='utf-8') as log:
        run=subprocess.run([sys.executable,str(script),'--phase','single','--variant','candidate','--graph',
            '--repeat',str(repeat)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if run.returncode:raise RuntimeError(label+' failed')
