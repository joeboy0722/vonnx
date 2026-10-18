"""Bounded three-binary validation and benchmarks. No AP, release or 500-request mode."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vonnx_tools.common import write_json,sha256
OUT=ROOT/'runs/software_wait'
BASES={'release':ROOT/'runs/latency_opt/baseline','current':OUT/'baseline_current'}

def worker(case):
    variant=case.get('variant','candidate')
    if variant in BASES:sys.path.insert(0,str(BASES[variant]))
    import vonnx
    if case['kind']=='functional':
        from vonnx_tools.precision_check import smoke
        row=dict(case=case,status='passed',detail=smoke(case))
    elif case['kind']=='single':
        from vonnx_tools import optimize
        original=optimize.native
        def factory(c,rgb):
            engine=original(c,rgb) if c.get('graph_specialization','off')=='off' else None
            if engine is not None:return engine
            from vonnx_tools.common import manifest
            cfg=vonnx.EngineConfig();cfg.model_path=str(ROOT/manifest(c['model'])['onnx'])
            cfg.precision=c['precision'];cfg.graph_specialization='auto';cfg.model_cache_dir=str(ROOT/'runs/fp16/cache')
            cfg.cuda_graph=c.get('cuda_graph',False)
            cfg.imgsz=640;cfg.max_batch_size=c['batch'];cfg.batch_shape_policy='fixed' if c['batch']>1 else 'auto'
            cfg.max_latency_ms=c['wait'];cfg.pipeline_depth=c['depth'];cfg.concurrency=max(32,c['lanes'])
            cfg.batch_mode='latency' if c['lanes']==1 else 'throughput';cfg.collect_timings=c['timings']
            cfg.max_input_width=max(a.shape[1] for a in rgb);cfg.max_input_height=max(a.shape[0] for a in rgb)
            return vonnx.vONNXEngine(cfg)
        optimize.native=factory
        row=optimize.run(case,OUT/'single')
    else:raise ValueError('Unknown worker kind')
    row['binary_sha256']=sha256(next(Path(vonnx.__file__).parent.glob('*.pyd')))
    write_json(OUT/(case['kind']+'/'+case['id']+'.json'),row)

def execute(case):
    kind=case['kind'];file=OUT/kind/(case['id']+'.case.json');write_json(file,case)
    expected=sha256(next((BASES.get(case.get('variant'),ROOT)/'vonnx').glob('*.pyd')))
    result=OUT/kind/(case['id']+'.json')
    if result.exists():
        row=json.loads(result.read_text(encoding='utf-8'))
        if row.get('case')==case and row.get('binary_sha256')==expected and row.get('status')=='passed' and (kind!='single' or row.get('request_issuance')=='shared_quota_v2'):return
    if result.exists() and kind=='single':
        previous=OUT/'single_per_lane_quota';previous.mkdir(exist_ok=True)
        import shutil
        for suffix in ('.json','.jsonl'):
            old=result.with_suffix(suffix)
            if old.exists():shutil.copy2(old,previous/old.name)
    print(case['id'],flush=True)
    with file.with_suffix('.log').open('w',encoding='utf-8') as log:
        try:
            run=subprocess.run([sys.executable,__file__,'--worker',str(file)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        except subprocess.TimeoutExpired:
            write_json(result,dict(case=case,status='timeout'));raise
    if run.returncode:
        write_json(result,dict(case=case,status='failed',exitcode=run.returncode,log=str(file.with_suffix('.log'))))
        raise RuntimeError(f"{case['id']} failed; inspect {file.with_suffix('.log')}")
    if json.loads(result.read_text(encoding='utf-8')).get('status')!='passed':
        raise RuntimeError(f"{case['id']} failed request validation")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('functional','main','single'),default='functional')
    p.add_argument('--variant',choices=('release','current','candidate','all'),default='all')
    p.add_argument('--precision',choices=('fp32','fp16','all'),default='all')
    p.add_argument('--repeat',type=int,choices=(1,2,3),default=1)
    p.add_argument('--worker',type=Path)
    p.add_argument('--graph',action='store_true',help='Optional Graph candidate; all baseline runs remain unchanged')
    args=p.parse_args()
    if args.graph and args.phase=='main':p.error('Use xln_1920.py --cuda-graph for explicitly labeled multi-model Graph diagnostics')
    if args.worker:worker(json.loads(args.worker.read_text(encoding='utf-8')));return
    precisions=('fp32','fp16') if args.precision=='all' else (args.precision,)
    if args.phase=='functional':
        for precision in precisions:
            for model in ('yolov8s','yolov8s-pose','yolov8s-seg'):
                for shape in ([640,640],[384,640]):
                    case=dict(kind='functional',model=model,precision=precision,imgsz=shape,cuda_graph=args.graph,
                              graph_specialization='auto',batch_limits=[1,2,4,8],image_limit=2,
                              id=f'{precision}_{model}_{shape[0]}'+('_graph' if args.graph else ''))
                    if args.graph:case['batch_shape_policy']='fixed'
                    execute(case)
        return
    variants=('release','current','candidate') if args.variant=='all' else (args.variant,)
    for precision in precisions:
        for variant in variants:
            if args.phase=='single':
                for lanes in (1,16,32):
                    case=dict(kind='single',id=f'{variant}_{precision}_c{lanes}_r{args.repeat}',variant=variant,
                        backend='native',model='yolov8s',precision=precision,lanes=lanes,
                        batch=1 if lanes==1 else 8,wait=0 if lanes==1 else 2,depth=1 if lanes==1 else 2,
                        batch_shape_policy='fixed' if lanes>1 else 'auto',graph_specialization='auto' if variant=='candidate' else 'off',
                        timings=False,total=100,repeat=args.repeat)
                    if args.graph and variant=='candidate':case.update(cuda_graph=True,id=case['id']+'_graph')
                    execute(case)
            else:
                for workload in ('fixed','saturated'):
                    name=f'{variant}_{precision}_{workload}_r{args.repeat}';folder=OUT/name
                    file=folder/'result.json'
                    expected=sha256(next((BASES.get(variant,ROOT)/'vonnx').glob('*.pyd')))
                    if file.exists():
                        r=json.loads(file.read_text(encoding='utf-8'))
                        tool_hash=sha256(BASES.get(variant,ROOT)/'vonnx/precision.py')
                        if r.get('status')=='passed' and r.get('closed') and r.get('binary_sha256')==expected and r.get('warmup_per_model')==25 and r.get('preparation_tool_sha256')==tool_hash:
                            print('Reuse',name,flush=True);continue
                    folder.mkdir(parents=True,exist_ok=True)
                    cmd=[sys.executable,str(ROOT/'tools/diagnostics/xln_1920.py'),'--backend','native','--precision',precision,
                         '--warmup','25','--concurrent-only','--output',str(folder)]
                    if workload=='fixed':cmd+=['--fixed-quota']
                    if variant=='candidate':cmd+=['--specialization','auto']
                    else:cmd+=['--binary-root',str(BASES[variant])]
                    print(name,flush=True)
                    with (folder/'parent.log').open('w',encoding='utf-8') as log:
                        run=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=315)
                    if run.returncode:raise RuntimeError(f'{name} failed')

if __name__=='__main__':main()
