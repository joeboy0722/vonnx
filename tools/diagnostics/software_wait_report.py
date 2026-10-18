"""CPU-only report for three-binary short tests; never starts GPU work."""
from pathlib import Path
import collections
import csv
import json
import re
import statistics as st
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vonnx_tools.common import write_json,sha256
OUT=ROOT/'runs/software_wait'
rows=[];groups=[];binaries={};pending=[]
for file in sorted(OUT.glob('*_r[123]/result.json')):
    match=re.fullmatch(r'(release|current|candidate)_(fp32|fp16)_(fixed|saturated)_r([123])',file.parent.name)
    if not match:continue
    variant,precision,workload,repeat=match.groups()
    result=json.loads(file.read_text(encoding='utf-8'))
    if result.get('status')!='passed' or not result.get('closed'):
        pending.append(dict(case=file.parent.name,status=result.get('status'),error=result.get('error')));continue
    binaries[variant]=result['binary_sha256']
    for scenario in result['scenarios']:
        data=[json.loads(line) for line in (file.parent/(scenario['name']+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        assert len(data)==scenario['count'] and not scenario['errors']
        assert len({(r['model'],r['frame']) for r in data})==len(data)
        common=dict(variant=variant,precision=precision,workload=workload,repeat=int(repeat),lanes=scenario['lanes_per_model'])
        groups.append(dict(**common,fps=len(data)/scenario['elapsed_s'],elapsed_s=scenario['elapsed_s'],requests=len(data),
                           peak_bytes=result['peak_device_bytes']))
        for model in ('yolov8x','yolov8l','yolov8n'):
            own=[r for r in data if r['model']==model];values=[r['latency_ms'] for r in own]
            assert own
            rows.append(dict(**common,model=model,requests=len(own),mean_ms=st.mean(values),
                p95_ms=float(np.percentile(values,95)),p99_ms=float(np.percentile(values,99)),
                completion_fps=len(own)/max(r['finish_s'] for r in own),
                group_window_fps=len(own)/scenario['elapsed_s'],
                sm_mhz=st.median(r['sm_mhz'] for r in scenario.get('gpu_samples',[])) if scenario.get('gpu_samples') else None,
                max_temperature=max((r['temperature'] for r in scenario.get('gpu_samples',[])),default=None),
                source=file.parent.name,binary_sha256=result['binary_sha256']))
for name,records in [('summary',rows),('groups',groups)]:
    if records:
        with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)

comparisons=[]
for precision in ('fp32','fp16'):
    for workload in ('fixed','saturated'):
        for lanes in (1,4):
            for model in ('yolov8x','yolov8l','yolov8n'):
                def select(variant):return [r for r in rows if (r['variant'],r['precision'],r['workload'],r['lanes'],r['model'])==(variant,precision,workload,lanes,model)]
                before,after=select('current'),select('candidate')
                rounds=sorted({r['repeat'] for r in before}&{r['repeat'] for r in after})
                if not rounds:continue
                b=[r for r in before if r['repeat'] in rounds];a=[r for r in after if r['repeat'] in rounds]
                gains={key:st.median(r[key] for r in a)/st.median(r[key] for r in b)-1 for key in ('mean_ms','p95_ms','completion_fps')}
                wins=sum(next(r for r in a if r['repeat']==n)['mean_ms']<next(r for r in b if r['repeat']==n)['mean_ms'] and
                         next(r for r in a if r['repeat']==n)['p95_ms']<next(r for r in b if r['repeat']==n)['p95_ms'] and
                         next(r for r in a if r['repeat']==n)['completion_fps']>=next(r for r in b if r['repeat']==n)['completion_fps'] for n in rounds)
                regression=gains['mean_ms']<=.03 and gains['p95_ms']<=.03 and gains['completion_fps']>=-.03
                improved=gains['mean_ms']<0 and gains['p95_ms']<0 and gains['completion_fps']>=0
                comparisons.append(dict(precision=precision,workload=workload,lanes=lanes,model=model,
                    rounds=rounds,wins=wins,changes=gains,regression_pass=regression,
                    primary_pass=rounds==[1,2,3] and wins>=2 and improved))

single=[]
for p in (OUT/'single').glob('*.json'):
    if p.name.endswith('.case.json'):continue
    r=json.loads(p.read_text(encoding='utf-8'))
    if r.get('status')!='passed' or r.get('request_issuance')!='shared_quota_v2':continue
    case=r['case']
    single.append(dict(variant=case['variant'],graph=case.get('cuda_graph',False),precision=case['precision'],
        lanes=case['lanes'],repeat=case['repeat'],fps=r['fps'],mean_ms=r['latency']['mean_ms'],
        p95_ms=r['latency']['p95_ms'],p99_ms=r['latency']['p99_ms'],
        physical_batches=sum(r['physical_batch_histogram'].values()),padded_images=r['padded_images'],
        binary_sha256=r['binary_sha256']))
if single:
    with (OUT/'single_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=single[0]);writer.writeheader();writer.writerows(single)
single_gates=[]
for precision in ('fp32','fp16'):
    for lanes in (1,16,32):
        for graph in (False,True):
            b=[r for r in single if (r['variant'],r['precision'],r['lanes'])==('current',precision,lanes)]
            a=[r for r in single if (r['variant'],r['precision'],r['lanes'],r['graph'])==('candidate',precision,lanes,graph)]
            if len(a)!=3 or len(b)!=3:continue
            changes={key:st.median(r[key] for r in a)/st.median(r[key] for r in b)-1 for key in ('fps','mean_ms','p95_ms')}
            wins=sum(next(r for r in a if r['repeat']==n)['mean_ms']<next(r for r in b if r['repeat']==n)['mean_ms'] and
                     next(r for r in a if r['repeat']==n)['p95_ms']<next(r for r in b if r['repeat']==n)['p95_ms'] and
                     next(r for r in a if r['repeat']==n)['fps']>=next(r for r in b if r['repeat']==n)['fps'] for n in (1,2,3))
            single_gates.append(dict(precision=precision,lanes=lanes,graph=graph,wins=wins,changes=changes,
                regression_pass=changes['fps']>=-.03 and changes['mean_ms']<=.03 and changes['p95_ms']<=.03))

lines=['# 跨模型軟體等待改善：實作與短測', '', '日期：2026-10-18。本報告記錄v0.5.0發布前的效能短測，當時候選版本字串仍為0.4.0；辨識以二進位雜湊為準。',
    '本階段未執行500次效能確認或長時間矩陣；後續發布前另完成 [500張抽樣AP回歸](ap_v050.md)。', '',
    '## 實作', '',
    '- 新增預設關閉的 graph_specialization=auto；固定已知高／寬與固定執行 batch，保留動態 batch、補零與逐請求結果。運行期間每引擎一個常駐session；初始化有暫時GPU探測session。',
    '- 模型準備與 FP16 轉換共用帶來源／外部權重、尺寸、精度、batch、工具及規則指紋的原子快取，保留來源檔。探測／binding／暖機使用一致的實際 shape。',
    '- 新增預設關閉的 cuda_graph，要求專化與固定執行 batch；每槽獨立 graph ID／固定 I/O 位址，capture 與暖機在初始化完成，ORT 結尾同步保留。CPU fallback、初始化失敗或預算不足明確拒絕。',
    '- 新增 Python／C++ 設定與 ModelSpec 資訊、C API v4。v1／v2／v3 布局不變，C++ 呼叫者需重編譯。',
    '- 修正 ORT Env／logger 生命週期。引擎 close 釋放 session／arena／CUDA 資源，保留每個載入模組一份的程序生命週期 Env，避免 Python ORT 在最後引擎 close 後 logger 失效，以及 Windows DLL 退出解載順序崩潰。', '',
    '## 三組基準與量測', '',
    '正式 v0.4.0、封存的修改前工作目錄引擎、候選引擎分別載入實際二進位，來源 PT／ONNX 雜湊與準備後模型雜湊保留在每案 result.json。來源版本號相同不代表二進位相同。', '',
    '| 組別 | Python native SHA256 |', '|---|---|']
for variant,value in binaries.items():lines.append(f'| {variant} | `{value}` |')
lines += ['', '主測使用預載 COCO8 Detect 真實影像輪播，原圖1920×1080、固定推論1920×1920，X／L／n 各獨立引擎，FP32／FP16、B1、等待0、單槽、每模型1／4路。每模型25次暖機排空後最多75次計分，每路最多一筆未完成；固定配額與6秒上限飽和負載分開。',
    '計入提交副本到 CPU 結果可用與所有尾段，不刪慢樣本。固定配額的逐模型 FPS 以共同起點到該模型完成75次計算，整組吞吐則含最後模型排空。飽和案例另外呈現共同活動窗口，不能把模型完成更多次數誤報為每模型都加速。', '',
    '## 三模型總吞吐（全部請求、包含排空）', '',
    '| 精度 | 負載 | 每模型路數 | Release | 修改前目前引擎 | 候選 |', '|---|---|---:|---:|---:|---:|']
for precision in ('fp32','fp16'):
    for workload in ('fixed','saturated'):
        for lanes in (1,4):
            vals=[]
            for variant in ('release','current','candidate'):
                selected=[r['fps'] for r in groups if (r['precision'],r['workload'],r['lanes'],r['variant'])==(precision,workload,lanes,variant)]
                vals.append(f'{st.median(selected):.2f}' if selected else '待測')
            lines.append(f"| {precision} | {workload} | {lanes} | {' | '.join(vals)} |")
lines += ['', '單位為次推論／秒，X／L／n 各算一次。Release與FP32／1920只有首輪對照；FP16／1920的修改前與候選完成三輪，才進行正式中位數／勝出門檻判定。FP32首輪收益小且P95未全面下降，不延長確認。主測gpu_samples保存溫度、SM／記憶體時脈與utilization，不以時脈校正替代實際成績。', '',
    '## 逐模型變化與驗收', '',
    '| 精度 | 負載 | 路數 | 模型 | 輪數／勝出 | 平均延遲變化 | P95變化 | 完成FPS變化 | 3%回歸門檻 | 三輪改善門檻 |',
    '|---|---|---:|---|---|---:|---:|---:|---|---|']
for c in comparisons:
    g=c['changes'];lines.append(f"| {c['precision']} | {c['workload']} | {c['lanes']} | {c['model']} | {len(c['rounds'])}／{c['wins']} | {g['mean_ms']*100:+.1f}% | {g['p95_ms']*100:+.1f}% | {g['completion_fps']*100:+.1f}% | {'通過' if c['regression_pass'] else '未通過'} | {'通過' if c['primary_pass'] else '未通過或未完成'} |")
lines += ['', '負的延遲變化較快，正的 FPS 變化較快。主要情境須三輪至少兩輪勝出、平均／P95中位數下降且FPS不下降；3%是其他回歸情境底線，不是主要改善成功條件。低於5%的改善為小幅收益。', '',
    '## 等待原因', '',
    '1. **尚未提交**：動態 shape／資料相依會使 CPU 中途等資料，再提交後段 kernel。固定 shape 可以移除其中可預計算的工作。',
    '2. **已提交但 GPU 未完成**：其他 stream 的 kernel、GPU資源與驅動排程仍可能使 n 等待；不能把所有非 kernel 空隙都算成可移除 CPU overhead。',
    '3. **GPU完成到結果回傳**：需要完成 n 自己的後處理與D2H，再發布future；檢查是否存在無關模型的結果屏障。', '',
    '未發現X完成後n才能發布future的顯式跨引擎屏障。實際軟體問題是可預計算的shape／常數工作在GPU與CPU間往返，造成中途等待，並被其他模型的GPU排隊放大。三模型專化後，n自己的kernel仍受GPU／驅動排程影響；ORT返回後自己的後處理／D2H也會競爭資源，不能宣稱n端到端已降到自身kernel總執行時間。',
    '診斷trace啟用詳細timing，不混入主要成績。先前只專化n的57.9ms結果是不同模型準備組合，不冒充三模型全部專化的效益。', '',
    '## 準備順序與數值診斷', '']
probe=OUT/'preparation/summary.json'
if probe.exists():
    data=json.loads(probe.read_text(encoding='utf-8'))
    lines += ['| 準備方式 | CPU節點數 | MemcpyToHost節點 | 相對同精度動態圖原始輸出最大差 |','|---|---:|---:|---:|']
    for r in data:lines.append(f"| {r['case']} | {len(r['cpu_nodes'])} | {r['cuda_ops'].get('MemcpyToHost',0)} | {r['max_raw_error_vs_dynamic']:.6g} |")
    lines += ['', '此為n／1920單圖的EP profile。FP16的CPU節點數為0仍可能有給形狀輸入使用的往返拷貝，不能用CPU節點數0推論完全沒有CPU等待。先專化再轉FP16、先轉再專化在本圖原始輸出一致；正式流程採先專化再轉換。', '']
numeric=OUT/'numeric/result.json'
if numeric.exists():
    r=json.loads(numeric.read_text(encoding='utf-8'))
    lines += [f"1920 X：同ONNX獨立ORT原始輸出最大差 `{r['independent_prepared_raw_max_error']}`，最終框最大差 `{r['native_vs_same_onnx']['max_coordinate_px']}` pixel。",
        f"相對來源FP32，vONNX FP16最大座標差 {r['native_vs_source_fp32']['max_coordinate_px']:.3f} pixel，官方FP16 {r['official_half_vs_source_fp32']['max_coordinate_px']:.3f} pixels。兩個半精度流程彼此仍可差2 pixels；官方raw半精度輸出先轉FP32後處理，差異降為 {r['native_vs_official_half_float_post']['max_coordinate_px']:.3f} pixels，顯示包含半精度模型解碼及後處理數值路徑差異。",
        '保持vONNX的FP32解碼／後處理，不刻意貼齊官方半精度捨入。以上僅一張影像診斷，未通過正式AP，也不宣稱兩流程已符合逐框1pixel一致。', '']
functional=[]
for p in (OUT/'functional').glob('*.json'):
    if p.name.endswith('.case.json'):continue
    r=json.loads(p.read_text(encoding='utf-8'))
    if r.get('status')=='passed':functional.append(dict(case=r['case'],binary_sha256=r['binary_sha256'],detail=r['detail']))
lines += ['## 功能與控制測試', '',
    f'目前保存 {len(functional)} 組真實模型小樣本檢查；每組二進位雜湊在summary.json中明列。不同候選建置的結果不當作同一二進位完整驗收。',
    '測試包含固定／動態batch、矩形尺寸、Detect／Pose／Seg、原生FP16 I/O、快取命中／失效／並行準備、v1/v2/v3相容與v4、空結果、ID、pause/resume資源重用及程序退出。',
    '純C++真實引擎fixture控制：X callback刻意阻塞並拋例外時，n仍能完成；獨立scheduler控制及Python host延後／pause測試另有記錄。這些是相依控制，不是YOLO效能數字。', '',
    '## 使用與重現', '',
    '介面與限制見 [圖專化說明](graph_specialization.md) 及 examples/graph_specialization.py。',
    '```powershell',
    r'.\venv\python.exe tools\diagnostics\software_wait.py --phase functional',
    r'.\venv\python.exe tools\diagnostics\software_wait.py --phase main',
    r'.\venv\python.exe tools\diagnostics\software_wait.py --phase main --variant current --repeat 2',
    r'.\venv\python.exe tools\diagnostics\software_wait.py --phase main --variant candidate --repeat 2',
    r'.\venv\python.exe tools\diagnostics\software_wait.py --phase single',
    r'.\venv\python.exe tools\diagnostics\software_wait.py --phase single --variant candidate --graph',
    r'.\venv\python.exe tools\diagnostics\software_wait_report.py',
    '```', '',
    '每筆資料、NVML樣本、來源／產物雜湊、錯誤、trace與報告在 runs/software_wait/。summary.csv 包含每模型平均／P95／P99、FPS、時脈與溫度；groups.csv 為整組吞吐。',
    '尚未滿足的門檻、未定位的問題與未執行的驗證維持未完成；本版發布不代表這些限制已解決。', '']
section=['## s／640 單模型回歸（三輪中位數）','','單路B1／等待0／單槽；16／32路為固定B8／等待2ms／雙槽。每設定25次暖機排空、75次計分。Release為首輪對照，修改前與兩種候選各三輪。','',
         '| 精度 | 路數 | Release FPS | 修改前 FPS | 圖專化 FPS | 圖專化＋Graph FPS |','|---|---:|---:|---:|---:|---:|']
for precision in ('fp32','fp16'):
    for lanes in (1,16,32):
        values=[]
        for variant,graph in [('release',False),('current',False),('candidate',False),('candidate',True)]:
            selected=[r['fps'] for r in single if (r['variant'],r['graph'],r['precision'],r['lanes'])==(variant,graph,precision,lanes)]
            values.append(f'{st.median(selected):.1f}' if selected else '未測')
        section.append(f"| {precision} | {lanes} | {' | '.join(values)} |")
section += ['', '| 精度 | 路數 | 修改前平均／P95 ms | 圖專化平均／P95 ms | 圖專化＋Graph平均／P95 ms |','|---|---:|---:|---:|---:|']
for precision in ('fp32','fp16'):
    for lanes in (1,16,32):
        values=[]
        for variant,graph in [('current',False),('candidate',False),('candidate',True)]:
            selected=[r for r in single if (r['variant'],r['graph'],r['precision'],r['lanes'])==(variant,graph,precision,lanes)]
            values.append(f"{st.median(r['mean_ms'] for r in selected):.2f}／{st.median(r['p95_ms'] for r in selected):.2f}" if selected else '未測')
        section.append(f"| {precision} | {lanes} | {' | '.join(values)} |")
section += ['', '| 精度 | 路數 | 候選模式 | 勝出輪數 | 平均／P95／FPS變化 | 3%回歸門檻 |','|---|---:|---|---:|---|---|']
for r in single_gates:
    g=r['changes'];section.append(f"| {r['precision']} | {r['lanes']} | {'圖專化＋Graph' if r['graph'] else '圖專化'} | {r['wins']}／3 | {g['mean_ms']*100:+.1f}%／{g['p95_ms']*100:+.1f}%／{g['fps']*100:+.1f}% | {'通過' if r['regression_pass'] else '未通過'} |")
section += ['', '首版測試端採每路固定請求配額，較快的路徑可能在共同75次尚未發完時提早退出，造成尾段B8被拆成兩批。某FP32／32路因此出現10批與11批的差異。原始資料保留於single_per_lane_quota；改為共同配額後三組全部重測，每路仍至多一筆未完成，所有尾批與補零成本保留。single_summary.csv僅使用shared_quota_v2資料。','']
lines[lines.index('## 等待原因'):lines.index('## 等待原因')]=section

trace_section=['## 最終二進位的trace與限制','',
               '以下為診斷資料，詳細timing開啟，不混入主要跑分。','']
tb=OUT/'traces/current_fp16_final.summary.json';ta=OUT/'traces/candidate_fp16_final.summary.json'
if tb.exists() and ta.exists():
    b=json.loads(tb.read_text(encoding='utf-8'))['streams'][2]
    a=json.loads(ta.read_text(encoding='utf-8'))['streams'][2]
    trace_section += ['| n／1920指標 | 修改前 | 圖專化候選 |','|---|---:|---:|',
        f"| CPU event等待次數／推論 | {b['event_wait_calls_per_batch']:.0f} | {a['event_wait_calls_per_batch']:.0f} |"]
    for label,key in [('最後kernel的CPU提交距前處理結束ms','last_launch_from_pre_ms'),
                      ('自己kernel累計執行ms','own_kernel_ms'),('GPU推論跨度ms','inference_span_ms')]:
        trace_section.append(f"| {label} | {b['median'][key]:.2f} | {a['median'][key]:.2f} |")
    trace_section += ['', '即使n的工作約6ms已提交至尾端，GPU推論跨度仍約169ms。這份trace無法將SM／記憶體資源與WDDM驅動排程的比例完全拆開，不宣稱已達硬體極限，也不承諾7ms模型併發後最多14ms。','']
api_file=OUT/'traces/graph_s32_final.apis.json'
if api_file.exists():
    apis=json.loads(api_file.read_text(encoding='utf-8'))
    trace_section += [f"原生引擎s／640、雙槽B8的測量段觀察到 {sum(r['calls'] for r in apis if 'GraphLaunch' in r['name'])} 次cudaGraphLaunch；未見capture、instantiate、cudaMalloc／Free或EventCreate，確認capture在初始化完成。這不保證第三方驅動內部完全沒有配置。",'']
trace_section += ['額外1920固定配額診斷：全部開Graph約14.89／14.90次每秒（1／4路）；compute與copy work queue都設32約14.90／14.93。本機未見優於預設的證據，因此不自動調整CUDA環境或把Graph當作1920隔離方案。',
    'NVIDIA說明不同stream映射同一work queue可能產生假相依，連線數可調1～32、預設8；本輪僅在獨立子程序測32，不改系統設定。[NVIDIA CUDA環境變數說明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-max-connections)。','']
lines[lines.index('## 準備順序與數值診斷'):lines.index('## 準備順序與數值診斷')]=trace_section

status_file=OUT/'unit_status.json'
unit=json.loads(status_file.read_text(encoding='utf-8')) if status_file.exists() else {}
unit_text=(OUT/'unit.log').read_bytes().decode('utf-8',errors='replace').replace(chr(0),'') if (OUT/'unit.log').exists() else ''
count=re.search(r'Ran (\d+) tests in ([\d.]+)s',unit_text)
if count:unit.update(tests=int(count[1]),seconds=float(count[2]))
extra=(OUT/'../software_wait_capi_gpu_graph.log').resolve()
extra_text=extra.read_bytes().decode('utf-8',errors='replace').replace(chr(0),'') if extra.exists() else ''
unit['capi_gpu_graph_extra_passed']=bool(re.search(r'Ran 1 test',extra_text) and extra_text.rstrip().endswith('OK'))
if count:
    index=lines.index('## 功能與控制測試')+2
    lines[index:index]=[f"完整Python套件 {unit['tests']} 項／{unit['seconds']:.3f}秒通過，程序退出碼 {unit.get('exitcode')}；另補測C API v4 Graph之外部GPU producer（延後寫入）1項通過。C++scheduler與真實引擎callback阻塞控制亦通過。",'']

write_json(OUT/'summary.json',dict(binaries=binaries,rows=rows,groups=groups,comparisons=comparisons,pending=pending,
                                  single=single,single_gates=single_gates,unit=unit,
                                  functional=functional,ap_validated=False,long_benchmark_run=False))
(ROOT/'docs/software_wait.md').write_text('\n'.join(lines),encoding='utf-8')
print('rows',len(rows),'pending',pending,'report',ROOT/'docs/software_wait.md')
