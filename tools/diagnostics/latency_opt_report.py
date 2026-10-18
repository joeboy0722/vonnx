"""CPU-only evidence/report for bounded latency optimization; never launches tests."""
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vonnx_tools.common import build_fingerprint,write_json
OUT=ROOT/'runs/latency_opt'


def main():
    def read(path):return json.loads(path.read_text(encoding='utf-8'))
    rows=[]
    baseline_hash=read(OUT/'baseline/provenance.json')['binary_sha256']
    for precision in ('fp32','fp16'):
        for lanes in (1,16,32):
            name=f'{precision}_s{lanes}_r1'
            before=read(OUT/'baseline/quick'/(name+'.json'))
            after=read(OUT/'candidate/quick'/(name+'.json'))
            assert before['binary_sha256']==baseline_hash and after['binary_sha256']==build_fingerprint()
            assert before['prepared_model']['sha256']==after['prepared_model']['sha256']
            assert {k:v for k,v in before['case'].items() if k!='variant'}=={k:v for k,v in after['case'].items() if k!='variant'}
            assert before['status']==after['status']=='passed' and before['success']==after['success']==75
            row=dict(precision=precision,lanes=lanes,before_fps=before['fps'],after_fps=after['fps'],
                before_mean_ms=before['latency']['mean_ms'],after_mean_ms=after['latency']['mean_ms'],
                before_p95_ms=before['latency']['p95_ms'],after_p95_ms=after['latency']['p95_ms'],
                mean_reduction_percent=100*(1-after['latency']['mean_ms']/before['latency']['mean_ms']),
                p95_reduction_percent=100*(1-after['latency']['p95_ms']/before['latency']['p95_ms']),
                p99_increase_percent=100*(after['latency']['p99_ms']/before['latency']['p99_ms']-1),
                fps_change_percent=100*(after['fps']/before['fps']-1),
                before_physical_batch=before['physical_batch_histogram'],after_physical_batch=after['physical_batch_histogram'])
            for variant,data in (('before',before),('after',after)):
                samples=data['gpu_samples']
                row[variant+'_gpu']=dict(sm_mhz_median=statistics.median(s['sm_mhz'] for s in samples),
                    temperature_min=min(s['temperature'] for s in samples),temperature_max=max(s['temperature'] for s in samples),
                    peak_memory=max(s['memory_used'] for s in samples))
            rows.append(row)
    trace=read(OUT/'traces/summary.json')
    host=[]
    names=('api_entry','admitted','copy_done','enqueued','dequeued','slot_ready','compute_begin',
           'ort_enter','ort_exit','d2h_ready','cpu_ready','future_publish')
    for file in sorted((OUT/'candidate/diagnostic').glob('*.jsonl')):
        record=read(file.with_suffix('.json'));assert record['binary_sha256']==build_fingerprint()
        requests=[json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
        assert len(requests)==75 and len({r['frame_id'] for r in requests})==75
        times=[r['diagnostic_trace'] for r in requests]
        for t in times:assert [t[n] for n in names]==sorted(t[n] for n in names)
        host.append(dict(case=file.stem,
            median_copy_ms=statistics.median(t['copy_done']-t['admitted'] for t in times),
            median_ready_slot_wait_ms=statistics.median(t['compute_begin']-t['slot_ready'] for t in times),
            median_ort_host_ms=statistics.median(t['ort_exit']-t['ort_enter'] for t in times),
            median_completion_ms=statistics.median(t['future_publish']-t['d2h_ready'] for t in times),
            median_completion_share_percent=100*statistics.median((t['future_publish']-t['d2h_ready'])/
                (t['future_publish']-t['compute_begin']) for t in times)))
    checks=read(OUT/'functional/summary.json');assert len(checks)==8
    assert all(c['status']=='passed' and c['binary_sha256']==build_fingerprint() for c in checks)
    graph=[*read(OUT/'graph_probe/fp16.json'),*read(OUT/'graph_probe/fp32.json')]
    quick_improves=all(r['mean_reduction_percent']>0 and r['p95_reduction_percent']>0 and
        r['fps_change_percent']>=-3 for r in rows if r['lanes']>1)
    candidate_threshold=all(r['mean_reduction_percent']>=5 and r['p95_reduction_percent']>=5 and
        r['fps_change_percent']>=-3 and r['p99_increase_percent']<=5 for r in rows if r['lanes']>1)
    result=dict(status='not_accepted',speed_acceptance_passed=False,confirmation_started=False,
        quick_all_four_improve=quick_improves,quick_all_four_meet_5_percent=candidate_threshold,
        original_binary_sha256=baseline_hash,candidate_binary_sha256=build_fingerprint(),
        environment=read(OUT/'baseline/provenance.json')['environment'],quick=rows,
        host_diagnostics=host,trace=trace,graph_probe=graph,
        functional=[dict(case=c['case'],status=c['status'],prepared_model=c['detail']['prepared_model'],
                         max_raw_error=c['detail']['max_raw_error']) for c in checks])
    # This report deliberately cannot certify acceptance without three confirmation repeats.
    write_json(OUT/'SUMMARY.json',result)
    write_json(ROOT/'docs/latency_optimization.json',result)
    with (OUT/'COMPARISON.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# FP32／FP16 延遲優化：實作與短測結果','',
        '日期：2026-10-18。Detect s／640，保留相同路數、batch 與完整 CPU 結果；無 TensorRT／新 runtime。','',
        '**結果：功能修改與回歸完成，尚未達到四個主要情境皆改善至少 5% 的效能門檻，不宣稱優化成功。**',
        '第一輪只有部分數據改善，其他情境持平或變慢；沒有啟動 500 次三輪確認、完整 AP 或長測。','',
        '## 已實作','',
        '- CPU 輸入先在鎖內預留容量／pinned page 與 FIFO 位置，再於鎖外複製；未完成的副本不允許出列，容量不足明確拒絕。',
        '- 複製或配置失敗會撤銷預留、歸還頁面並移除 FIFO 位置；後續請求不被未完成位置永久阻擋。',
        '- 引擎生命週期追蹤進行中的 submit；pause／stop 先關閉新請求入口，等待既有提交發布後再排空，close 最後才回收 scheduler／staging。',
        '- 同批所有 futures 都先發布，再按原有順序執行 callbacks；callback 例外保持隔離。callback 本身仍在 worker，並未新增非同步 callback executor。',
        '- collect_timings 開啟時，新增唯讀 `result.diagnostic_trace`，以 C++ API 入口為 0 ms 記錄 schema-v2 host 時間點；關閉時為空，不執行額外時戳／map 填寫。',
        '- 既有 queue／batch_wait／dispatch／CUDA 區間保留原義；C API v1～v3 的布局、結果與精度設定不變。',
        '- 版本號、預設模式與原始 ONNX 不變；ORT 尾端同步保留，沒有啟用 Graph 或新增 C API v4。','',
        '## 相同設定的修改前後短測','',
        '修改前為封存的 v0.4.0 原二進位，並非引用舊報告的不同設定成績。',
        '兩方單路都採 latency／batch 1／wait 0／depth 1；16／32 路都採 throughput／固定執行 batch 8／wait 2 ms／depth 2。',
        '來源為預載 COCO8 Detect 8 張 RGB 真實影像輪播。每例 25 次暖機排空、75 次計分，timing 關閉，包含提交到 CPU boxes 可用的時間。',
        '尾批照算，補零不計成功；路數不降低，不刪慢樣本，無失敗、拒絕或 ID 遺失。','',
        '| 精度／路數 | 前／後 FPS | 前／後平均 ms | 前／後 P95 ms | 平均降低 | P95 降低 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['precision'].upper()} / {r['lanes']} | {r['before_fps']:.2f} / {r['after_fps']:.2f} | "
            f"{r['before_mean_ms']:.2f} / {r['after_mean_ms']:.2f} | {r['before_p95_ms']:.2f} / {r['after_p95_ms']:.2f} | "
            f"{r['mean_reduction_percent']:+.2f}% | {r['p95_reduction_percent']:+.2f}% |")
    lines+=['',
        '正的「降低」表示改善，負值表示變慢。這是單輪 75 筆初篩，非三輪穩定結論。',
        'NVML 每 50 ms 記錄 clocks／溫度／使用率／顯存，包含於各案例 JSON；未鎖定 GPU clocks，也不能排除桌面／其他程序排程影響。',
        '即使批次上限與固定執行 shape 相同，邊界的實際湊批與補零次數仍可能不同；分布有完整保留。','',
        '## Trace 與條件式功能決策','',
        '使用既有 Nsight Systems，25 次暖機後才 capture，結束即停止；只擷取四個短區段。',
        '每批以 letterbox 完成到 score kernel 開始界定 ORT 相關區間，統計相同 compute stream 上的 kernel 與非 kernel 空隙。','',
        '| 精度／實際執行 batch | 區間中位數 ms | Kernel ms 中位數 | 非 kernel 占比中位數 |',
        '|---|---:|---:|---:|']
    for r in trace:
        m=r['median'];label=r['case'].split('_s')[0].upper()+' / '+('1' if '_s1_' in r['case'] else '8')
        lines.append(f"| {label} | {m['span_ms']:.3f} | {m['kernel_ms']:.3f} | {m['non_kernel_percent']:.1f}% |")
    lines+=['',
        '區間、kernel 與占比各自取中位數，因此不可用表中的中位數相減重算占比。',
        '非 kernel 空隙含 driver／host 提交與非 kernel stream 工作，且 profiling 本身會擾動時間；這是可改善機會的上界，不等於可省掉的固定百分比。',
        'FP16 的 batch 1／8 與 FP32 的 batch 1 達到 10% 研究門檻。FP32 batch 8 只有約 6.6%，依門檻不測該 Graph 組合。','',
        '**Graph 未列入交付功能：**',
        '- FP16 不改模型圖時，ORT 明確拒絕 capture：`all compute graph nodes have not been partitioned to the CUDAExecutionProvider`。',
        '- FP32 禁止 CPU fallback 的資格檢查直接指出存在分配至 CPU EP 的節點，亦不符合 capture 前提。',
        '- 未默默退回一般執行或更改精度／batch；失敗原文保留。因資格檢查未過，不宣稱雙槽 capture／pause／退出已通過驗證。',
        '- 沒有為了啟用 Graph 改寫／專化 ONNX，沒有加入無法工作的 `cuda_graph` 選項或 C API v4。','',
        '**完成執行緒未加入：**','',
        '| 精度／路數 | 副本中位數 ms | Ready slot 等待 ms | ORT host call ms | D2H 後完成 ms | 完成占服務時間 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in host:
        lines.append(f"| {r['case']} | {r['median_copy_ms']:.4f} | {r['median_ready_slot_wait_ms']:.2f} | "
            f"{r['median_ort_host_ms']:.2f} | {r['median_completion_ms']:.4f} | {r['median_completion_share_percent']:.2f}% |")
    lines+=['',
        'CPU 收尾以 D2H 完成到 future 發布前的 host 時間估計，包含診斷資料填寫，是保守上界；各例中位數遠低於 5%，不增加執行緒／佇列與生命週期複雜度。',
        '同批 callback 阻塞 futures 的問題有確定性原生測試，本輪效能案例沒有 callback，不能把小幅 FPS 差異歸因於 callback 修改。','',
        '## 正確性與生命週期驗證','',
        '- 最終 Python 回歸 35 項通過，包含真實 FP16 I/O、C API v1/v2/v3、輸入格式、空結果、錯誤派送、雙槽與 pause／resume。',
        '- 新增多執行緒大影像 submit 與 pause／close 交錯的測試，已接受請求在排空後都有對應結果，CUDA 資源無重建或熱路徑自有 device allocation。',
        '- 原生 C++ scheduler 測試用可控制的 copy barrier／例外，確認鎖外 copy、FIFO、不完整副本不可出列、容量拒絕、頁面回滾及暫停後排空。',
        '- 原生慢 callback 測試在 callback 仍阻塞時，確認同批兩個 futures 已 ready；例外不影響結果。',
        '- Detect FP32／FP16：640 與 (384,640)、batch 上限 1／2／4／8；Pose／Seg 各精度只做 2 張、batch 1／4 的共用路徑回歸。',
        '- 共 8 組獨立 ORT 同 ONNX 小樣本比較通過：前處理與 raw output 最大誤差皆 0，框／keypoint 最大誤差約 0.000031 pixel，mask IoU 最小 1.0。',
        '- 這些是小樣本回歸，不取代正式 AP；本輪沒有重跑 AP，也沒有把過去 AP 成績當成新二進位已通過完整 AP。','',
        '## 診斷介面與設定範例','',
        '`diagnostic_trace` 僅在 `collect_timings=True` 時填寫。所有值是相對 C++ predict_async 入口的 steady-clock ms：',
        '`api_entry, admitted, copy_done, enqueued, dequeued, slot_ready, compute_begin, ort_enter, ort_exit, d2h_ready, cpu_ready, future_publish`。',
        '`future_publish` 是呼叫 promise.set_value 前的時戳，不含 Python 包裝／呼叫端排程；benchmark 的 latency 仍由外部提交前開始，到 CPU 結果取得後結束。',
        '`d2h_ready` 代表 boxes／keypoints 的第一階段 D2H；Seg mask 是後續階段，本輪收尾門檻只對 Detect 判讀。',
        '可用 `compute_begin-slot_ready` 看真正 ready queue 等待，`copy_done-admitted` 看輸入配置／複製；ORT host call 不等於純 GPU kernel。',
        'C++ source 新增診斷欄位，須以相同 headers 重編譯；C ABI 舊結構保持不變。','',
        '```python',
        '# 單路低延遲：',
        'c.batch_mode = "latency"',
        'c.max_batch_size = 1',
        'c.max_latency_ms = 0',
        'c.pipeline_depth = 1',
        '',
        '# 滿載吞吐（兩方 A/B 採相同設定）：',
        'c.batch_mode = "throughput"',
        'c.max_batch_size = 8',
        'c.batch_shape_policy = "fixed"',
        'c.max_latency_ms = 2',
        'c.pipeline_depth = 2',
        'c.concurrency = 32  # 16 路測試由呼叫端只維持 16 筆在途',
        '```','',
        'FP16 throughput 單張仍可能執行固定 B8。只把 wait 改成 0 不會取消補零；應選 latency／batch 1。既有預設沒有改動。','',
        '## 後續建議與保留限制','',
        '這輪只改 host admission／future 完成順序，正常 s／640 的時間仍主要花在 ORT，沒有證據顯示這些修改能達成 5～15% 整體改善。',
        '若繼續追求較大收益，優先另案評估「固定 shape 圖專化／消除 CPU shape 分配」是否能讓 ORT Graph 成立；它不需要 TensorRT，但超出本輪禁止改寫模型圖的範圍。',
        'FP32 batch 8 的 trace 主要是 kernel 時間，需另作 cuDNN 演算法／layout 的小規模 A/B，而不是把所有問題歸咎於 host 同步。',
        '沒有減路數、放寬 NMS／輸出、增加 sessions，亦未改寫 NMS／Seg kernel。未自動更新版本、推送 Git 或發布 Release。','',
        '## 原始資料與重現','',
        f'- 原版二進位 SHA256：`{baseline_hash}`。',
        f'- 候選二進位 SHA256：`{build_fingerprint()}`。',
        '- 模型與準備後 ONNX 雜湊、GPU samples、所有參數：本機各 case JSON。',
        '- 每請求 JSONL、Nsight .nsys-rep／SQLite、Graph 拒絕紀錄、unit log、原生測試 log：`runs/latency_opt/`。',
        '- [比較與 trace 摘要 JSON](latency_optimization.json)。','',
        '```powershell',
        'python tools/diagnostics/latency_opt.py --variant baseline # 必須在修改前封存',
        'python tools/diagnostics/latency_trace.py',
        'python tools/diagnostics/latency_trace_summary.py',
        'python tools/diagnostics/latency_opt.py --variant candidate',
        'python tools/diagnostics/latency_opt.py --variant candidate --phase diagnostic',
        'python tools/diagnostics/latency_functional.py',
        'python tools/diagnostics/latency_opt_report.py # 只讀資料／CPU',
        '```','',
        'Nsight 使用暫存 ASCII junction 解決 Windows CLI 的非 ASCII 路徑限制，沒有改名專案或複製大資料集。']
    (ROOT/'docs/latency_optimization.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report generated; quick_all_four_improve =',quick_improves,'; speed acceptance remains NOT MET')


if __name__=='__main__':main()
