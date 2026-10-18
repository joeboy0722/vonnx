"""Read-only GPU-wise: summarize the bounded X/L/n 1920 experiment."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"runs/xln_1920"
MODELS=("yolov8x","yolov8l","yolov8n")

def read(name):
    return json.loads((BASE/name/"result.json").read_text(encoding="utf-8"))

def rows(name,case):
    return [json.loads(line) for line in (BASE/name/(case+".jsonl")).read_text().splitlines()]

def metrics(data):
    values=np.array([r["latency_ms"] for r in data])
    return dict(mean=float(values.mean()),p95=float(np.percentile(values,95)),p99=float(np.percentile(values,99)))

records=[]
for precision in ("fp32","fp16"):
    for backend in ("native","official"):
        name=f"{backend}_{precision}"
        result=read(name)
        assert result['status']=='passed' and result['closed']
        for case in result['scenarios']:
            data=rows(name,case['name'])
            assert len(data)==case['count'] and not case['errors']
            for model in MODELS:
                subset=[r for r in data if r['model']==model]
                if not subset:continue
                records.append(dict(precision=precision,backend=backend,case=case['name'],model=model,
                                    requests=len(subset),fps=len(subset)/case['elapsed_s'],**metrics(subset)))
with (BASE/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)

checks=[]
for precision in ('fp32','fp16'):
    for model in MODELS:
        a=np.load(BASE/f'native_{precision}'/(model+'_check.npz'))['alone']
        b=np.load(BASE/f'official_{precision}'/(model+'_check.npz'))['alone']
        used=set();coords=[];scores=[]
        for box in a:
            candidates=[j for j,w in enumerate(b) if j not in used and w[5]==box[5]]
            if not candidates:raise AssertionError('Unmatched box')
            j=min(candidates,key=lambda j:np.abs(b[j,:4]-box[:4]).max());used.add(j)
            coords.append(float(np.abs(b[j,:4]-box[:4]).max()))
            scores.append(float(abs(b[j,4]-box[4])))
        assert len(used)==len(b)
        checks.append(dict(precision=precision,model=model,count=len(a),max_coord_px=max(coords,default=0),
                           max_score_delta=max(scores,default=0)))

lines=['# X／L／n，1920 三模型非同步短測', '', '日期：2026-10-18。RTX 3070 8GB；Detect YOLOv8x／l／n，各一個獨立引擎或官方 PyTorch predictor。', '',
       '## 結論', '',
       '- 原版 vONNX 在本輪四組三模型併發情境皆較官方原版快：FP32 約 3～4%，FP16 約 7～8%。FP32 屬小幅優勢；每組一輪短測，不能宣稱穩態統計驗收。',
       '- X／L 確實大幅增加 n 的延遲。FP16 已找到可改善的軟體因素：動態 ONNX 的執行期 shape／資料相依與中途同步，使小模型進度與大模型牽連。不是全部都能歸咎硬體。',
       '- 僅對 n 產生固定輸入 shape 的實驗模型後，FP16 n 延遲顯著降低；仍存在共享 GPU 競爭，而且 X／L 會因 n 取得更多運算機會而變慢。這不等於完成硬體隔離。',
       '- FP32 的同一固定 shape 實驗未呈現相近收益；不得把 FP16 的改善幅度套用到 FP32。', '',
       '## 方法與範圍', '',
       '預載 COCO8 Detect 8 張真實影像輪播，RGB 原始輸入 1920×1080，固定 letterbox 至 1920×1920。下載、解碼、初始化與每模型 5 次暖機不計分。',
       'batch 上限 1、等待 0 ms、pipeline depth 1、每引擎 admission 上限 32；每模型測 1／4 路非同步 producer，每路最多一筆未完成請求。',
       '單跑約 3 秒，三模型同跑約 6 秒；每情境每模型最多 75 筆計分請求。所有已接受請求排空，主表計入尾批與排空；不丟慢樣本。每案例 300 秒硬性上限，未跑 500 次、完整 AP 或長測。',
       '單跑時其他兩模型保持初始化但不提交工作。官方各有獨立 predictor／工作執行緒／CUDA stream；計時包含輸入副本、官方前後處理及 CPU 框可用。雙方 FP32／FP16、TF32 關閉、conf 0.25／IoU 0.7／max_det 300。',
       '官方原版保留內建 profiling。此環境官方精度由 quantize=16 選擇，已斷言 backend.fp16 與要求一致。只使用 half=True 的初次錯誤設定存於 discarded_official_half_flag_only，完全排除。', '',
       '## 原版總吞吐：包含排空的全部完成請求', '',
       '| 精度 | 每模型路數 | vONNX 次／秒 | 官方次／秒 | 差異 |', '|---|---:|---:|---:|---:|']
for precision in ('fp32','fp16'):
    for lanes in (1,4):
        rates=[]
        for backend in ('native','official'):
            case=next(s for s in read(f'{backend}_{precision}')['scenarios'] if s['name']==f'concurrent_{lanes}')
            rates.append(case['count']/case['elapsed_s'])
        lines.append(f'| {precision} | {lanes} | {rates[0]:.2f} | {rates[1]:.2f} | {(rates[0]/rates[1]-1)*100:+.1f}% |')
lines += ['', '三模型推論各算一次；不代表三任務整組影像 FPS。原始 JSON 另有共同活動窗口 FPS，避免 n 先達 75 次停止後的尾段影響定位實驗；兩種口徑不可混用。', '',
          '## 每模型延遲：全部請求，含尾段', '',
          '| 精度 | 模型 | vONNX 單跑平均 ms | 官方單跑平均 ms | vONNX 同跑 1 路 平均／P95 ms | 官方同跑 1 路 平均／P95 ms |',
          '|---|---|---:|---:|---:|---:|']
for precision in ('fp32','fp16'):
    for model in MODELS:
        def get(backend,case):
            return next(r for r in records if r['precision']==precision and r['backend']==backend and r['case']==case and r['model']==model)
        a,b=get('native','solo_'+model),get('official','solo_'+model)
        x,y=get('native','concurrent_1'),get('official','concurrent_1')
        lines.append(f"| {precision} | {model} | {a['mean']:.1f} | {b['mean']:.1f} | {x['mean']:.1f}／{x['p95']:.1f} | {y['mean']:.1f}／{y['p95']:.1f} |")
lines += ['', '4 路詳細平均／P95／P99、逐模型 FPS 在 summary.csv。增加路數沒有明顯增加總吞吐，主要增加各引擎佇列等待；此處 batch 固定 1，不能推廣到批次吞吐測試。', '',
          '## GPU trace：FP16 問題定位', '',
          '只擷取三模型 1 路的短段；trace 成績不混入主要跑分。原版 compute stream 分別為 13／64／115，初始化順序 X／L／n。沒有 cudaDeviceSynchronize、熱路徑 cudaMalloc／cudaFree。ORT 結尾是各自 stream 的同步，並未移除。', '',
          '| n 指標，中位數／每推論 | 動態來源 | 僅 n 固定 shape |', '|---|---:|---:|']
before=json.loads((BASE/'traces/native_fp16.summary.json').read_text())
after=json.loads((BASE/'traces/native_fp16_static_n.summary.json').read_text())
a,b=before['streams'][2],after['streams'][2]
for title,key in [('GPU kernel 自身總執行 ms','own_kernel_ms'),('GPU 推論跨度 ms','inference_span_ms'),('最後 kernel 的 CPU launch 距前處理結束 ms','last_launch_from_pre_ms'),('kernel 數','kernel_count')]:
    lines.append(f"| {title} | {a['median'][key]:.2f} | {b['median'][key]:.2f} |")
lines.append(f"| CPU event 等待次數／推論，含必要結果等待 | {a['event_wait_calls_per_batch']:.0f} | {b['event_wait_calls_per_batch']:.0f} |")
lines += ['', '動態版 n 的最後 kernel 到約 182 ms 才完成 CPU 提交，不能說「所有工作早就提交，只剩硬體忙」。固定 shape 後約 2.6 ms 就提交到尾端，且少掉 25 次中途 event 等待與一批 shape／anchor 相關工作，證實有軟體可改善空間。',
          'trace 的非 kernel 空隙同時含其他 stream、必要資料相依、驅動排程；不能全算成可刪除 overhead。固定版 n 第一張的 kernel 仍可在提交後等約 162 ms，證明移除中途等待也無法消除 GPU 共享競爭。', '',
          '## 固定 shape 定位實驗（非正式引擎功能）', '',
          '只複製 n 的來源 ONNX 並將輸入改為 [1,3,1920,1920]，由相同初始化／FP16 內部轉換流程載入；X／L 不改，來源檔不覆寫。產物只能用此尺寸／batch 1。', '',
          '| 精度 | 每模型路數 | 原 n 平均 ms | 固定 n 平均 ms | 原／固定 n P95 ms |', '|---|---:|---:|---:|---:|']
experiments=[]
for precision in ('fp32','fp16'):
    for lanes in (1,4):
        case=f'concurrent_{lanes}'
        original=metrics([r for r in rows(f'native_{precision}',case) if r['model']=='yolov8n'])
        fixed=metrics([r for r in rows(f'native_{precision}_static_n',case) if r['model']=='yolov8n'])
        experiments.append(dict(precision=precision,lanes=lanes,original=original,fixed=fixed))
        lines.append(f"| {precision} | {lanes} | {original['mean']:.1f} | {fixed['mean']:.1f} | {original['p95']:.1f}／{fixed['p95']:.1f} |")
lines += ['', 'FP16 1 路共同活動窗口：n 約 203.5→57.9 ms；X 約 203.6→257.7 ms、L 約 203.5→251.1 ms。小模型跑得更多會改變 GPU 分配；總推論次數提高不能解讀成 X／L 也加速。',
          'FP32 同實驗 n 約 425.9→419.3 ms，只有小幅變化，未證明值得整合。', '',
          '官方診斷控制：僅將 Profile.time 改為 perf_counter、取消計時用全 GPU 同步，其餘 predictor／CPU 結果不變。FP16 1 路共同窗口總吞吐約 13.35→13.53，n 仍約 216 ms；官方 profiling 不是此次小模型延遲牽連的唯一原因。此組不是官方原版，不放進勝負表。', '',
          '## 正確性與限制', '',
          '- 已驗證請求 ID、batch=1、一筆一次回傳、無失敗／拒絕／OOM、正常 close；同一引擎單跑與同跑檢查框完全一致。',
          '- 六組官方對照的檢查影像，框類別與數量一致。FP32 最大座標差 0.00025 pixel；FP16 n 為 0、L 為 1、X 為 2 pixels，最大分數差 0.001953。',
          '- FP16 X 在 1920 的對照差異超過既有 1 pixel 嚴格門檻，不能宣稱完整正確性／AP 驗收通過；可能涉及官方 FP16 座標量化與本引擎保留 FP32 的數值路徑，需要另案分層比對。',
          '- 固定／動態 n 在本張影像的框差為 0；僅小樣本定位，不代表所有影像準確度已驗證。',
          '- 裝置記憶體峰值包含桌面與其他 resident 資源：vONNX FP32 約 5012 MiB、FP16 約 3180 MiB；官方約 4120／3042 MiB。',
          '- 本輪沒有純 C++ 控制測試；GPU trace 已直接觀測到 ORT／CUDA 原生層的等待與圖專化改善，未將 Python 列為主要原因。', '',
          '## 建議正式改善方向', '',
          '最有根據的是初始化時按已固定的 imgsz／執行 batch 專化圖，讓 ORT 提前折疊 shape 與常數路徑。不能直接把所有動態 batch 都固定為 1；須保留使用者的 batch 語意、不同 batch binding、快取鍵與來源模型驗證。',
          '本次已交付可重現的固定 n 實驗選項；沒有改正式引擎、啟用 CUDA Graph、移除 ORT 安全同步、更新版本、推送或發布。FP16 的收益值得獨立修正，FP32 不套用同樣收益預期。', '',
          '## 重現與輸出', '',
          '```powershell',
          r'.\venv\python.exe tools\diagnostics\xln_1920.py --backend native --precision fp16 --output runs\xln_1920\native_fp16',
          r'.\venv\python.exe tools\diagnostics\xln_1920.py --backend official --precision fp16 --output runs\xln_1920\official_fp16',
          r'.\venv\python.exe tools\diagnostics\xln_1920.py --backend native --precision fp16 --specialize-n --concurrent-only --output runs\xln_1920\native_fp16_static_n',
          r'.\venv\python.exe tools\diagnostics\xln_report.py',
          '```', '',
          'FP32 改 --precision fp32，使用不同輸出目錄。trace 可用 xln_trace.py；需本機已有 Nsight Systems。',
          '各結果目錄的 result.json 含環境、二進位與來源模型雜湊、初始化與裝置記憶體；*.jsonl 為每筆結果。summary.csv 為全部主測請求統計；traces/*.summary.json、*.sqlite、*.nsys-rep 為原生時序證據。', '']
payload=dict(records=records,reference_checks=checks,experiments=experiments)
(BASE/'summary.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
report=ROOT/'docs/xln_1920.md';report.write_text('\n'.join(lines),encoding='utf-8')
print(report)
