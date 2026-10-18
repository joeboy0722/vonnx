"""Create the FP32/FP16 latency audit report from bounded measurements; CPU only."""
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vonnx_tools.common import build_fingerprint,sha256,write_json
OUT=ROOT/'runs/latency_audit'
MODELS=('yolov8s','yolov8s-pose','yolov8s-seg')


def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    records={}
    for case in manifest['cases']:
        row=json.loads((OUT/(case['id']+'.json')).read_text(encoding='utf-8'))
        assert row['status']=='passed' and row['success']==75 and not row['failures']
        assert row['binary_sha256']==build_fingerprint() and row['case']==case
        records[case['id']]=row
    assert len(records)==18
    def get(precision,model,lanes):return records[f'{precision}_{model}_lanes{lanes}']
    def stage(row,name):return row['stages'][name]['mean_ms']
    rows=[]
    for row in records.values():
        c=row['case'];lat=row['latency']['mean_ms']
        wait=stage(row,'queue_ms')+stage(row,'batch_wait_ms')+stage(row,'dispatch_ms')
        rows.append(dict(case=c['id'],precision=c['precision'],model=c['model'],lanes=c['lanes'],
            fps=row['fps'],mean_ms=lat,p95_ms=row['latency']['p95_ms'],
            **{key:stage(row,key) for key in row['stages']},
            waiting_share_percent=100*wait/lat,remainder_wall_ms=lat-wait,
            physical_batch_histogram=row['physical_batch_histogram'],padded_images=row['padded_images']))
    with (OUT/'SUMMARY.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(binary_sha256=build_fingerprint(),environment=manifest['environment'],
        models={model:dict(weight_sha256=get('fp32',model,1)['model']['weight_sha256'],
                          source_onnx_sha256=get('fp32',model,1)['model']['onnx_sha256'],
                          fp16_prepared={k:v for k,v in get('fp16',model,1)['prepared_model'].items() if k!='path'})
                for model in MODELS},
        source_sha256={p:sha256(ROOT/p) for p in ('src/vonnx_engine.cpp','src/continuous_scheduler.cpp',
            'src/onnx_runner.cpp','src/cuda_nms.cu','src/cuda_preprocess.cu','vonnx/precision.py')},
        sample='100 requests per case; 25 warmup + 75 measured; detailed timing enabled',rows=rows)
    write_json(OUT/'SUMMARY.json',summary)
    write_json(ROOT/'docs/fp32_fp16_latency.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(16,4.7))
    labels=('Detect','Pose','Seg')
    x=np.arange(3)
    for i,precision in enumerate(('fp32','fp16')):
        values=[get(precision,m,1)['latency']['mean_ms'] for m in MODELS]
        bars=axes[0].bar(x+(i-.5)*.35,values,.35,label=precision.upper())
        axes[0].bar_label(bars,fmt='%.2f',fontsize=8)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Submit to CPU result (ms)',title='1 lane / latency mode')
    axes[0].legend()
    high=[get(p,m,32) for m in MODELS for p in ('fp32','fp16')]
    bottom=np.zeros(6)
    for key,label,color in [('queue_ms','Queue','#9baec8'),('batch_wait_ms','Batch wait','#eec95b'),
                            ('dispatch_ms','Slot/worker wait','#e6a678'),('remainder','Remaining wall time','#59a99c')]:
        values=[r['latency']['mean_ms']-sum(stage(r,k) for k in ('queue_ms','batch_wait_ms','dispatch_ms'))
                if key=='remainder' else stage(r,key) for r in high]
        axes[1].bar(np.arange(6),values,bottom=bottom,label=label,color=color)
        bottom+=values
    axes[1].set(xticks=np.arange(6),xticklabels=[f'{m}\nFP{p}' for m in labels for p in ('32','16')],
        ylabel='Mean latency (ms)',title='32 lanes: CPU wall-time partition')
    axes[1].legend(fontsize=7)
    low=[get('fp16','yolov8s',1),records['fp16_yolov8s_single_throughput_dynamic'],
         records['fp16_yolov8s_single_throughput_auto']]
    bars=axes[2].bar(np.arange(3),[r['latency']['mean_ms'] for r in low],color=['#59a99c','#eec95b','#cc706b'])
    axes[2].bar_label(bars,fmt='%.2f')
    axes[2].set(xticks=np.arange(3),xticklabels=['Latency\nB1','Throughput\ndynamic B1','Throughput\nfixed B8'],
        ylabel='Mean latency (ms)',title='FP16 Detect: one actual request')
    fig.suptitle('Released vONNX binary, s / 640; bounded timing diagnostics, not a throughput acceptance test')
    fig.tight_layout()
    asset=ROOT/'docs/assets/fp32_fp16_latency.png';asset.parent.mkdir(exist_ok=True)
    fig.savefig(asset,dpi=160);plt.close(fig)
    lines=['# FP32／FP16 延遲瓶頸與改善建議','',
        '日期：2026-10-18。分析版本：v0.4.0，核心未修改；Windows／RTX 3070 8GB。',
        f'Python 二進位 SHA256：`{build_fingerprint()}`。','',
        '## 結論','',
        '目前最嚴重的延遲要依負載區分：','',
        '1. 高併發：最大部分是已接受請求等待前面的批次；單一 compute stream／worker 有序處理，32 路不代表 32 個模型運算同時完成。',
        '2. 單路：主要時間在 ORT／cuDNN 的執行區間；FP16 單張使用吞吐預設時，固定 batch 8 的補零計算還會顯著增加延遲。',
        '3. FP32：除了卷積計算，batch 尺寸切換也出現計畫重建型尖峰；FP16 的固定 shape 已避開這項主要風險。',
        '4. Seg：ROI mask 重建的實作仍有重複計算，但本輪 s／640 正常影像中，它尚未超過 ORT 成為第一瓶頸。','',
        '建議優先改善模式選擇／批次穩定性與延遲量測，再研究 ORT launch／kernel 路徑及完成階段流水線。',
        '不建議優先增加 stream／session、重寫 Python 測試或直接移除 ORT 同步。這份建議不引入 TensorRT／INT8。','',
        '![Latency audit](assets/fp32_fp16_latency.png)','',
        '## 量測範圍與讀法','',
        '本輪 18 個獨立案例：Detect／Pose／Seg s、640；FP32／FP16，各單路與 32 路；另有單路吞吐模式與 FP32 固定 batch 對照。',
        '每例共 100 次，25 次暖機完成並排空後，75 次計分；各案例最多 240 秒，不連接完整跑分。',
        '預載官方 COCO8／COCO8-pose／COCO8-seg 真實影像，RGB24；conf=.25、IoU=.7、max_det=300，Seg 為 ROI uint8 masks。',
        '每路最多一筆未完成請求，batch 上限 8；單路 latency／wait 0／depth 1；32 路 throughput／wait 2／depth 2。',
        '詳細 timing 開啟；所有案例 75/75 成功、無失敗、無拒絕。下載、解碼、轉換、初始化與暖機不計。',
        '75 筆很短，起始／排空與湊不滿 batch 的影響很大；本輪 FPS 不取代先前三輪 475 筆、timing 關閉的速度驗收，亦不單獨證明效能回歸。',
        '未記錄本輪 GPU clocks／溫度／其他程序使用量，因此不將小幅跨案例差異視為穩定因果。','',
        '**時間欄位不是互斥的 GPU 工作清單：**',
        '- `queue_ms`：scheduler 佇列內等待，扣掉估計湊批段；不包含提交前的 CPU 複製。',
        '- `batch_wait_ms`：pop_batch 的等待；以每筆 queue 總時間限制，並非每筆真正「只為湊批」等待的精確量。',
        '- `dispatch_ms`：pop_batch 到 worker 開始 process_slot；包含 CPU 上傳提交與 ready slot 等待上一批，不是純 CPU dispatch 計算。',
        '- `inference_ms`：pre event 到 infer event 的 CUDA 時間軸區間，含 ORT 內部提交空隙／等待；原生半精度輸出時也含轉 FP32 kernel。它不是純卷積 kernel 加總。',
        '- H2D 可以與前一批 compute 重疊，也可能已被 dispatch 的 wall time 包住；不能把所有欄位直接相加作為端到端時間。',
        '- CPU 結果組裝、future 喚醒、Python 物件包裝與 submit staging copy 沒有完整獨立欄位。圖中剩餘 wall time 不可解讀成純 GPU 時間。','',
        '## 單路：ORT 執行與模式選擇','',
        '| 任務 | FP32 平均／P95 ms | FP16 平均／P95 ms | FP32／FP16 ORT 區間 ms |',
        '|---|---:|---:|---:|']
    for model,label in zip(MODELS,labels):
        a=get('fp32',model,1);b=get('fp16',model,1)
        lines.append(f"| {label} | {a['latency']['mean_ms']:.2f} / {a['latency']['p95_ms']:.2f} | "
            f"{b['latency']['mean_ms']:.2f} / {b['latency']['p95_ms']:.2f} | {stage(a,'inference_ms'):.2f} / {stage(b,'inference_ms'):.2f} |")
    lines+=['',
        'FP16 並不保證 batch 1 一定比 FP32 快；本輪 Pose FP16 平均反而較高，且僅單輪，不能據此斷言 FP16 Pose 穩定較慢。',
        '這裡的小 batch 核心時間與 host launch／layout 等成本同等重要；需 Nsight 才能把純 GPU kernel 與提交空隙拆開。','',
        '**Detect 單筆串行請求，batch 上限仍為 8：**','',
        '| 精度／模式 | 真實 batch／執行 batch | 平均延遲 ms | 湊批 ms | ORT 區間 ms |',
        '|---|---|---:|---:|---:|']
    single=[('FP32 latency',get('fp32','yolov8s',1)),('FP32 throughput auto',records['fp32_yolov8s_single_throughput_auto']),
        ('FP16 latency',get('fp16','yolov8s',1)),('FP16 throughput dynamic',records['fp16_yolov8s_single_throughput_dynamic']),
        ('FP16 throughput auto/fixed',records['fp16_yolov8s_single_throughput_auto'])]
    for name,r in single:
        physical=list(r['physical_batch_histogram'])[0]
        lines.append(f"| {name} | 1 / {physical} | {r['latency']['mean_ms']:.2f} | {stage(r,'batch_wait_ms'):.2f} | {stage(r,'inference_ms'):.2f} |")
    lines+=['',
        'FP16 throughput 預設會對不足 batch 8 的位置補零；這保護高併發時的 cuDNN shape 穩定性，但單筆情境多做了七個位置的推論。',
        '本輪固定 B8 約 25.76 ms，dynamic B1 約 9.65 ms，latency B1 約 7.06 ms；約 16 ms 差距主要是執行 batch，不是 2 ms timer。',
        '最直接且已可用的單路設定是 `batch_mode="latency"`、`max_latency_ms=0`；容量確定僅需單張時可設 `max_batch_size=1`。',
        '目前 FP16 auto/fixed 不能只靠 wait=0 取消補零；應同時選擇 latency 或明確 dynamic。','',
        '## 高併發：排隊才是最大端到端區段','',
        '| 精度／任務 | 平均／P95 ms | Queue ms | Batch wait ms | Dispatch ms | ORT 區間 ms | Queue+wait+dispatch 占比 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model,label in zip(MODELS,labels):
        for precision in ('fp32','fp16'):
            r=get(precision,model,32);wait=sum(stage(r,k) for k in ('queue_ms','batch_wait_ms','dispatch_ms'))
            lines.append(f"| {precision.upper()} {label} | {r['latency']['mean_ms']:.2f} / {r['latency']['p95_ms']:.2f} | "
                f"{stage(r,'queue_ms'):.2f} | {stage(r,'batch_wait_ms'):.3f} | {stage(r,'dispatch_ms'):.2f} | "
                f"{stage(r,'inference_ms'):.2f} | {100*wait/r['latency']['mean_ms']:.1f}% |")
    lines+=['',
        '32 路時約七成延遲落在 queue／dispatch 等待；減少 2 ms 湊批上限並不能解除已在前面等待的批次。',
        '單一 session／compute stream 有序執行，雙槽目前主要重疊下一批 H2D；worker 仍等 CPU 結果可讀，完成結果組裝之後才處理下一槽。',
        '排隊是服務時間與在途數共同造成；如果核心吞吐不變，只增加 concurrency 可能主要增加延遲。減少在途數能減等待，不能宣稱提高硬體吞吐。','',
        '先前 timing 關閉的三輪確認也呈現同一特徵（相同二進位，但屬於既有測試）：','',
        '| 任務 | 16 路 FPS／平均延遲 ms | 32 路 FPS／平均延遲 ms |','|---|---:|---:|']
    confirmed=[]
    for path in (ROOT/'runs/fp16/fp16').glob('confirm__*.json'):
        r=json.loads(path.read_text(encoding='utf-8'))
        if r['case']['backend']=='native' and r['binary_sha256']==build_fingerprint():confirmed.append(r)
    for model,label in zip(MODELS,labels):
        values=[]
        for lanes in (16,32):
            selected=[r for r in confirmed if r['case']['model']==model and r['case']['lanes']==lanes]
            assert len(selected)==3
            values.append(f"{statistics.median(r['fps'] for r in selected):.2f} / {statistics.median(r['latency']['mean_ms'] for r in selected):.2f}")
        lines.append(f'| {label} | '+' | '.join(values)+' |')
    lines+=['',
        '16 → 32 路吞吐幾乎不變，平均延遲約加倍；使用者能控制實際在途數時，16 路值得優先評估。',
        '若使用容量上限拒絕新請求，必須讓呼叫端背壓／處理明確錯誤；不能丟棄已接受請求來美化延遲。','',
        '## FP32 的 batch 切換尖峰與固定對照','',
        '| 任務 | Auto 平均／P95 ms | Fixed 平均／P95 ms | Auto／Fixed 短測 FPS |','|---|---:|---:|---:|']
    for model,label in zip(MODELS,labels):
        a=get('fp32',model,32);b=records[f'fp32_{model}_lanes32_fixed']
        lines.append(f"| {label} | {a['latency']['mean_ms']:.2f} / {a['latency']['p95_ms']:.2f} | "
            f"{b['latency']['mean_ms']:.2f} / {b['latency']['p95_ms']:.2f} | {a['fps']:.2f} / {b['fps']:.2f} |")
    lines+=['',
        'Detect auto 的首個 B8 ORT 區間約 71.91 ms，之後完整 B8 的中位數約 35.46 ms，尾端 B2／B1 約 74.80／74.37 ms。',
        '固定 B8 的中位數約 36.13 ms，表示主要改善來自避免切換尖峰；穩定完整 B8 並未因此更快。',
        '暖機 25 筆剛好可能以 B1 結束；開始計分回到 B8，以及 75 筆尾端切回 B2／B1，會放大短測差距。',
        '這與既有 cuDNN last-shape 計畫重建追查一致；本輪 fixed A/B 支持 shape 切換成本存在，但未新增 kernel trace 證明每一筆尖峰的內部分解。',
        'FP32 已可明確指定 `batch_shape_policy="fixed"`。它適合負載能維持滿批的情境；稀疏輸入同樣會浪費補零計算，不建議無條件改成所有模式的預設。','',
        '## ORT、同步與其他計算的判斷','',
        'ORT／cuDNN 執行區間是主要「每批服務」成本。以 FP16 Detect 32 路為例，平均 ORT 約 24.07 ms、前處理 0.86 ms、NMS 0.85 ms、D2H 0.05 ms。',
        '這些 CUDA 區間之中 ORT 占約 93%；它包含 host 提交空隙，尚不能說 93% 都是 GPU 卷積。',
        '所以外圍小項目即使全部消除，可見區間的理想改善也只有約 7%；這不是端到端延遲的保證，真正大幅提升仍要動到 ORT 路徑或每批工作量。',
        'ORT 尾端同步在 Run 內完成。它會阻擋 host 提前提交後處理，但大部分等待也在等真正 GPU 工作，不能將整個 inference_ms 視為可刪除的同步浪費。',
        'worker 之後仍會 `cudaEventSynchronize(copied)`，Seg 還有 masks 的第二階段同步；只拿掉 ORT 同步並不會自動形成跨批完整流水線。',
        '既有純 C++／1920 實驗移除 ORT 同步沒有提高吞吐，且退出崩潰；本輪不重試或使用該設定。這個歷史結論不能量化目前 s／640 的同步獨立成本。','',
        '前後處理的程式證據：',
        '- NMS 先對所有 anchors 求類別最高分、逐影像 radix sort；greedy suppression 每影像只有一個 CUDA thread。一般 s／640 約 0.1～1 ms，密集候選時仍可能成為尾延遲來源。',
        '- Seg 每個原圖 ROI 像素重做四次 prototype／coefficients 點積，再雙線性插值；有重複計算機會。本輪 postprocess 約 0.28 ms（單張）／1.43 ms（32 路每批區間），大圖或大量 masks 尚未重測。',
        '- CPU submit 的 memcpy 位於 lifecycle／scheduler mutex 內，且 submitted timestamp 在 copy 之後；它是本輪 queue 欄位沒有涵蓋的潛在高解析度／高路數成本。',
        '- Callback 在 worker 的 batch completion 迴圈內同步執行；慢 callback 還會延後同批後面的 future。這是程式確認的條件性瓶頸，本輪沒有 callback，不能拿來解釋本輪數字。',
        '- Detect／Pose 已跳過 masks 工作；debug 未開，timing 關閉也不建立純 timing event。常駐 GPU 資源與 bindings 已存在，不應再把「每批初始化 CUDA」當作目前主因。',
        '- 自動 FP16 模型保留 FP32 I/O，這輪沒有外部 I/O half cast kernel；真正 FP16 I/O 才走額外轉型。內部 Cast／layout 成本仍需分析，不能直接移除受保護的 FP32 DFL／座標節點。','',
        '## 改善優先順序','',
        '| 優先 | 建議 | 有機會改善什麼 | 限制與驗證 |',
        '|---|---|---|---|',
        '| 1 | 明確 latency／throughput 預設、文件與診斷提示；單路選 latency，滿載限制實際在途量 | 已實測可大幅降低單路 FP16 與排隊延遲 | 不以降低成功量或丟幀冒充吞吐提升；實際流量要驗證 |',
        '| 2 | FP32 滿載選固定 batch；研究少量常駐 batch bucket 與切換遲滯 | 減少 batch 切換尖峰與不足批的浪費 | 單 session 仍可能只留最後 cuDNN 計畫；bucket 不是免費解法，不先複製多個 session |',
        '| 3 | 把時間欄位拆成 submit copy／queue／slot ready wait／ORT host call／GPU span／CPU completion | 避免把 dispatch 與同步誤判，找出剩餘 ms | 時戳須保持熱路徑低成本；正確處理重疊與批次共用時間 |',
        '| 4 | 短 Nsight trace，檢查卷積、Transpose／Cast、CPU shape ops、kernel 間空隙；FP32 NHWC／cuDNN algorithm A/B | 對準占大部分區間的 ORT 路徑 | FP16 已 prefer_nhwc，FP32 尚未；workspace／EXHAUSTIVE 的記憶體與初始化成本要預算化，不承諾固定增幅 |',
        '| 5 | 在現有 ORT CUDA 內研究 CUDA Graph，固定 slot／batch 各自 graph ID | 若 host launch 空隙顯著，可降低提交成本且不增加 TensorRT 依賴 | 需全部節點可落 CUDA、固定位址與 shape、單 session 有序 Run；先驗證 capture／退出／pause／雙槽正確性 |',
        '| 6 | 將 Detect／Pose 的 CPU 結果收尾與 callback 從 compute worker 分離 | 減少每批結果處理造成的下一批提交空隙與慢 callback 阻塞 | 持有 slot 到真正 D2H 完成；關閉須 drain；有界完成佇列。Seg 需另設兩階段依賴 |',
        '| 7 | NMS 先穩定過濾 confidence，再批次排序／bitmask greedy | 密集候選的 NMS 長尾；一般影像收益可能小 | 保留原始索引、相同分數 tie-break 與精確 greedy；不可任意 top-k 改變輸出 |',
        '| 8 | Seg 先重建低解析度 mask logits，再對 ROI 插值；限量／分塊工作區 | 大圖與大量 masks 的重複點積成本 | FP32 累加與裁切／縮放／二值化順序一致；mask IoU、AP、顯存預算要重驗 |',
        '| 9 | Submit pinned page 先預留後在鎖外 copy；支援既有 GPU producer 直送 | 高解析度 CPU 輸入的鎖競爭與拷貝 | submit 與 pause／close 要有明確 admission 順序；失敗必須歸還容量與頁面；不能刪掉輸入持有保證 |','',
        '建議先做 1～3，接著用 4 的證據決定 5；6～9 依實際影像／輸入來源挑選。',
        '若 trace 顯示 CPU shape ops／layout 轉換明顯，可研究初始化時以固定 imgsz／實際執行 batch 專化 ORT 圖；',
        '必須同步驗證 ModelSpec 與靜態 batch 相容檢查，不能直接凍結 shape 後讓既有動態批次 binding 誤用。',
        '不為 CPU await future 另寫 C#／C++ 重做整個引擎；Python binding 已釋放等待時的 GIL，既有多模型 C++ 實驗也未發現 Python 是該情境主因。',
        '也不把低於毫秒的 D2H boxes 複製作為首要目標：先讀 counts 再複製有效框可能額外增加同步，反而更慢。','',
        '## 每項修改如何驗證','',
        '先維持三種 s／640、FP32／FP16，不增加模型矩陣。每次只改一項，單路與 16／32 路分開看；測試滿載、稀疏單張、突發尾批。',
        '主要速度以 timing 關閉的相同輸入、CPU 結果語意作 A/B；短診斷可用 100 次，但正式短測確認仍採 25 暖機＋475 計分、三輪。',
        '記錄 batch／physical batch／padding、P50/P95/P99、接受／成功／拒絕數、GPU clocks／記憶體、binary／模型雜湊。改完須保留 IDs、一次完成、pause/resume／close 與小樣本框／keypoints／mask 正確性。',
        '若動到前後處理或計算圖，通過短測後才另討論正式 AP；完整驗證與長測不自動開始。',
        '本輪只分析與產生研究腳本／報告；未改正式推論核心、API、版本或 Release。','',
        '## 多模型與大解析度的適用範圍','',
        '上述新數據只對單引擎 s／640 有直接證據。四模型／1920 的既有純 C++ trace 顯示：小模型 kernel 已提交，但在 GPU 資源競爭中延後完成；多 stream 不等於 GPU 硬體隔離。',
        '不能把 s／640 的單路平均套用到多模型 1920。若接著研究多模型，優先做跨引擎在途工作與公平配額，限制大模型持續提交；這改善可預期性，不保證總吞吐或每模型單獨速度。',
        '本輪沒有分離 Windows WDDM 排程、SM、記憶體頻寬、cuDNN workspace 的占比，不把全部責任推給硬體吃滿。','',
        '## 程式與官方依據','',
        '- [`src/vonnx_engine.cpp`](../src/vonnx_engine.cpp)：upload_loop、worker_loop、process_slot、同步與結果組裝。',
        '- [`src/continuous_scheduler.cpp`](../src/continuous_scheduler.cpp)：鎖內輸入複製、deadline、callback／future 完成順序。',
        '- [`src/onnx_runner.cpp`](../src/onnx_runner.cpp)：單 session、user stream、HEURISTIC／TF32／NHWC 設定與同步 Run。',
        '- [`src/cuda_nms.cu`](../src/cuda_nms.cu)：逐影像排序、單 thread greedy、Pose／Seg kernel。',
        '- [ORT CUDA provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)：NHWC、cuDNN workspace／演算法及 CUDA Graph 的官方限制。',
        '- [ORT I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)：避免不必要的輸入輸出複製；本引擎已使用 bindings。',
        '- [歷史多模型追查](performance.md)：1920／WDDM／同步移除試驗的版本與範圍。','',
        '本輪機器、版本、來源與模型雜湊、計時摘要：[JSON](fp32_fp16_latency.json)。',
        '原始每請求 JSONL 與 logs 在本機 `runs/latency_audit/`；既有確認資料在 `runs/fp16/fp16/`。','',
        '重現：`python tools/diagnostics/latency_audit.py`（有界 GPU 短診斷）；',
        '`python tools/diagnostics/latency_report.py`（只讀資料與 CPU 製圖）。']
    text='\n'.join(lines)+'\n'
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    (ROOT/'docs/fp32_fp16_latency.md').write_text(text,encoding='utf-8')
    print(f'Report: docs/fp32_fp16_latency.md; {len(rows)} passed cases')


if __name__=='__main__':main()
