"""Summarize the isolated 100-image Detect INT8 experiment; no GPU work."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/int8_detect'


def main():
    def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
    def median(rows,key):return statistics.median(r[key] for r in rows)
    preparation=read('preparation.json')
    assert len(set(preparation['calibration_ids']))==100
    assert len(set(preparation['evaluation_ids']))==100
    assert not set(preparation['calibration_ids']) & set(preparation['evaluation_ids'])
    backends={k:read('trt_'+k+'.json') for k in ('ort-fp16','fp16','int8')}
    native=[read(f'speed_{k}_{n}.json') for k in ('fp16','qdq') for n in (1,16,32)]
    layers=read('trt_int8_layers.json')['Layers']
    convolutions=[layer for layer in layers if 'conv' in layer.get('LayerType','').lower()]
    int8_convolutions=[layer for layer in convolutions if layer.get('Weights',{}).get('Type')=='Int8']
    rows=[]
    for kind,result in backends.items():
        assert result['ap_images']==100
        for core,prototype in zip(result['measured'],result['prototype']):
            rows.append(dict(backend=kind,batch=core['batch'],
                core_fps=median(core['rounds'],'fps_core_wall'),
                gpu_batch_ms=median(core['rounds'],'mean_batch_gpu_ms'),
                prototype_fps=median(prototype['rounds'],'fps'),
                prototype_p95_batch_ms=median(prototype['rounds'],'p95_batch_latency_ms'),
                ap_percent=result['ap50_95']*100))
    ap_drop=(backends['fp16']['ap50_95']-backends['int8']['ap50_95'])*100
    result=dict(model='YOLOv8s Detect',imgsz=[640,640],calibration_images=100,evaluation_images=100,
        disjoint=True,seed=20261018,convolutions=len(convolutions),int8_convolutions=len(int8_convolutions),
        same_backend_ap_drop_percentage_points=ap_drop,core_and_prototype=rows,
        current_engine_end_to_end=[dict(kind=r['kind'],lanes=r['lanes'],fps=r['median_fps']) for r in native],
        provenance=preparation,placement=read('placement.json'),
        production_int8_implemented=False,full_ap_run=False,long_benchmark_run=False)
    (OUT/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Detect INT8 可行性短測（2026-10-18）','',
        'Windows／RTX 3070 8GB，YOLOv8s Detect、640×640、batch 1／8。',
        'COCO val2017 固定 seed=20261018：100 張校正、另外 100 張評估，完全不重疊。',
        '每項速度以 100 張真實影像重複三輪；暖機不計分。沒有完整 AP 或長跑分。','',
        '## 相同推論流程的後端比較','',
        '| 後端 | Batch | 核心 FPS | GPU batch ms | 同步原型 FPS | 原型 P95 batch ms | 100 張 AP50–95 (%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['backend']} | {r['batch']} | {r['core_fps']:.2f} | {r['gpu_batch_ms']:.3f} | "
                     f"{r['prototype_fps']:.2f} | {r['prototype_p95_batch_ms']:.2f} | {r['ap_percent']:.4f} |")
    lines+=['',
        '核心 FPS：GPU 已有 NCHW FP32 輸入，CPU enqueue 至 GPU 完成；不含前後處理與 H2D/D2H。',
        'GPU batch ms：CUDA events；補零不計入成功影像。batch 8 的 100 張實際執行 104 個位置。',
        '同步原型 FPS：預載 RGB → CPU letterbox/normalize → H2D → 推論 → GPU NMS/座標還原 → CPU boxes。',
        '三個後端共用完全相同的原型程式；此原型未接上 vONNX C++ 有界佇列、雙槽、批次排程。',
        '原型延遲是整批延遲，不是 16／32 路的逐請求延遲；不可冒充完整引擎併發成績。','',
        '## 真正的 INT8 執行證據','',
        f'TensorRT layer inspector：{len(int8_convolutions)}/{len(convolutions)} 卷積權重及整數路徑為 INT8；其餘為浮點。',
        'INT8 的量化設定為 S8S8、per-channel weights、MinMax PTQ；DFL 與最後框／分類輸出層保持浮點。',
        'TensorRT 10.13.3.9，FP16 與 INT8 皆使用 builder optimization level 1、workspace 1024 MiB，TF32 關閉。',
        'batch profile=1/8/8。建置不計推論時間，engine、layer inspector、模型與二進位雜湊留在本機。','',
        '## 目前 vONNX CUDA 路徑的完整請求短測','',
        '| 路數 | FP16 FPS | QDQ 模型 FPS |','|---:|---:|---:|']
    for lanes in (1,16,32):
        a=next(r for r in native if r['kind']=='fp16' and r['lanes']==lanes)
        b=next(r for r in native if r['kind']=='qdq' and r['lanes']==lanes)
        lines.append(f"| {lanes} | {a['median_fps']:.2f} | {b['median_fps']:.2f} |")
    lines+=['',
        '這裡每輪 25 次暖機排空、100 張計分、重複 3 輪。16／32 路 batch 上限 8、等待 2 ms、雙槽、fixed shape。',
        '詳細 timing 關閉、CPU boxes 必須可用、保留 frame/stream ID；下載／解碼／初始化／轉換不計時。',
        'CUDA profiler 觀察到 64 個浮點 Conv，沒有整數卷積；Q/DQ 與部分 CPU 常數反量化造成額外工作。',
        '這項 QDQ 結果不能視為真正 GPU INT8 的成績。',
        '目前引擎只辨識浮點圖配置；含 INT8 權重的 QDQ 圖仍可能顯示 fp32，此實驗不新增正式精度承諾。','',
        '## 判讀與下一步','',
        f'相對同後端 TensorRT FP16，這 100 張的 INT8 AP 下降 {ap_drop:.4f} 個百分點。',
        '這是初步樣本結果，不能宣稱已通過完整 COCO 的 0.5 個百分點準確度門檻。',
        '若繼續，先比較同後端 FP16／INT8，再研究較保守的逐層量化與獨立校正資料；',
        '有收益後才討論把 TensorRT provider、常駐 bindings、並發排程與校正快取接入正式引擎。',
        '此次只新增研究腳本，未改正式 API、未發布、未開始完整 INT8 製作。','',
        '## 重現與來源','',
        '```powershell',
        'python tools/diagnostics/int8_probe.py',
        '# TensorRT runtime 以 --target runs/int8_detect/trt_runtime 隔離安裝，未改正式 venv。',
        'python tools/diagnostics/int8_trt_probe.py --kind fp16',
        'python tools/diagnostics/int8_trt_probe.py --kind int8',
        'python tools/diagnostics/int8_trt_probe.py --kind ort-fp16',
        'python tools/diagnostics/int8_report.py','```','',
        '- [ORT GPU 量化](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)',
        '- [Ultralytics TensorRT INT8 匯出](https://docs.ultralytics.com/integrations/tensorrt)',
        '- [NVIDIA pip 安裝與 runtime 選擇](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html)',
        'Ultralytics 可使用 TensorRT INT8 匯出；這次沒有另測官方 INT8 predictor。',
        '完整逐請求記錄、layer inspector、runtime 版本與來源雜湊見本機 `runs/int8_detect/`。']
    text='\n'.join(lines)+'\n'
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    (ROOT/'docs/int8_research.md').write_text(text,encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
