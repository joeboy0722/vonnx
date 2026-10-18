"""Check and export the v0.5.0 sampled AP release gate; CPU only."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vonnx_tools.common import build_fingerprint, sha256, write_json, manifest


def main():
    folder=ROOT/'runs/v050/ap500'
    rows=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    expected={(m,p,s,g) for m in ('yolov8s','yolov8s-pose','yolov8s-seg')
              for p in ('fp32','fp16') for s,g in (('off',False),('auto',False),('auto',True))}
    if len(rows)!=18:raise ValueError('Require all 18 configurations')
    got=set(); exported=[]; ids_by_model={}
    binary=build_fingerprint(); converter=sha256(ROOT/'vonnx/precision.py')
    for row in rows:
        c=row['case']; d=row['detail']; model=c['model']
        got.add((model,c['precision'],c['graph_specialization'],c['cuda_graph']))
        if (row['status']!='passed' or c['build_fingerprint']!=binary or
            c['conversion_fingerprint']!=converter or c['imgsz']!=[640,640] or
            c['model_hash']!=manifest(model)['onnx_sha256'] or
            c['limit']!=500 or d['images']!=500 or len(set(d['image_ids']))!=500 or not d['limited']):
            raise ValueError('Incomplete/stale AP gate: '+c['id'])
        prepared=d['prepared_model']
        if prepared['graph_specialized']!=(c['graph_specialization']=='auto') or prepared['cuda_graph']!=c['cuda_graph']:
            raise ValueError('Prepared model mode does not match requested mode')
        kinds={'bbox'} | ({'keypoints'} if model.endswith('-pose') else {'segm'} if model.endswith('-seg') else set())
        if set(d['metrics'])!=kinds:raise ValueError('Missing AP metric')
        if model in ids_by_model and ids_by_model[model]!=d['image_ids']:
            raise ValueError('Mismatched subset')
        ids_by_model[model]=d['image_ids']
        for metric in d['metrics'].values():
            if (not all(math.isfinite(metric[k]) and 0<=metric[k]<=1 for k in ('engine_ap','official_ap'))
                or abs(metric['engine_ap']-metric['official_ap'])>.005+1e-12):
                raise ValueError('AP difference exceeds 0.5 percentage points')
        exported.append(dict(case=c,metrics=d['metrics'],images=d['images'],
            image_ids_sha256=hashlib.sha256(json.dumps(d['image_ids']).encode()).hexdigest(),
            prepared_model=d['prepared_model'],model=row['model'],elapsed_seconds=row['elapsed_seconds']))
    if got!=expected:raise ValueError('Missing configurations')
    result=dict(passed=True,scope='500-image deterministic COCO val2017 subset per task; not full AP',
                seed=20261011,binary_sha256=binary,converter_sha256=converter,
                sdk_dll_sha256=sha256(ROOT/'vonnx_standalone_sdk/vonnx.dll'),
                environment=rows[0]['environment'],allowed_difference_percentage_points=.5,
                reference_precision='official PyTorch FP32',imgsz=[640,640],batch=1,
                confidence=.001,iou=.7,max_det=300,rect=False,models=exported)
    write_json(ROOT/'docs/ap_v050.json',result)
    lines=['# v0.5.0：500 張抽樣 AP 驗證','',
        '三種 YOLOv8 s、640×640；FP32／FP16 × 原模式／圖專化／圖專化＋CUDA Graph，共18個設定。',
        '每設定使用500張固定抽樣的COCO val2017影像；Pose先篩選有有效關鍵點標註的影像，再抽500張。',
        '對照相同權重的官方PyTorch FP32，AP50–95差距皆不超過0.5個百分點。',
        '**這是抽樣回歸驗證，並非完整COCO AP，不代表其他模型／尺寸均已通過。**','',
        '| 模型 | 精度 | 模式 | 指標 | vONNX AP (%) | 官方 FP32 AP (%) | 差異（百分點） |',
        '|---|---|---|---|---:|---:|---:|']
    for r in exported:
        c=r['case']; mode='Graph' if c['cuda_graph'] else '專化' if c['graph_specialization']=='auto' else '原模式'
        for kind,m in r['metrics'].items():
            lines.append(f"| {c['model']} | {c['precision']} | {mode} | {kind} | {m['engine_ap']*100:.4f} | {m['official_ap']*100:.4f} | {(m['engine_ap']-m['official_ap'])*100:+.4f} |")
    lines+=['','固定設定：batch=1、confidence=0.001、IoU=0.7、max_det=300、rect=False。',
        'Seg使用原圖ROI mask，評估前重建；官方retina_masks=True。COCOeval使用標準maxDets。',
        f'二進位SHA256：`{binary}`。',f'模型準備工具SHA256：`{converter}`。','',
        '模型／工具／環境與抽樣ID指紋見 [JSON明細](ap_v050.json)。',
        '逐影像預測、IDs與完整log保留於本機 `runs/v050/ap500/`。',
        'v0.4.0的完整資料集結果僅作歷史記錄，見 [舊版AP報告](ap_fp16.md)。','',
        '重現：`python tools/diagnostics/ap_release.py --resume`，完成後執行 `python tools/diagnostics/ap_release_report.py`。',
        '本次未執行長時間效能跑分。']
    (ROOT/'docs/ap_v050.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PASS: 18 configurations, 500 images each')


if __name__=='__main__':main()
