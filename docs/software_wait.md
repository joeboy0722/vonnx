# 跨模型軟體等待改善：實作與短測

日期：2026-10-18。本報告記錄v0.5.0發布前的效能短測，當時候選版本字串仍為0.4.0；辨識以二進位雜湊為準。
本階段未執行500次效能確認或長時間矩陣；後續發布前另完成 [500張抽樣AP回歸](ap_v050.md)。

## 實作

- 新增預設關閉的 graph_specialization=auto；固定已知高／寬與固定執行 batch，保留動態 batch、補零與逐請求結果。運行期間每引擎一個常駐session；初始化有暫時GPU探測session。
- 模型準備與 FP16 轉換共用帶來源／外部權重、尺寸、精度、batch、工具及規則指紋的原子快取，保留來源檔。探測／binding／暖機使用一致的實際 shape。
- 新增預設關閉的 cuda_graph，要求專化與固定執行 batch；每槽獨立 graph ID／固定 I/O 位址，capture 與暖機在初始化完成，ORT 結尾同步保留。CPU fallback、初始化失敗或預算不足明確拒絕。
- 新增 Python／C++ 設定與 ModelSpec 資訊、C API v4。v1／v2／v3 布局不變，C++ 呼叫者需重編譯。
- 修正 ORT Env／logger 生命週期。引擎 close 釋放 session／arena／CUDA 資源，保留每個載入模組一份的程序生命週期 Env，避免 Python ORT 在最後引擎 close 後 logger 失效，以及 Windows DLL 退出解載順序崩潰。

## 三組基準與量測

正式 v0.4.0、封存的修改前工作目錄引擎、候選引擎分別載入實際二進位，來源 PT／ONNX 雜湊與準備後模型雜湊保留在每案 result.json。來源版本號相同不代表二進位相同。

| 組別 | Python native SHA256 |
|---|---|
| candidate | `a98a4054be3b4804005df062a6415727c34b81c7d76267001f3226b19d897b9a` |
| current | `83bb3275045805d8086de06805811741bce56fd21df31d196047056fecd7f710` |
| release | `55fbcfeab264247356752880a13e4d7a70720cb72ed3439703f1cbcd1f4c8948` |

主測使用預載 COCO8 Detect 真實影像輪播，原圖1920×1080、固定推論1920×1920，X／L／n 各獨立引擎，FP32／FP16、B1、等待0、單槽、每模型1／4路。每模型25次暖機排空後最多75次計分，每路最多一筆未完成；固定配額與6秒上限飽和負載分開。
計入提交副本到 CPU 結果可用與所有尾段，不刪慢樣本。固定配額的逐模型 FPS 以共同起點到該模型完成75次計算，整組吞吐則含最後模型排空。飽和案例另外呈現共同活動窗口，不能把模型完成更多次數誤報為每模型都加速。

## 三模型總吞吐（全部請求、包含排空）

| 精度 | 負載 | 每模型路數 | Release | 修改前目前引擎 | 候選 |
|---|---|---:|---:|---:|---:|
| fp32 | fixed | 1 | 7.30 | 7.22 | 7.31 |
| fp32 | fixed | 4 | 7.34 | 7.32 | 7.33 |
| fp32 | saturated | 1 | 7.37 | 7.35 | 7.37 |
| fp32 | saturated | 4 | 7.35 | 7.35 | 7.36 |
| fp16 | fixed | 1 | 15.05 | 14.66 | 15.03 |
| fp16 | fixed | 4 | 14.81 | 14.68 | 15.00 |
| fp16 | saturated | 1 | 15.27 | 14.71 | 15.08 |
| fp16 | saturated | 4 | 15.24 | 14.71 | 15.00 |

單位為次推論／秒，X／L／n 各算一次。Release與FP32／1920只有首輪對照；FP16／1920的修改前與候選完成三輪，才進行正式中位數／勝出門檻判定。FP32首輪收益小且P95未全面下降，不延長確認。主測gpu_samples保存溫度、SM／記憶體時脈與utilization，不以時脈校正替代實際成績。

## 逐模型變化與驗收

| 精度 | 負載 | 路數 | 模型 | 輪數／勝出 | 平均延遲變化 | P95變化 | 完成FPS變化 | 3%回歸門檻 | 三輪改善門檻 |
|---|---|---:|---|---|---:|---:|---:|---|---|
| fp32 | fixed | 1 | yolov8x | 1／1 | -1.3% | -1.1% | +1.3% | 通過 | 未通過或未完成 |
| fp32 | fixed | 1 | yolov8l | 1／1 | -1.3% | -1.1% | +1.3% | 通過 | 未通過或未完成 |
| fp32 | fixed | 1 | yolov8n | 1／1 | -1.3% | -1.1% | +1.3% | 通過 | 未通過或未完成 |
| fp32 | fixed | 4 | yolov8x | 1／0 | -0.1% | +0.5% | +0.1% | 通過 | 未通過或未完成 |
| fp32 | fixed | 4 | yolov8l | 1／0 | -0.1% | +0.4% | +0.1% | 通過 | 未通過或未完成 |
| fp32 | fixed | 4 | yolov8n | 1／0 | -0.1% | +0.4% | +0.1% | 通過 | 未通過或未完成 |
| fp32 | saturated | 1 | yolov8x | 1／1 | -0.3% | -0.4% | +0.3% | 通過 | 未通過或未完成 |
| fp32 | saturated | 1 | yolov8l | 1／1 | -0.3% | -0.3% | +0.3% | 通過 | 未通過或未完成 |
| fp32 | saturated | 1 | yolov8n | 1／1 | -0.3% | -0.3% | +0.3% | 通過 | 未通過或未完成 |
| fp32 | saturated | 4 | yolov8x | 1／1 | -0.1% | -0.3% | +0.1% | 通過 | 未通過或未完成 |
| fp32 | saturated | 4 | yolov8l | 1／1 | -0.1% | -0.3% | +0.1% | 通過 | 未通過或未完成 |
| fp32 | saturated | 4 | yolov8n | 1／1 | -0.1% | -0.3% | +0.1% | 通過 | 未通過或未完成 |
| fp16 | fixed | 1 | yolov8x | 3／3 | -2.5% | -3.0% | +2.5% | 通過 | 通過 |
| fp16 | fixed | 1 | yolov8l | 3／3 | -2.5% | -3.3% | +2.5% | 通過 | 通過 |
| fp16 | fixed | 1 | yolov8n | 3／3 | -2.5% | -3.7% | +2.5% | 通過 | 通過 |
| fp16 | fixed | 4 | yolov8x | 3／3 | -2.3% | -2.5% | +2.1% | 通過 | 通過 |
| fp16 | fixed | 4 | yolov8l | 3／3 | -1.9% | -2.1% | +2.0% | 通過 | 通過 |
| fp16 | fixed | 4 | yolov8n | 3／3 | -2.3% | -2.4% | +2.1% | 通過 | 通過 |
| fp16 | saturated | 1 | yolov8x | 3／3 | -2.4% | -3.9% | +2.5% | 通過 | 通過 |
| fp16 | saturated | 1 | yolov8l | 3／3 | -1.5% | -3.9% | +1.6% | 通過 | 通過 |
| fp16 | saturated | 1 | yolov8n | 3／3 | -1.5% | -4.0% | +1.6% | 通過 | 通過 |
| fp16 | saturated | 4 | yolov8x | 3／3 | -1.9% | -2.4% | +2.0% | 通過 | 通過 |
| fp16 | saturated | 4 | yolov8l | 3／3 | -1.9% | -2.4% | +2.0% | 通過 | 通過 |
| fp16 | saturated | 4 | yolov8n | 3／3 | -1.9% | -3.8% | +2.0% | 通過 | 通過 |

負的延遲變化較快，正的 FPS 變化較快。主要情境須三輪至少兩輪勝出、平均／P95中位數下降且FPS不下降；3%是其他回歸情境底線，不是主要改善成功條件。低於5%的改善為小幅收益。

## s／640 單模型回歸（三輪中位數）

單路B1／等待0／單槽；16／32路為固定B8／等待2ms／雙槽。每設定25次暖機排空、75次計分。Release為首輪對照，修改前與兩種候選各三輪。

| 精度 | 路數 | Release FPS | 修改前 FPS | 圖專化 FPS | 圖專化＋Graph FPS |
|---|---:|---:|---:|---:|---:|
| fp32 | 1 | 144.6 | 139.0 | 142.5 | 157.1 |
| fp32 | 16 | 206.8 | 200.6 | 201.7 | 203.1 |
| fp32 | 32 | 207.9 | 199.8 | 201.8 | 202.8 |
| fp16 | 1 | 156.0 | 150.2 | 222.8 | 256.2 |
| fp16 | 16 | 335.5 | 314.9 | 339.4 | 363.2 |
| fp16 | 32 | 331.7 | 317.1 | 351.1 | 361.2 |

| 精度 | 路數 | 修改前平均／P95 ms | 圖專化平均／P95 ms | 圖專化＋Graph平均／P95 ms |
|---|---:|---:|---:|---:|
| fp32 | 1 | 7.17／7.55 | 6.99／7.66 | 6.34／6.80 |
| fp32 | 16 | 70.80／75.39 | 70.60／76.07 | 70.05／74.76 |
| fp32 | 32 | 126.15／150.44 | 124.99／149.43 | 124.10／148.42 |
| fp16 | 1 | 6.64／7.27 | 4.47／5.01 | 3.88／4.36 |
| fp16 | 16 | 45.36／49.65 | 42.05／45.67 | 39.25／42.08 |
| fp16 | 32 | 79.96／97.55 | 72.01／85.98 | 69.87／83.16 |

| 精度 | 路數 | 候選模式 | 勝出輪數 | 平均／P95／FPS變化 | 3%回歸門檻 |
|---|---:|---|---:|---|---|
| fp32 | 1 | 圖專化 | 1／3 | -2.5%／+1.4%／+2.5% | 通過 |
| fp32 | 1 | 圖專化＋Graph | 3／3 | -11.6%／-10.0%／+13.0% | 通過 |
| fp32 | 16 | 圖專化 | 1／3 | -0.3%／+0.9%／+0.5% | 通過 |
| fp32 | 16 | 圖專化＋Graph | 3／3 | -1.1%／-0.8%／+1.2% | 通過 |
| fp32 | 32 | 圖專化 | 2／3 | -0.9%／-0.7%／+1.0% | 通過 |
| fp32 | 32 | 圖專化＋Graph | 3／3 | -1.6%／-1.3%／+1.5% | 通過 |
| fp16 | 1 | 圖專化 | 3／3 | -32.7%／-31.0%／+48.3% | 通過 |
| fp16 | 1 | 圖專化＋Graph | 3／3 | -41.5%／-40.1%／+70.6% | 通過 |
| fp16 | 16 | 圖專化 | 3／3 | -7.3%／-8.0%／+7.8% | 通過 |
| fp16 | 16 | 圖專化＋Graph | 3／3 | -13.5%／-15.3%／+15.3% | 通過 |
| fp16 | 32 | 圖專化 | 3／3 | -9.9%／-11.9%／+10.7% | 通過 |
| fp16 | 32 | 圖專化＋Graph | 3／3 | -12.6%／-14.8%／+13.9% | 通過 |

首版測試端採每路固定請求配額，較快的路徑可能在共同75次尚未發完時提早退出，造成尾段B8被拆成兩批。某FP32／32路因此出現10批與11批的差異。原始資料保留於single_per_lane_quota；改為共同配額後三組全部重測，每路仍至多一筆未完成，所有尾批與補零成本保留。single_summary.csv僅使用shared_quota_v2資料。

## 等待原因

1. **尚未提交**：動態 shape／資料相依會使 CPU 中途等資料，再提交後段 kernel。固定 shape 可以移除其中可預計算的工作。
2. **已提交但 GPU 未完成**：其他 stream 的 kernel、GPU資源與驅動排程仍可能使 n 等待；不能把所有非 kernel 空隙都算成可移除 CPU overhead。
3. **GPU完成到結果回傳**：需要完成 n 自己的後處理與D2H，再發布future；檢查是否存在無關模型的結果屏障。

未發現X完成後n才能發布future的顯式跨引擎屏障。實際軟體問題是可預計算的shape／常數工作在GPU與CPU間往返，造成中途等待，並被其他模型的GPU排隊放大。三模型專化後，n自己的kernel仍受GPU／驅動排程影響；ORT返回後自己的後處理／D2H也會競爭資源，不能宣稱n端到端已降到自身kernel總執行時間。
診斷trace啟用詳細timing，不混入主要成績。先前只專化n的57.9ms結果是不同模型準備組合，不冒充三模型全部專化的效益。

## 最終二進位的trace與限制

以下為診斷資料，詳細timing開啟，不混入主要跑分。

| n／1920指標 | 修改前 | 圖專化候選 |
|---|---:|---:|
| CPU event等待次數／推論 | 26 | 1 |
| 最後kernel的CPU提交距前處理結束ms | 190.16 | 6.14 |
| 自己kernel累計執行ms | 15.94 | 16.64 |
| GPU推論跨度ms | 199.15 | 169.16 |

即使n的工作約6ms已提交至尾端，GPU推論跨度仍約169ms。這份trace無法將SM／記憶體資源與WDDM驅動排程的比例完全拆開，不宣稱已達硬體極限，也不承諾7ms模型併發後最多14ms。

原生引擎s／640、雙槽B8的測量段觀察到 10 次cudaGraphLaunch；未見capture、instantiate、cudaMalloc／Free或EventCreate，確認capture在初始化完成。這不保證第三方驅動內部完全沒有配置。

額外1920固定配額診斷：全部開Graph約14.89／14.90次每秒（1／4路）；compute與copy work queue都設32約14.90／14.93。本機未見優於預設的證據，因此不自動調整CUDA環境或把Graph當作1920隔離方案。
NVIDIA說明不同stream映射同一work queue可能產生假相依，連線數可調1～32、預設8；本輪僅在獨立子程序測32，不改系統設定。[NVIDIA CUDA環境變數說明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-max-connections)。

## 準備順序與數值診斷

| 準備方式 | CPU節點數 | MemcpyToHost節點 | 相對同精度動態圖原始輸出最大差 |
|---|---:|---:|---:|
| fp32_dynamic | 43 | 0 | 0 |
| fp32_shape | 0 | 0 | 0 |
| fp16_dynamic | 0 | 17 | 0 |
| fp16_shape_before | 0 | 0 | 0 |
| fp16_shape_after | 0 | 0 | 0 |

此為n／1920單圖的EP profile。FP16的CPU節點數為0仍可能有給形狀輸入使用的往返拷貝，不能用CPU節點數0推論完全沒有CPU等待。先專化再轉FP16、先轉再專化在本圖原始輸出一致；正式流程採先專化再轉換。

1920 X：同ONNX獨立ORT原始輸出最大差 `0.0`，最終框最大差 `0.0` pixel。
相對來源FP32，vONNX FP16最大座標差 0.779 pixel，官方FP16 1.390 pixels。兩個半精度流程彼此仍可差2 pixels；官方raw半精度輸出先轉FP32後處理，差異降為 1.500 pixels，顯示包含半精度模型解碼及後處理數值路徑差異。
保持vONNX的FP32解碼／後處理，不刻意貼齊官方半精度捨入。以上僅一張影像診斷，未通過正式AP，也不宣稱兩流程已符合逐框1pixel一致。

## 功能與控制測試

完整Python套件 44 項／57.699秒通過，程序退出碼 0；另補測C API v4 Graph之外部GPU producer（延後寫入）1項通過。C++scheduler與真實引擎callback阻塞控制亦通過。

目前保存 25 組真實模型小樣本檢查；每組二進位雜湊在summary.json中明列。不同候選建置的結果不當作同一二進位完整驗收。
測試包含固定／動態batch、矩形尺寸、Detect／Pose／Seg、原生FP16 I/O、快取命中／失效／並行準備、v1/v2/v3相容與v4、空結果、ID、pause/resume資源重用及程序退出。
純C++真實引擎fixture控制：X callback刻意阻塞並拋例外時，n仍能完成；獨立scheduler控制及Python host延後／pause測試另有記錄。這些是相依控制，不是YOLO效能數字。

## 使用與重現

介面與限制見 [圖專化說明](graph_specialization.md) 及 examples/graph_specialization.py。
```powershell
.\venv\python.exe tools\diagnostics\software_wait.py --phase functional
.\venv\python.exe tools\diagnostics\software_wait.py --phase main
.\venv\python.exe tools\diagnostics\software_wait.py --phase main --variant current --repeat 2
.\venv\python.exe tools\diagnostics\software_wait.py --phase main --variant candidate --repeat 2
.\venv\python.exe tools\diagnostics\software_wait.py --phase single
.\venv\python.exe tools\diagnostics\software_wait.py --phase single --variant candidate --graph
.\venv\python.exe tools\diagnostics\software_wait_report.py
```

每筆資料、NVML樣本、來源／產物雜湊、錯誤、trace與報告在 runs/software_wait/。summary.csv 包含每模型平均／P95／P99、FPS、時脈與溫度；groups.csv 為整組吞吐。
尚未滿足的門檻、未定位的問題與未執行的驗證維持未完成；本版發布不代表這些限制已解決。
