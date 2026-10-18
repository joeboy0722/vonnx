# vONNX v0.5.0

新增預設關閉的初始化圖專化與ORT CUDA Graph，並修正提交及ORT生命週期問題。

- Python／C++新增 `graph_specialization="off"|"auto"`、`cuda_graph=False`；C API新增v4，保留v1～v3布局。
- 專化已知尺寸／常數，動態batch仍可使用；Graph要求固定執行batch，各槽初始化capture、常駐重用。
- CPU副本移出提交大鎖、保留FIFO與pause/close排空；同批futures先發布，再執行callbacks。
- 修正最後引擎close後的ORT logger與Windows DLL退出生命週期問題；保留ORT結尾安全同步。
- 三種s／640、FP32／FP16、三模式共18設定，各500張COCO抽樣AP通過，最大差距0.0900個百分點。
- s／640三輪短測相對修改前工作引擎：FP16單路6.64→3.88ms，FP16／32路317→361FPS；1920三模型只有約2%收益，沒有延遲隔離。

AP為500張抽樣回歸，並非完整資料集驗收；未執行長時間跑分。其他模型／尺寸、FP16 X數值差異與GPU共享限制詳見報告。

Windows x64、CPython3.12；需要CUDA13 runtime、ORT GPU1.28.0及相容cuBLAS／cuDNN9。二進位資產不包含模型或第三方runtime DLL。
引擎MIT授權，LICENSE與第三方聲明包含於wheel／SDK。C++呼叫者須重編譯；Python與C部署須整套更新配套工具。

- [安裝與調用](https://github.com/joeboy0722/vonnx/blob/v0.5.0/docs/release.md)
- [圖專化／Graph設定](https://github.com/joeboy0722/vonnx/blob/v0.5.0/docs/graph_specialization.md)
- [本版500張AP報告](https://github.com/joeboy0722/vonnx/blob/v0.5.0/docs/ap_v050.md)
- [跨模型等待與短測報告](https://github.com/joeboy0722/vonnx/blob/v0.5.0/docs/software_wait.md)

資產：Python3.12 wheel、原生C/C++ SDK、使用指南、AP JSON與SHA256SUMS。
