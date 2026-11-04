# backend/legacy — `/snav` 後端與定位研究

**導航伺服器不是從這裡啟動。** 導航後端是上一層的 `backend/server/`，啟動方式：

```
cd backend
uvicorn server.server:app --host 0.0.0.0 --port 8000   # 或 python -m server.run_server
```

這個資料夾放的是 `/snav` 感測器導航後端（`topomap_v2` 地圖、`locate_v2` 定位）和定位研究腳本。
研究裡驗證有效的方法（DINOv2 ＋ 物件 ＋ OCR 融合、感測器機率融合）已經搬進 `backend/server/`
（`locate_fusion.py`、`loc_fusion.py`），現在的做法與結果見 [定位方法彙整與測試規劃.md](定位方法彙整與測試規劃.md)。

## 這裡放什麼

| 內容 | 說明 |
|---|---|
| `server/` | `/snav` 後端（含 `topomap_v2`、`locate_v2`、`sensor_nav`），研究腳本 import 這裡的模組；測試在 `server/tests/` |
| `eval/` | 動作準確率 harness，和 `backend/eval/` 同一套，量的是這裡的後端（見 [eval/README.md](eval/README.md)） |
| `locate_photo_v2.py`、`README_locate_v2.md` | 照片定位 CLI 與說明 |
| `build_topomap_v2.py`、`README_topomap_v2.md` | `topomap_v2` 建圖 CLI 與說明 |
| `build_topomap_v2_from_sensors.py`、`csv_to_manifest.py`、`merge_*.py`、`augment_topomap_4dir.py`、`rerun_vlm_detections.py`、`detect_and_annotate.py` | `topomap_v2` 建圖工具（Sensor Lab 資料、四方向照片、合併、重跑偵測） |
| `import_neo4j_to_topomap.py` | 從 Neo4j 匯出 `carrefour_a7_imported.json`（`topomap_v2` 格式） |
| `embed_dino.py`、`build_map_dino_sidecar.py`、`carrefour_a7_dino.pkl` | DINOv2 向量快取，與 A7 地圖的 DINOv2 sidecar |
| `eval_localization_*.py`、`eval_visual.py`、`eval_fusion_v2_compare.py`、`verify_server_fusion.py` | 單張照片定位實驗（物件比對、graph matching、CLIP）、fusion-v2 比較、融合進入點驗證 |
| `test_localization*.py`、`test_realworld_0916.py`、`test_goal_search.py` | 定位與目標搜尋的手動測試 |
| `analyze_pdr.py` | PDR 感測器資料分析 |
| `verify_gateway.py`、`verify_gateway.ps1` | 檢查 CGU LLM gateway 能不能用 |
| `requirements-server.txt` | 這裡的後端用的套件 |
| `定位方法彙整與測試規劃.md`、`定位檔案索引.md` | 定位方法的結論、結果，與相關檔案索引 |

**不在 git 裡的資料**（`.gitignore`）：`testPDR/`、`benchmark_results/`、`*.bak-*`，以及
`carrefour_a7_imported.json`（61 MB，用 `import_neo4j_to_topomap.py` 重新產生）。
多數研究腳本要用到這些資料（例如 `eval_fusion_v2_compare.py` 要
`benchmark_results/realworld_0916/dino_sigs.pkl` 和 `carrefour_a7_imported.json`），沒有的話跑不起來。

## 怎麼跑

腳本會把自己所在的資料夾加進 `sys.path`，所以 `import server...` 用的是這裡的模組，不會跟上一層的
`backend/server/` 混在一起。

```
cd backend/legacy
python eval_fusion_v2_compare.py              # Windows：..\venv\Scripts\python.exe eval_fusion_v2_compare.py
python -m pytest server/tests eval/tests -q
```

目前 161 個通過、11 個失敗、7 個錯誤：`test_locate_v2.py` 2 個失敗＋7 個錯誤（原因見
[README_locate_v2.md](README_locate_v2.md)「測試」），`test_vlm.py` 6 個、`test_goal_decomposer.py` 2 個、
`test_phase1.py` 1 個失敗：這些測試是照早期的設定寫的（例如 `test_vlm.py` 預期 Ollama 後端、
`test_phase1.py` 預期 banana 不會被同義詞表改成 fruit），和現在的程式不合。
