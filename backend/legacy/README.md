# backend/legacy — 舊後端與定位研究

**這裡不是導航後端，不要從這裡啟動伺服器。**
導航後端是上一層的 `backend/server/`（來自 APPNAV_ios 的共用後端），啟動方式：

```
cd backend
python -m server.run_server
```

## 這裡放什麼

| 內容 | 說明 |
|---|---|
| `server/` | 舊後端（Android 原本用的版本，含 `/snav` 感測器導航、`topomap_v2`、`locate_v2`）。只留給下面的研究腳本 import，不再當伺服器用 |
| `eval_localization_*.py`、`eval_visual.py`、`eval_fusion_v2_compare.py`、`verify_server_fusion.py` | 單張照片定位實驗、fusion-v2 比較與 k(σ) 交叉驗證 |
| `embed_dino.py`、`build_map_dino_sidecar.py` | 產生 DINOv2 向量快取與參考向量檔 |
| `import_neo4j_to_topomap.py` | 從 Neo4j 匯出 `carrefour_a7_imported.json` |
| `定位方法彙整與測試規劃.md`、`定位檔案索引.md` | 定位實驗的結論與檔案索引 |
| `build_topomap_v2*.py`、`merge_*.py`、`augment_topomap_4dir.py` 等 | 舊建圖工具 |
| `testPDR/`、`benchmark_results/`、`carrefour_a7_*.json/.pkl` | 實驗資料（不進 git） |

## 怎麼跑研究腳本

腳本會把自己所在的資料夾加進 `sys.path`，所以 `import server...` 會用到這裡的舊後端，不會跟上一層的共用後端混在一起。

```
cd backend/legacy
../venv/Scripts/python.exe eval_fusion_v2_compare.py
```

在這裡驗證有效的定位方法，要搬進上一層的 `backend/server/` 才會用在導航上。
