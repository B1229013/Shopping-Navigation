#!/usr/bin/env python3
"""
locate_photo_v2.py — 給一張新照片，推理它在既有拓樸地圖 v2 裡最可能是哪個位置
Given a new photo, infer which node in an existing place-level TopoGraphV2
it most likely corresponds to.

════════════════════════════════════════════════════════════════════════════
用途 (Purpose)
════════════════════════════════════════════════════════════════════════════
這是拓樸地圖 v2（見 `server/topomap_v2.py` / `build_topomap_v2.py`）的
配套測試工具，用來驗證「給一張新照片，能不能在既有地圖裡推理出它可能
位於哪個位置」這件事。

跟 `build_topomap_v2.py` 一樣，先做成「不跟系統整合」的獨立離線工具：
不需要啟動 FastAPI server。預設吃的是「已經算好的偵測結果」（格式跟
`detections_{node_id}.json` 相同），這樣不需要 GPU／模型權重也能測試
比對演算法本身；如果你的環境剛好有裝好系統既有的 GroundingDINO/OCR，
也可以不給 --detections、只給 --photo，讓這支腳本自己跑一次偵測（見
`_auto_detect()`）——沒裝好的話會印出清楚的錯誤訊息，請改用 --detections。

比對邏輯細節見 `server/locate_v2.py` 開頭的說明。

════════════════════════════════════════════════════════════════════════════
使用方式 (Usage)
════════════════════════════════════════════════════════════════════════════
    # 用既有的偵測結果（例如另一次導航的 annotated/detections_5.json）
    python locate_photo_v2.py \\
        --map output/topomaps/資工系系辦/topomap.json \\
        --detections output/sessions/<session_id>/annotated/detections_5.json \\
        --photo output/sessions/<session_id>/photo/5.jpg

    # 環境有裝好模型的話，讓腳本自己偵測——預設用「兩階段 VLM」（見
    # --detect-mode 說明），因為系統實際建圖已經改用這個模式，不再是
    # GroundingDINO
    python locate_photo_v2.py \\
        --map output/topomaps/資工系系辦/topomap.json \\
        --photo /path/to/new_photo.jpg

    其他選項：
        --top-k 5              只列出前幾名候選（預設 5）
        --min-score 0.05        單一物件配對門檻，跟 locate_v2.py 的評分尺度一致
                                （只濾掉純位置巧合命中的雜訊配對，不用刻意調高）
        --debug-png locate_queries/debug.png
                                另存一張除錯用的地圖圖片：把查詢照片畫成
                                地圖上的一個節點，接在分數最高的候選節點
                                旁邊，並用線把配對到的物件連起來，方便
                                直接用眼睛檢查比對結果、比文字報告更直觀
        --allow-structural      也允許用門/牆/窗等環境結構物做比對（預設關閉）
        --json-out result.json  另存一份完整結果 json（預設只印文字報告）
        --detect-mode vlm|groundingdino|auto
                                --photo 自動偵測要用哪條路徑（預設 vlm，
                                見下方「偵測模式」說明）
        --goal "尋找電腦"        可選：VLM 模式下要不要帶入導航目標（跟
                                系統實際導航時的 goal 概念一樣，沒有的話
                                留空即可，通用場景描述不受影響）

════════════════════════════════════════════════════════════════════════════
偵測模式 (Detection modes)
════════════════════════════════════════════════════════════════════════════
系統本身依 GroundingDINO 有沒有成功載入，自動切換兩套完全不同的偵測
邏輯（見 `server.py` 的 `has_grounding` 判斷）：

    模式 A（GroundingDINO）：逐一精確框選候選詞清單裡的物件
    模式 B/C（兩階段 VLM）：改用大語言模型直接用文字描述整個畫面，
        物件的畫面位置是「粗略九宮格＋近遠」文字描述，不是精確量測的
        像素座標（模型沒辦法可靠輸出精確座標，這是刻意的設計取捨）

實測發現：如果拿模式 A 重新偵測一張查詢照片，去跟一張用模式 B/C 建出來
的地圖節點比對，即使畫面內容完全一樣，兩邊物件標籤的用詞、精細度也會
差很多（不是同一套方法產生的東西，比不起來，不是環境設定或版本問題）。
目前系統實際建圖已經統一改用模式 B/C（VLM），`--detect-mode` 預設也
跟著改成 `vlm`，讓查詢照片跟地圖節點是同一套方法產生的資料，比對才有
意義。

`--detect-mode auto` 會模擬 `server.py` 的行為：優先嘗試 GroundingDINO，
失敗（沒裝 torch/模型權重）才退回 VLM；`--detect-mode groundingdino`
強制只用 GroundingDINO，失敗就直接報錯，不會偷偷退回 VLM。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓 `from server...` 可用

from server.locate_v2 import format_localization_report, localize_photo, render_localization_debug_png  # noqa: E402
from server.topomap_v2 import TopoGraphV2  # noqa: E402


def _load_detections_file(path: Path) -> Tuple[List[dict], List[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("detections", []), data.get("ocr", [])


def _guess_image_size(photo_path: Optional[Path]) -> Tuple[int, int]:
    """讀原始照片取得寬高；讀不到就給常見手機解析度（跟
    build_topomap_v2.py 的 `_guess_image_size()` 邏輯一致）。"""
    if photo_path and photo_path.exists():
        try:
            from PIL import Image
            with Image.open(photo_path) as im:
                return im.size
        except Exception:
            pass
    return (1600, 1200)


def _bbox_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _ocr_item_center(ocr_item: dict) -> Optional[Tuple[float, float]]:
    """算一筆 OCR 結果的中心點座標，相容兩種常見格式：
    四點多邊形 [[x,y],[x,y],[x,y],[x,y]]（VLM/EasyOCR 常見格式），
    或簡單的 [x1,y1,x2,y2] 矩形。抓不到座標就回傳 None。
    """
    bbox = ocr_item.get("bbox")
    if not bbox:
        return None
    try:
        if isinstance(bbox[0], (list, tuple)):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            return sum(xs) / len(xs), sum(ys) / len(ys)
        if len(bbox) == 4:
            return _bbox_center(bbox)
    except (TypeError, IndexError, ZeroDivisionError):
        pass
    return None


def _associate_ocr_with_detections(det_list: List[dict], ocr_list: List[dict]) -> None:
    """把 OCR 結果關聯到最近的物件，寫進該物件的 "nameplate_text" 欄位
    ——這一步是必要的，`TopoGraphV2.build_subgraph_from_detections()`
    的 `ocr_items` 參數本身**不會**自動幫忙做這件事（該參數目前接了
    但函式內部沒有用到，物件要有 OCR 文字，必須是這個物件的字典裡
    本來就帶著 "nameplate_text" 這個欄位）。跟系統正式流程裡
    `postprocess_detections()` 的「OCR-door」步驟做的是同一件事，只是
    這裡用最簡單的「距離最近就關聯」規則，不追求跟正式系統完全一致。

    每筆 OCR 結果只會關聯給距離最近的一個物件（避免一段文字同時「借」
    給好幾個物件，稀釋掉其他物件應有的判斷）；物件本身如果已經有
    "nameplate_text"（例如 --detections 直接給的資料），不會被覆蓋。
    """
    if not ocr_list or not det_list:
        return

    det_centers = []
    for det in det_list:
        box = det.get("box")
        if box and len(box) == 4:
            det_centers.append(_bbox_center(box))
        else:
            det_centers.append(None)

    for ocr_item in ocr_list:
        text = ocr_item.get("text", "").strip()
        if not text:
            continue
        ocr_center = _ocr_item_center(ocr_item)
        if ocr_center is None:
            continue

        best_idx, best_dist = None, None
        for i, center in enumerate(det_centers):
            if center is None:
                continue
            dist = (center[0] - ocr_center[0]) ** 2 + (center[1] - ocr_center[1]) ** 2
            if best_dist is None or dist < best_dist:
                best_idx, best_dist = i, dist

        if best_idx is not None and not det_list[best_idx].get("nameplate_text"):
            det_list[best_idx]["nameplate_text"] = text


def _auto_detect_groundingdino(photo_path: Path) -> Tuple[List[dict], List[dict]]:
    """模式 A：呼叫系統既有的 GroundingDINO + OCR 模組跑一次——刻意
    不重新寫一套偵測邏輯，而是直接沿用 `server.py` `upload_photo()`
    實際在用的同幾個模組（`server/perception.py` 的 `Perception`、
    `server/ocr.py` 的 `OCR`/`OpenAIOCR`）。

    設定值（GENERIC_INDOOR_OBJECTS、OCR_BACKEND 等）一律用 `getattr()`
    從 `server.config` 模組上安全讀取、讀不到就退回預設值，刻意不用
    `from server.config import NAME` 這種寫法——不同專案版本的
    `config.py` 可能沒有這些名字（例如比較舊的版本、或有人本地改過），
    用 `from ... import` 只要少一個名字整支工具就會直接 ImportError
    連物件偵測都跑不了；用 `getattr` 才能在「這台環境的 config.py 剛好
    跟這支工具當初參考的版本不完全一樣」時仍然堪用（頂多是 OCR
    後端猜錯、退化成本地 EasyOCR，而不是整個失敗）。

    這需要環境裝好 torch / groundingdino 與模型權重（OCR 若用 OpenAI
    後端則需要 OPENAI_API_KEY），在沒裝好的環境下會直接失敗——失敗時
    讓例外往外拋，由呼叫端（`_auto_detect()` 或 `main()`）決定要退回
    VLM 模式還是直接報錯。
    """
    from server import config as _config  # type: ignore
    from server.perception import Perception  # type: ignore

    generic_objects = getattr(
        _config, "GENERIC_INDOOR_OBJECTS",
        ["door", "chair", "table", "desk", "sign", "shelf", "trash can", "plant"],
    )
    ocr_backend = getattr(_config, "OCR_BACKEND", "easyocr")
    ocr_languages = getattr(_config, "OCR_LANGUAGES", ["en", "ch_tra"])
    ocr_min_confidence = getattr(_config, "OCR_MIN_CONFIDENCE", 0.3)
    ocr_max_results = getattr(_config, "OCR_MAX_RESULTS", 15)

    # 這支工具沒有「導航目標」的概念（不像 upload_photo() 會把
    # s.goal_objects 併進 prompt），所以只用系統既有的通用室內物件清單
    # ——這批物件本來就是「不管目標是什麼，都一律會偵測」的基礎詞彙，
    # 拿來做定位比對正合適。
    perception = Perception()
    perception.load()
    detections = perception.detect(str(photo_path), list(generic_objects))
    det_list = [
        {"label": d.label, "box": d.box, "score": d.score, "position": d.position}
        for d in detections
    ]

    ocr_list: List[dict] = []
    try:
        if ocr_backend == "openai":
            from server.ocr import OpenAIOCR  # type: ignore
            ocr_engine = OpenAIOCR(languages=ocr_languages)
        else:
            from server.ocr import OCR  # type: ignore
            ocr_engine = OCR(languages=ocr_languages)
        ocr_engine.load()
        ocr_results = ocr_engine.read(
            str(photo_path), min_confidence=ocr_min_confidence, max_results=ocr_max_results,
        )
        ocr_list = [
            {"text": r.text, "confidence": r.confidence, "bbox": getattr(r, "bbox", None)}
            for r in ocr_results
        ]
        _associate_ocr_with_detections(det_list, ocr_list)
    except Exception as exc:  # noqa: BLE001
        # OCR 失敗不擋整體偵測——跟物件偵測不同，OCR 只是加分項，這裡
        # 印出警告但繼續（呼叫端仍然能用純物件偵測結果做比對）。
        print(f"[警告] OCR 執行失敗，將只用物件偵測結果比對：{exc}", file=sys.stderr)

    return det_list, ocr_list


def _auto_detect_vlm(
    photo_path: Path, img_w: int, img_h: int, goal: str = "",
) -> Tuple[List[dict], List[dict]]:
    """模式 B/C：呼叫系統既有的「兩階段 VLM」模組（`server/vlm.py` 的
    `perceive_and_decide()`）跑一次——這是系統在 GroundingDINO 不可用時
    的後備路徑，也是目前團隊實際建圖標準採用的模式（見對話紀錄）。

    跟 GroundingDINO 模式最大的差別：VLM 沒辦法可靠輸出精確的像素座標，
    物件的「畫面位置」是模型自己描述的粗略九宮格＋近遠文字（例如
    "left middle, near"），對應的 `box` 只是系統依這段文字換算出來的
    固定模板方框，不是逐一量測的結果——這是刻意的設計取捨，不是資料
    有問題。`locate_v2.py` 的比對邏輯本來就只讀文字描述的 `position`
    欄位，不直接用 `box` 座標，所以這個限制不影響比對演算法本身。

    `perceive_and_decide()` 其實會回傳兩階段結果：第一階段是「看到了
    什麼」（我們要的），第二階段是「該怎麼走」（給實際導航用的建議，
    這裡用不到，直接捨棄）。第二階段一樣會被呼叫（函式沒有拆開兩段的
    介面），只是回傳值不使用，不影響結果正確性，只是稍微多花一點時間
    / API 額度。

    這支工具沒有「現有導航進度」的概念，`topomap_summary` 固定傳空字串
    （只影響第二階段的導航建議內容，我們不使用那段結果，不影響第一階段
    的物件描述）；沒有「上一輪追問」的概念，`prior_question`/
    `prior_answer` 固定傳 `None`。

    需要環境設定好 `VLM_BACKEND` 對應的 API 金鑰（預設 OpenAI，需要
    `OPENAI_API_KEY`），沒設定好會直接失敗，例外往外拋，由呼叫端決定
    後續處理。
    """
    from server.vlm import perceive_and_decide  # type: ignore

    goal_objects = [goal] if goal else []
    perception, _decision = perceive_and_decide(
        image_path=str(photo_path),
        goal=goal,
        goal_objects=goal_objects,
        topomap_summary="",
        img_w=img_w,
        img_h=img_h,
        prior_question=None,
        prior_answer=None,
    )

    det_list = [
        {"label": d.label, "box": list(d.bbox), "score": d.score, "position": d.position}
        for d in perception.detections
    ]
    ocr_list = [
        {"text": t.text, "confidence": t.score, "bbox": list(t.bbox) if t.bbox else None}
        for t in perception.ocr_texts
    ]
    _associate_ocr_with_detections(det_list, ocr_list)
    return det_list, ocr_list


def _auto_detect(
    photo_path: Path, img_w: int, img_h: int, *, mode: str = "vlm", goal: str = "",
) -> Tuple[List[dict], List[dict], str]:
    """依 `mode` 決定要走哪條偵測路徑，回傳 (detections, ocr_items, 實際用的模式)。

    Args:
        mode:
            "vlm" —— 直接用兩階段 VLM（預設；系統實際建圖目前統一採用
                這個模式，見模組開頭「偵測模式」說明）。
            "groundingdino" —— 強制只用 GroundingDINO，失敗就直接報錯，
                不會偷偷退回 VLM（適合想確認 GroundingDINO 本身能不能
                跑的情境）。
            "auto" —— 模擬 `server.py` 的 `has_grounding` 判斷：優先
                嘗試 GroundingDINO，失敗（沒裝 torch/模型權重）才退回
                VLM，並印出警告說明已切換模式。
        goal: 可選的導航目標文字，只有 VLM 模式會用到（併入 prompt）。
    """
    if mode == "vlm":
        det, ocr = _auto_detect_vlm(photo_path, img_w, img_h, goal)
        return det, ocr, "VLM 兩階段"

    if mode == "groundingdino":
        det, ocr = _auto_detect_groundingdino(photo_path)
        return det, ocr, "GroundingDINO"

    if mode == "auto":
        try:
            det, ocr = _auto_detect_groundingdino(photo_path)
            return det, ocr, "GroundingDINO"
        except Exception as exc:  # noqa: BLE001
            print(
                f"[提示] GroundingDINO 無法使用（{exc}），改用兩階段 VLM 模式...",
                file=sys.stderr,
            )
            det, ocr = _auto_detect_vlm(photo_path, img_w, img_h, goal)
            return det, ocr, "VLM 兩階段（自動退回）"

    raise ValueError(f"未知的 --detect-mode：{mode!r}（應為 vlm / groundingdino / auto）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True, help="場所地圖 topomap.json 路徑")
    parser.add_argument("--photo", default=None,
                         help="查詢照片路徑（用來讀取寬高；沒給 --detections 時也用來自動偵測）")
    parser.add_argument("--detections", default=None,
                         help="這張照片預先算好的偵測結果 json（格式同 detections_*.json）")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.05,
                         help="單一物件配對門檻，對應 locate_v2.py 的 min_pair_score"
                              "（新的評分尺度下預設 0.05，只濾掉雜訊配對；不是舊版的 1.6 那個尺度）")
    parser.add_argument("--allow-structural", action="store_true",
                         help="也允許用門/牆/窗等環境結構物做比對（預設關閉，這類物件到處都是，容易誤判）")
    parser.add_argument("--json-out", default=None, help="把完整結果另存一份 json（預設只印文字報告）")
    parser.add_argument("--debug-png", default=None,
                         help="把查詢照片畫成地圖上的節點，另存一張除錯用的 PNG"
                              "（跟平常拓樸地圖 PNG 同樣的畫法，查詢節點用紅星標示）")
    parser.add_argument("--detect-mode", default="vlm", choices=["vlm", "groundingdino", "auto"],
                         help="--photo 自動偵測要用哪條路徑（預設 vlm；"
                              "見模組開頭「偵測模式」說明）")
    parser.add_argument("--goal", default="",
                         help="可選：VLM 模式下的導航目標文字（例如「尋找電腦」），"
                              "只有 --detect-mode vlm/auto 退回 VLM 時會用到")
    args = parser.parse_args()

    map_path = Path(args.map).resolve()
    if not map_path.exists():
        raise SystemExit(f"找不到地圖檔案：{map_path}")
    topo = TopoGraphV2.load(map_path)

    photo_path = Path(args.photo).resolve() if args.photo else None
    img_w, img_h = _guess_image_size(photo_path)

    if args.detections:
        det_path = Path(args.detections).resolve()
        if not det_path.exists():
            raise SystemExit(f"找不到偵測結果檔案：{det_path}")
        detections, ocr_items = _load_detections_file(det_path)
    else:
        if not photo_path:
            raise SystemExit("--detections 和 --photo 至少要給一個"
                              "（沒給 --detections 時需要 --photo 才能自動偵測）")
        try:
            detections, ocr_items, mode_used = _auto_detect(
                photo_path, img_w, img_h, mode=args.detect_mode, goal=args.goal,
            )
            print(f"偵測模式：{mode_used}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — 環境沒裝模型時給清楚引導，而不是丟一堆 traceback
            raise SystemExit(
                f"自動偵測失敗（--detect-mode={args.detect_mode}）：\n"
                f"  {exc}\n"
                "GroundingDINO 模式需要這個環境裝好 torch/groundingdino 與模型權重；"
                "VLM 模式需要對應的 API 金鑰（見 server/config.py 的 VLM_BACKEND 設定）。\n"
                "也可以改用 --detections 指定一份已經算好的偵測結果 json"
                "（格式同 output/sessions/<id>/annotated/detections_*.json）。"
            )

    if args.debug_png:
        png_bytes, result = render_localization_debug_png(
            topo, detections, img_w, img_h, ocr_items=ocr_items,
            top_k=args.top_k, min_pair_score=args.min_score, allow_structural=args.allow_structural,
        )
        out_png = Path(args.debug_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(png_bytes)
    else:
        result = localize_photo(
            topo, detections, img_w, img_h, ocr_items=ocr_items,
            top_k=args.top_k, min_pair_score=args.min_score, allow_structural=args.allow_structural,
        )

    print(format_localization_report(result, topo))

    if args.debug_png:
        print(f"\n除錯用地圖圖片已存到：{Path(args.debug_png)}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(
            json.dumps({
                "query_object_count": result.query_object_count,
                "warning": result.warning,
                "candidates": [asdict(c) for c in result.candidates],
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n完整結果已存到：{out_path}")


if __name__ == "__main__":
    main()
