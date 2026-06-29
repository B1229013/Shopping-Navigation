"""Render goal graph and cumulative scene graph as PNG images.

Includes detection post-processing: NMS, merge, entity tracking, spatial
context, and OCR-door association — aligned with classmate's project.
"""
from __future__ import annotations

import contextlib
import io as _io
import re
from collections import Counter, defaultdict
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


# ── Font helper (Windows CJK) ──────────────────────────────────────────

def _cjk():
    from matplotlib.font_manager import FontProperties, fontManager
    available = {f.name for f in fontManager.ttflist}
    for name in ["Microsoft JhengHei", "Microsoft YaHei",
                 "Heiti TC", "LiHei Pro", "Arial Unicode MS"]:
        if name in available:
            return FontProperties(family=name)
    return FontProperties()


# ── Known goal → detection classes mapping ─────────────────────────────

GOAL_CLASS_MAP: dict[str, dict] = {
    "電腦": {
        "classes":   ["電腦", "桌上型電腦", "筆記型電腦", "螢幕", "鍵盤", "滑鼠",
                      "computer", "laptop", "desktop computer", "monitor",
                      "keyboard", "mouse", "screen"],
        "ocr_parts": ["電腦", "桌機", "筆電", "本機", "computer", "PC", "desktop", "laptop"],
    },
    "冰箱": {
        "classes":   ["冰箱", "refrigerator", "fridge", "冷藏"],
        "ocr_parts": ["冰箱", "refrigerator", "fridge"],
    },
    "飲水機": {
        "classes":   ["飲水機", "water dispenser", "water cooler", "water fountain"],
        "ocr_parts": ["飲水機", "water dispenser"],
    },
    "印表機": {
        "classes":   ["印表機", "printer", "laser printer"],
        "ocr_parts": ["印表機", "printer"],
    },
    "辦公室": {
        "classes":   ["辦公室", "教師辦公室", "office", "nameplate", "name plate",
                      "door sign", "plaque", "office door"],
        "ocr_parts": [],
    },
}


def augment_goal_classes(goal: str, goal_objects: list[str]) -> list[str]:
    """Expand goal_objects with known mappings from GOAL_CLASS_MAP."""
    extra: list[str] = []
    for key, info in GOAL_CLASS_MAP.items():
        if key in goal:
            extra.extend(info["classes"])
    merged = list(dict.fromkeys(goal_objects + extra))
    return merged


def get_goal_ocr_parts(goal: str) -> list[str]:
    """Return OCR confirmation keywords for the goal."""
    parts: list[str] = [goal]
    for key, info in GOAL_CLASS_MAP.items():
        if key in goal:
            parts.extend(info["ocr_parts"])
    return list(dict.fromkeys(parts))


# ── Object ID assignment ────────────────────────────────────────────────

def assign_detection_ids(detections: list) -> None:
    """Add unique 'id' field to each detection dict in-place."""
    label_count = Counter(d["label"] for d in detections)
    label_idx: dict[str, int] = {}

    for d in detections:
        lbl = d["label"]
        nameplate = d.get("nameplate_text", "")
        if nameplate:
            nid = _nameplate_id(nameplate)
            if nid:
                d["id"] = f"{lbl}:{nid}"
                continue
        if label_count[lbl] == 1:
            d["id"] = lbl
        else:
            label_idx[lbl] = label_idx.get(lbl, 0) + 1
            d["id"] = f"{lbl}_{label_idx[lbl]}"


def _nameplate_id(nameplate_text: str) -> str:
    segments = re.findall(r'[一-鿿]{2,}', nameplate_text)
    if segments:
        return "".join(segments[:2])[:8]
    cleaned = re.sub(r'[A-Za-z0-9\s\.%"\'_,\-]+', '', nameplate_text).strip()
    return cleaned[:8]


# ── NMS (deduplicate overlapping boxes) ─────────────────────────────────

def _iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua else 0.0


def apply_nms(detections: list[dict], iou_thr: float = 0.50) -> list[dict]:
    """Remove lower-confidence boxes that overlap significantly."""
    dets = sorted(detections, key=lambda d: -d["score"])
    keep, suppressed = [], set()
    for i, d in enumerate(dets):
        if i in suppressed:
            continue
        keep.append(d)
        for j in range(i + 1, len(dets)):
            if j in suppressed:
                continue
            if _iou(d["box"], dets[j]["box"]) > iou_thr:
                suppressed.add(j)
    return keep


# ── Merge adjacent same-label boxes ────────────────────────────────────

def _boxes_nearby(a: list, b: list, gap_ratio: float = 0.6) -> bool:
    h_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    v_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    min_side = min(a[2]-a[0], a[3]-a[1], b[2]-b[0], b[3]-b[1])
    thr = gap_ratio * min_side
    return h_gap <= thr and v_gap <= thr


def merge_adjacent_same_label(detections: list[dict]) -> list[dict]:
    """Merge detections of the same label whose boxes are adjacent."""
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(detections):
        by_label[d["label"]].append(i)

    result = []
    for label, idxs in by_label.items():
        clusters: list[list[int]] = []
        for idx in idxs:
            merged_into = None
            for cluster in clusters:
                if any(_boxes_nearby(detections[idx]["box"], detections[c]["box"]) for c in cluster):
                    cluster.append(idx)
                    merged_into = cluster
                    break
            if merged_into is None:
                clusters.append([idx])

        for cluster in clusters:
            if len(cluster) == 1:
                result.append(detections[cluster[0]])
            else:
                boxes = [detections[i]["box"] for i in cluster]
                merged_box = [
                    min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes),
                ]
                best = max(cluster, key=lambda i: detections[i]["score"])
                merged_det = dict(detections[best])
                merged_det["box"] = [round(x, 1) for x in merged_box]
                result.append(merged_det)

    result.sort(key=lambda d: -d["score"])
    return result


# ── Cross-step entity tracking ─────────────────────────────────────────

def _norm_center(box: list, img_w: int, img_h: int) -> tuple[float, float]:
    return ((box[0] + box[2]) / (2 * img_w),
            (box[1] + box[3]) / (2 * img_h))


def _norm_iou(a: list, b: list, wa: int, ha: int, wb: int, hb: int) -> float:
    a_n = [a[0]/wa, a[1]/ha, a[2]/wa, a[3]/ha]
    b_n = [b[0]/wb, b[1]/hb, b[2]/wb, b[3]/hb]
    ix1, iy1 = max(a_n[0], b_n[0]), max(a_n[1], b_n[1])
    ix2, iy2 = min(a_n[2], b_n[2]), min(a_n[3], b_n[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a_n[2]-a_n[0])*(a_n[3]-a_n[1]) + (b_n[2]-b_n[0])*(b_n[3]-b_n[1]) - inter
    return inter / ua if ua else 0.0


def _label_matches(a: str, b: str) -> bool:
    if a == b:
        return True
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    return bool(a_words & b_words)


def tag_same_entity(detections: list[dict], img_w: int, img_h: int,
                    prev_dets: list[dict], prev_w: int, prev_h: int,
                    pos_thresh: float = 0.25) -> None:
    """Mark same_entity=True if detection matches one in previous step."""
    for det in detections:
        nc = _norm_center(det["box"], img_w, img_h)
        for pd in prev_dets:
            if not _label_matches(det["label"], pd["label"]):
                continue
            pnc = _norm_center(pd["box"], prev_w, prev_h)
            center_ok = (abs(nc[0] - pnc[0]) < pos_thresh
                         and abs(nc[1] - pnc[1]) < pos_thresh)
            iou_ok = _norm_iou(det["box"], pd["box"], img_w, img_h, prev_w, prev_h) > 0.05
            if center_ok or iou_ok:
                det["same_entity"] = True
                break


# ── Spatial context inference ──────────────────────────────────────────

_SURFACE_KEYWORDS = {"table", "desk", "cabinet", "counter", "shelf",
                     "sofa", "chair", "glass table", "windowsill"}


def infer_spatial_context(detections: list[dict]) -> None:
    """Add 'context' field: e.g. '在cabinet上'."""
    for det in detections:
        det.setdefault("context", "")
        lbl_lower = det["label"].lower()
        if any(s in lbl_lower for s in _SURFACE_KEYWORDS):
            continue
        x1a, y1a, x2a, y2a = det["box"]
        cx_a = (x1a + x2a) / 2
        best_surface, best_overlap = None, 0.0
        for other in detections:
            if other is det:
                continue
            if not any(s in other["label"].lower() for s in _SURFACE_KEYWORDS):
                continue
            x1b, y1b, x2b, y2b = other["box"]
            if not (x1b < cx_a < x2b):
                continue
            if not (y1b <= y2a <= y2b + (y2b - y1b) * 0.35):
                continue
            overlap = min(x2a, x2b) - max(x1a, x1b)
            if overlap > best_overlap:
                best_overlap = overlap
                best_surface = other["label"]
        if best_surface:
            det["context"] = f"在{best_surface}上"


# ── OCR → door association ─────────────────────────────────────────────

_DOOR_LABELS = {"door", "office door"}


def associate_ocr_to_doors(detections: list[dict],
                           ocr_with_bbox: list[dict]) -> None:
    """Link OCR nameplate text to nearby door detections.

    ocr_with_bbox: list of {"text": str, "confidence": float,
                            "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}
    """
    for det in detections:
        det.setdefault("nameplate_text", "")
        det.setdefault("nameplate_conf", 0.0)
        if not any(kw in det["label"].lower() for kw in _DOOR_LABELS):
            continue
        dx1, dy1, dx2, dy2 = det["box"]
        door_h = dy2 - dy1
        door_w = dx2 - dx1

        texts, best_conf = [], 0.0
        for ocr in ocr_with_bbox:
            bbox_pts = ocr["bbox"]
            ox1 = min(p[0] for p in bbox_pts)
            oy1 = min(p[1] for p in bbox_pts)
            ox2 = max(p[0] for p in bbox_pts)
            oy2 = max(p[1] for p in bbox_pts)
            ocr_w = ox2 - ox1
            ocr_h = oy2 - oy1
            ocr_cy = (oy1 + oy2) / 2
            conf = ocr["confidence"]

            overlap_x = max(0.0, min(ox2, dx2) - max(ox1, dx1))
            overlap_y = max(0.0, min(oy2, dy2) - max(oy1, dy1))
            overlap_ratio = (overlap_x * overlap_y) / max(ocr_w * ocr_h, 1)
            if overlap_ratio > 0.50:
                continue
            if ocr_w > door_w * 0.50 or ocr_h > door_h * 0.30:
                continue

            v_ok = dy1 <= ocr_cy <= dy1 + door_h * 0.75
            h_gap = max(0.0, max(ox1 - dx2, dx1 - ox2))
            if h_gap < door_w * 1.5 and v_ok and conf >= 0.35:
                texts.append(ocr["text"])
                best_conf = max(best_conf, conf)

        if texts:
            det["nameplate_text"] = " ".join(texts)
            det["nameplate_conf"] = best_conf


# ── Full post-processing pipeline ──────────────────────────────────────

def postprocess_detections(
    det_dicts: list[dict],
    img_w: int,
    img_h: int,
    ocr_with_bbox: Optional[list[dict]] = None,
    prev_dets: Optional[list[dict]] = None,
    prev_w: int = 0,
    prev_h: int = 0,
) -> list[dict]:
    """Run NMS → merge → OCR-door → spatial context → entity tracking → ID assignment."""
    if not det_dicts:
        return det_dicts

    det_dicts = apply_nms(det_dicts)
    det_dicts = merge_adjacent_same_label(det_dicts)

    if ocr_with_bbox:
        associate_ocr_to_doors(det_dicts, ocr_with_bbox)

    infer_spatial_context(det_dicts)

    if prev_dets and prev_w > 0 and prev_h > 0:
        tag_same_entity(det_dicts, img_w, img_h, prev_dets, prev_w, prev_h)

    assign_detection_ids(det_dicts)
    return det_dicts


# ── Goal classification helpers ─────────────────────────────────────────

def _is_goal(label: str, goal_objects: List[str]) -> bool:
    label_l = label.lower()
    for g in goal_objects:
        if g.lower() in label_l or label_l in g.lower():
            return True
    return False


# ── Detection node color (6-state) ────────────────────────────────────

def _det_color(det: dict, goal_objects: List[str]):
    """Return (facecolor, edgecolor, textcolor) for a detection node."""
    status = det.get("status", "")
    if status == "arrived":
        return "#d7ccc8", "#6d4c41", "#3e2723"    # brown
    if status == "false_positive":
        return "#eeeeee", "#757575", "#424242"     # gray
    if status == "wrong_instance":
        return "#e8d5f5", "#7b1fa2", "#4a0072"     # purple
    if det.get("same_entity"):
        return "#cce5ff", "#004085", "#003060"     # blue
    if _is_goal(det.get("label", ""), goal_objects):
        return "#f8d7da", "#dc3545", "#721c24"     # red
    return "#d4edda", "#28a745", "#155724"          # green


# ── Goal Graph (3-node: place → goal → target) ─────────────────────────

def render_goal_graph(
    goal: str,
    goal_objects: List[str],
    dst_path: str,
    place: str = "室內",
) -> None:
    fp = _cjk()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5)
    ax.axis("off")

    goal_text = f"尋找{goal}"
    ax.set_title(f"目標圖 Goal Graph\n目標文字：{goal_text}",
                 fontsize=13, fontweight="bold", color="#222",
                 fontproperties=fp, pad=10)

    def node(cx, cy, lines, type_lbl, fc, tc="#333", ec="#aaa"):
        box = FancyBboxPatch((cx - 1.1, cy - 0.48), 2.2, 0.96,
                             boxstyle="round,pad=0.1",
                             facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        for i, ln in enumerate(lines):
            ax.text(cx, cy + 0.12 - i * 0.26, ln,
                    ha="center", va="center", fontsize=10,
                    color=tc, fontproperties=fp, zorder=3)
        ax.text(cx, cy - 0.35, f"(type: {type_lbl})",
                ha="center", fontsize=7.5, color="#888", zorder=3)

    def arrow(x1, y1, x2, y2, lbl=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
        if lbl:
            ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, lbl,
                    fontsize=8, color="#666", ha="left", va="center")

    lns = [goal_text[:10], goal_text[10:]] if len(goal_text) > 10 else [goal_text]
    node(4.5, 2.5, lns, "goal", "#d0e4f7", "#1a4a7a", "#5b9bd5")
    node(4.5, 4.2, [place], "place", "#d4edda", "#155724", "#28a745")
    target_label = goal if len(goal) <= 10 else goal[:9] + "…"
    node(7.5, 1.2, [target_label], "target", "#f8d7da", "#721c24", "#dc3545")

    arrow(4.5, 3.0, 4.5, 3.75, "located_in")
    arrow(5.3, 2.1, 6.4, 1.5, "target_of")

    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        plt.tight_layout()
        plt.savefig(dst_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ── Scene Graph (cumulative, multi-step) ────────────────────────────────

def _row_h_for(n_dets: int, is_current: bool = False) -> float:
    if is_current:
        NODE_H, GAP, base = 0.58, 0.24, 3.2
    else:
        NODE_H, GAP, base = 0.44, 0.20, 2.2
    return max(base, n_dets * (NODE_H + GAP) + 0.7)


def render_scene_graph(
    observations: List[dict],
    goal: str,
    goal_objects: List[str],
    dst_path: str,
) -> None:
    fp = _cjk()
    n_obs = len(observations)
    if n_obs == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "尚無觀察資料", ha="center", va="center",
                fontsize=14, fontproperties=fp)
        ax.axis("off")
        fig.savefig(dst_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return

    current_step = observations[-1].get("step", n_obs)

    for obs in observations:
        for d in obs.get("detections", []):
            if "id" not in d:
                d["id"] = d["label"]

    row_heights = [
        _row_h_for(len(obs.get("detections", [])),
                    obs.get("step", 0) == current_step)
        for obs in observations
    ]
    total_h = sum(row_heights)
    fig_h = max(6, total_h + 2.5)

    fig, ax = plt.subplots(figsize=(15, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    ax.set_title(
        f"場景圖 Scene Graph — 累積至步驟 {current_step}\n"
        f"(紅=目標候選  綠=地標  藍=同實體  棕=已到達  紫=同類非目標  灰=誤判)",
        fontsize=11, fontweight="bold", color="#222",
        fontproperties=fp, pad=8)

    obs_x = 1.8

    obs_y: dict[int, float] = {}
    y_cursor = fig_h - 1.3
    for i, obs in enumerate(observations):
        rh_i = row_heights[i]
        s = obs.get("step", i + 1)
        obs_y[s] = y_cursor - rh_i / 2
        y_cursor -= rh_i

    node_data: list[dict] = []

    for i, obs in enumerate(observations):
        s = obs.get("step", i + 1)
        y = obs_y[s]
        rh = row_heights[i]
        is_current = (s == current_step)
        photo = obs.get("photo", "")
        vlm_summary = obs.get("vlm_summary", "")
        dets = obs.get("detections", [])
        ocr_texts = obs.get("ocr_texts", [])
        is_false_pos = obs.get("false_positive", False)

        # Observation node color: yellow if false_positive
        fc = "#ffeeba" if is_false_pos else ("#d0e4f7" if is_current else "#e8f0f8")
        ec = "#ffc107" if is_false_pos else ("#5b9bd5" if is_current else "#a0b8cc")
        lw = 2.0 if is_current else (2.0 if is_false_pos else 1.2)

        box = FancyBboxPatch((obs_x - 1.5, y - 0.77), 3.0, 1.54,
                             boxstyle="round,pad=0.08",
                             facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(box)

        ax.text(obs_x, y + 0.30, f"觀察點 {s}", ha="center", va="center",
                fontsize=12, fontproperties=fp,
                color="#1a4a7a" if is_current else "#4a6a8a",
                fontweight="bold" if is_current else "normal", zorder=4)
        ax.text(obs_x, y + 0.10, photo,
                ha="center", va="center", fontsize=8.5, color="#777", zorder=4)

        ocr_hit = any(
            g.lower() in t.lower()
            for t in ocr_texts
            for g in goal_objects[:3]
        )
        if ocr_hit:
            ax.text(obs_x, y - 0.12, "OCR：相關文字",
                    ha="center", va="center", fontsize=8.5,
                    color="#155724", fontproperties=fp, zorder=4)

        if vlm_summary:
            clean = vlm_summary.replace("\n", " ")
            first_sent = clean.split("。")[0]
            L = 24
            line1 = first_sent[:L]
            ax.text(obs_x, y - 0.40, f"VLM: {line1}",
                    ha="center", va="center", fontsize=7,
                    color="#444", fontproperties=fp, zorder=4)
            if len(first_sent) > L:
                line2 = first_sent[L:L * 2]
                if len(first_sent) > L * 2:
                    line2 += "…"
                ax.text(obs_x, y - 0.58, f"   {line2}",
                        ha="center", va="center", fontsize=7,
                        color="#444", fontproperties=fp, zorder=4)

        if i < n_obs - 1:
            ns = observations[i + 1].get("step", i + 2)
            ny = obs_y[ns]
            ax.annotate("", xy=(obs_x, ny + 0.77),
                        xytext=(obs_x, y - 0.77),
                        arrowprops=dict(arrowstyle="->", color="#7a9fc0", lw=1.3))

        if not dets:
            continue

        ox = 12.5 if is_current else 5.0
        n = len(dets)
        spacing = rh / (n + 1)
        fs = 11.0 if is_current else 8.0
        MIN_H = 0.52 if is_current else 0.38
        node_h = max(MIN_H, min(MIN_H * 1.2, spacing - 0.14))
        char_w_est = 0.140 if is_current else 0.098

        for j, det in enumerate(dets):
            lbl = det.get("label", "")
            uid = det.get("id", lbl)
            score = det.get("score", 0)
            context = det.get("context", "")
            nameplate = det.get("nameplate_text", "")
            dy = y + rh / 2 - spacing * (j + 1)

            # Display label: nameplate format for doors, spatial context, or plain
            if nameplate and any(kw in lbl.lower() for kw in _DOOR_LABELS):
                np_short = nameplate[:10]
                display_lbl = f"{uid}:{np_short}"
            elif context:
                ctx_s = context if len(context) <= 9 else context[:8] + "…"
                display_lbl = f"{uid} ({ctx_s})"
            else:
                display_lbl = uid
            _mc = 26 if is_current else 17
            if len(display_lbl) > _mc:
                display_lbl = display_lbl[:_mc - 1] + "…"

            node_w = min(
                6.0 if is_current else 3.4,
                max(4.2 if is_current else 2.3,
                    len(display_lbl) * char_w_est + 0.55)
            )

            fc2, ec2, tc = _det_color(det, goal_objects)

            nb = FancyBboxPatch((ox - node_w / 2, dy - node_h / 2), node_w, node_h,
                                boxstyle="round,pad=0.05",
                                facecolor=fc2, edgecolor=ec2,
                                linewidth=1.4 if is_current else 0.8,
                                alpha=1.0 if is_current else 0.6, zorder=2)
            ax.add_patch(nb)

            label_y = dy + node_h * 0.18
            conf_y = dy - node_h * 0.22
            ax.text(ox, label_y, display_lbl,
                    ha="center", va="center", fontsize=fs,
                    color=tc, fontproperties=fp, zorder=3)
            ax.text(ox, conf_y, f"conf: {score:.3f}",
                    ha="center", va="center", fontsize=fs - 1.5,
                    color="#999", zorder=3)

            edge_color = "#5b9bd5" if is_current else "#ccc"
            edge_lw = 0.9 if is_current else 0.4
            ax.annotate("", xy=(ox - node_w / 2, dy),
                        xytext=(obs_x + 1.5, y),
                        arrowprops=dict(arrowstyle="->", color=edge_color, lw=edge_lw))

            node_data.append({"label": lbl, "step": s, "x": ox, "y": dy,
                              "node_w": node_w, "context": context})

    # ── Cross-step same-entity arcs (per-label color + fan-out) ──
    _ARC_COLOURS = ["#004085", "#7b1fa2", "#1a7a40", "#b34700", "#005f73", "#8b0000"]
    _label_idx: dict[str, int] = {}

    for i, obs in enumerate(observations[1:], 1):
        prev_step_n = observations[i - 1].get("step", i)
        curr_step_n = obs.get("step", i + 1)
        for det in obs.get("detections", []):
            if not det.get("same_entity"):
                continue
            prev_n = next((n for n in node_data
                           if n["step"] == prev_step_n
                           and _label_matches(n["label"], det["label"])), None)
            curr_n = next((n for n in node_data
                           if n["step"] == curr_step_n
                           and _label_matches(n["label"], det["label"])), None)
            if not (prev_n and curr_n):
                continue

            lbl_key = det["label"]
            if lbl_key not in _label_idx:
                _label_idx[lbl_key] = len(_label_idx)
            idx = _label_idx[lbl_key]
            colour = _ARC_COLOURS[idx % len(_ARC_COLOURS)]

            sx = prev_n["x"] + prev_n["node_w"] / 2
            sy = prev_n["y"]
            ex = curr_n["x"] + curr_n["node_w"] / 2
            ey = curr_n["y"]

            same_col = abs(prev_n["x"] - curr_n["x"]) < 1.0
            if same_col:
                rad = -(0.30 + idx * 0.18)
            else:
                rad = -(0.08 + idx * 0.04)

            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1.2,
                                        linestyle="dashed", alpha=0.75,
                                        connectionstyle=f"arc3,rad={rad:.2f}"),
                        zorder=1)

    # ── Spatial context edges「在…上」──
    curr_obs_obj = next((o for o in observations
                         if o.get("step") == current_step), None)
    if curr_obs_obj:
        for det in curr_obs_obj.get("detections", []):
            ctx = det.get("context", "")
            if not ctx:
                continue
            surface_lbl = ctx.replace("在", "").replace("上", "").strip()
            curr_n = next((n for n in node_data
                           if n["step"] == current_step
                           and n["label"] == det["label"]), None)
            surf_n = next((n for n in node_data
                           if n["step"] == current_step
                           and surface_lbl in n["label"].lower()), None)
            if curr_n and surf_n:
                ax.annotate("",
                    xy=(surf_n["x"] + surf_n["node_w"] / 2, surf_n["y"]),
                    xytext=(curr_n["x"] + curr_n["node_w"] / 2, curr_n["y"]),
                    arrowprops=dict(arrowstyle="->", color="#1a7a40", lw=0.8,
                                    linestyle="dotted", alpha=0.65,
                                    connectionstyle="arc3,rad=0.4"),
                    zorder=1)
                mid_x = max(curr_n["x"], surf_n["x"]) + curr_n["node_w"] / 2 + 0.3
                mid_y = (curr_n["y"] + surf_n["y"]) / 2
                ax.text(mid_x, mid_y, "在…上", fontsize=7,
                        color="#1a7a40", ha="left", va="center",
                        fontproperties=fp, zorder=2)

    from matplotlib.lines import Line2D
    legend_items = [
        mpatches.Patch(facecolor="#d7ccc8", edgecolor="#6d4c41", label="正式找到（導航結束）"),
        mpatches.Patch(facecolor="#f8d7da", edgecolor="#dc3545", label="目標候選（待確認）"),
        mpatches.Patch(facecolor="#e8d5f5", edgecolor="#7b1fa2", label="同種非目標（用戶確認）"),
        mpatches.Patch(facecolor="#eeeeee", edgecolor="#757575", label="誤判（與目標完全無關）"),
        mpatches.Patch(facecolor="#cce5ff", edgecolor="#004085", label="同一實體（跨步追蹤）"),
        mpatches.Patch(facecolor="#d4edda", edgecolor="#28a745", label="地標物件"),
        Line2D([0], [0], color="#004085", lw=1.2, linestyle="dashed",
               label="- - 跨步同一實體追蹤"),
        Line2D([0], [0], color="#1a7a40", lw=1.0, linestyle="dotted",
               label="... 空間關係（在…上）"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=10,
              facecolor="white", edgecolor="#ccc", prop=fp)

    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        plt.tight_layout()
        plt.savefig(dst_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
