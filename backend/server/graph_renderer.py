"""Render goal graph and cumulative scene graph as PNG images."""
from __future__ import annotations

import difflib
import io
from typing import List, Optional


def _is_goal(label: str, goal_objects: List[str]) -> bool:
    label_l = label.lower()
    for g in goal_objects:
        if g.lower() in label_l or label_l in g.lower():
            return True
    return False


def _is_similar(label: str, goal_objects: List[str]) -> bool:
    label_l = label.lower()
    for g in goal_objects:
        if difflib.SequenceMatcher(None, label_l, g.lower()).ratio() > 0.5:
            return True
    return False


def render_goal_graph(
    goal: str,
    goal_objects: List[str],
    dst_path: str,
) -> None:
    """Render a goal decomposition graph showing goal → target objects."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title(f"Goal Graph\n{goal}", fontsize=14, fontweight="bold")

    # Goal node (center top)
    goal_box = FancyBboxPatch((3.5, 4.5), 3, 1, boxstyle="round,pad=0.2",
                               facecolor="#4A90D9", edgecolor="#2C5F8A", linewidth=2)
    ax.add_patch(goal_box)
    ax.text(5, 5, goal, ha="center", va="center", fontsize=12,
            color="white", fontweight="bold")

    # Target object nodes (bottom row)
    n = len(goal_objects)
    if n == 0:
        fig.savefig(dst_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    cols = min(n, 5)
    spacing = 9.0 / cols
    start_x = 0.5 + (9.0 - cols * spacing) / 2

    for i, obj in enumerate(goal_objects[:10]):
        row = i // 5
        col = i % 5
        x = start_x + col * spacing
        y = 2.0 - row * 1.8

        box = FancyBboxPatch((x, y), spacing * 0.85, 0.8,
                              boxstyle="round,pad=0.15",
                              facecolor="#E74C3C", edgecolor="#C0392B", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + spacing * 0.425, y + 0.4, obj, ha="center", va="center",
                fontsize=9, color="white")

        ax.annotate("", xy=(x + spacing * 0.425, y + 0.8), xytext=(5, 4.5),
                     arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.5))

    ax.set_ylim(-0.5 if n > 5 else 1.0, 6.5)
    fig.savefig(dst_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_scene_graph(
    observations: List[dict],
    goal: str,
    goal_objects: List[str],
    dst_path: str,
) -> None:
    """Render a cumulative scene graph across all navigation steps.

    observations: list of dicts, each with:
        - step (int)
        - photo (str): filename
        - detections: list of {label, score}
        - vlm_summary (str)
        - ocr_texts: list of str
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    n_obs = len(observations)
    fig_height = max(6, n_obs * 3.5)
    fig, ax = plt.subplots(1, 1, figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title(f"Scene Graph — {goal}", fontsize=14, fontweight="bold")

    y_top = fig_height - 1
    node_h = 1.2
    det_h = 0.7
    y_spacing = 3.0

    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, fig_height)

    for i, obs in enumerate(observations):
        y = y_top - i * y_spacing
        step = obs.get("step", i)
        photo = obs.get("photo", "")
        summary = obs.get("vlm_summary", "")
        dets = obs.get("detections", [])
        ocr_texts = obs.get("ocr_texts", [])

        # Observation node
        box = FancyBboxPatch((0.5, y - node_h / 2), 4, node_h,
                              boxstyle="round,pad=0.2",
                              facecolor="#3498DB", edgecolor="#2C3E50", linewidth=2)
        ax.add_patch(box)

        label_lines = [f"Step {step}: {photo}"]
        if summary:
            label_lines.append(summary[:40])
        ax.text(2.5, y, "\n".join(label_lines), ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

        # Arrow to next observation
        if i < n_obs - 1:
            next_y = y_top - (i + 1) * y_spacing
            ax.annotate("", xy=(2.5, next_y + node_h / 2 + 0.1),
                         xytext=(2.5, y - node_h / 2 - 0.1),
                         arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=2))

        # Detection sub-nodes
        det_x = 6.0
        for j, det in enumerate(dets[:6]):
            det_label = det.get("label", "")
            det_score = det.get("score", 0)
            det_y = y + (len(dets[:6]) / 2 - j) * (det_h + 0.15)

            if _is_goal(det_label, goal_objects):
                fc, ec = "#E74C3C", "#C0392B"
            elif _is_similar(det_label, goal_objects):
                fc, ec = "#F39C12", "#D68910"
            else:
                fc, ec = "#27AE60", "#1E8449"

            det_box = FancyBboxPatch((det_x, det_y - det_h / 2), 3.5, det_h,
                                      boxstyle="round,pad=0.1",
                                      facecolor=fc, edgecolor=ec, linewidth=1.5)
            ax.add_patch(det_box)
            ax.text(det_x + 1.75, det_y, f"{det_label}\n{det_score:.0%}",
                    ha="center", va="center", fontsize=8, color="white")

            ax.annotate("", xy=(det_x, det_y), xytext=(4.5, y),
                         arrowprops=dict(arrowstyle="->", color="#BDC3C7", lw=1))

        # OCR texts
        if ocr_texts:
            ocr_x = 10.5
            ocr_str = ", ".join(ocr_texts[:3])
            if len(ocr_str) > 30:
                ocr_str = ocr_str[:30] + "..."
            ocr_box = FancyBboxPatch((ocr_x, y - det_h / 2), 2.5, det_h,
                                      boxstyle="round,pad=0.1",
                                      facecolor="#17A2B8", edgecolor="#117A8B", linewidth=1)
            ax.add_patch(ocr_box)
            ax.text(ocr_x + 1.25, y, f"OCR:\n{ocr_str}", ha="center", va="center",
                    fontsize=7, color="white")

    # Legend
    legend_items = [
        ("#E74C3C", "Target"),
        ("#F39C12", "Similar"),
        ("#27AE60", "Generic"),
        ("#17A2B8", "OCR"),
        ("#3498DB", "Observation"),
    ]
    for k, (c, t) in enumerate(legend_items):
        lx, ly = 11, 1.5 - k * 0.5
        ax.add_patch(FancyBboxPatch((lx, ly - 0.15), 0.3, 0.3,
                                     boxstyle="round,pad=0.05", facecolor=c, edgecolor="none"))
        ax.text(lx + 0.5, ly, t, fontsize=8, va="center")

    fig.savefig(dst_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
