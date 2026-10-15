"""Renders a PNG plot of a standalone phone PDR sensor-test run (SensorTestView.swift).

Independent of the VLM navigation session/topomap — this just visualizes the raw
walked path plus the lap markers the user recorded, to inspect drift/error visually.
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt


def render_sensor_test_png(payload: dict) -> bytes:
    points = payload.get("points", [])
    laps = payload.get("laps", [])

    fig, ax = plt.subplots(figsize=(7, 7))

    if points:
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        ax.plot(xs, ys, "-", color="steelblue", linewidth=1.5, alpha=0.7, label="path")

    ax.scatter([0], [0], color="green", s=140, marker="*", zorder=5, label="origin")

    for lap in laps:
        ax.scatter([lap["x"]], [lap["y"]], color="red", s=70, zorder=5)
        ax.annotate(
            f"lap {lap['lap_number']}\n{lap['distance_from_origin']:.1f}m from origin",
            (lap["x"], lap["y"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color="red",
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Sensor test path")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()
