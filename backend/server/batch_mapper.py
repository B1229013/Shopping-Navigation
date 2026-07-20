"""Batch environment mapper: process a folder of photos using GroundingDINO,
build a topological map with object-photo relationships, and output results.

This uses the same GroundingDINO model that UniGoal uses for real-time
navigation — no LLMs, no manual data.

Usage:
    # From the project .venv:
    python -m server.batch_mapper --input <photo_folder> --goal "find refrigerator"
    python -m server.batch_mapper --input <photo_folder> --goal "find refrigerator" --output detections.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class PhotoDetection:
    """Detection results for a single photo."""
    filename: str
    filepath: str
    objects: List[Dict] = field(default_factory=list)  # [{label, score, box}]
    summary: str = ""


@dataclass
class ObjectInstance:
    """Tracks where a specific object type appears across photos."""
    label: str
    photos: List[str] = field(default_factory=list)
    total_count: int = 0
    best_score: float = 0.0
    best_photo: str = ""


def build_detect_classes(goal: str) -> List[str]:
    """Build detection classes from goal text.

    Uses the same approach as UniGoal's goal_decomposer but without LLM —
    just extracts meaningful words and adds related visual terms.
    """
    import re
    words = re.findall(r"[a-zA-Z]+", goal.lower())
    stop = {"find", "the", "a", "an", "to", "where", "is", "are", "all", "every",
            "i", "want", "need", "look", "for", "me", "my", "can", "you", "show"}
    classes = [w for w in words if w not in stop and len(w) > 1]

    # Add common indoor objects for context
    base_objects = [
        "door", "cabinet", "chair", "table", "desk", "sofa",
        "sign", "fire extinguisher", "plant", "printer",
    ]
    # Merge without duplicates
    all_classes = list(dict.fromkeys(classes + base_objects))
    return all_classes


def process_photos_groundingdino(
    photo_dir: str,
    detect_classes: List[str],
) -> Tuple[List[PhotoDetection], Dict[str, ObjectInstance]]:
    """Process all photos using GroundingDINO object detection.

    This is the same model UniGoal uses in its perception pipeline.
    """
    from server.perception import Perception

    photo_dir = Path(photo_dir)
    # Deduplicate: on Windows *.jpg and *.JPG match the same files
    seen_names = set()
    jpg_files = []
    for pattern in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
        for f in photo_dir.glob(pattern):
            if f.name.lower() not in seen_names:
                seen_names.add(f.name.lower())
                jpg_files.append(f)
    jpg_files.sort()
    if not jpg_files:
        log.error("No image files found in %s", photo_dir)
        return [], {}

    # Load GroundingDINO model (same as server startup)
    log.info("Loading GroundingDINO model...")
    perception = Perception()
    perception.load()
    log.info("Model loaded on %s", perception.device)

    all_detections: List[PhotoDetection] = []
    object_index: Dict[str, ObjectInstance] = {}

    for i, photo_path in enumerate(jpg_files):
        filename = photo_path.name
        log.info("[%d/%d] Detecting in %s ...", i + 1, len(jpg_files), filename)

        # Run GroundingDINO detection
        detections = perception.detect(str(photo_path), detect_classes)
        objects = [
            {"label": d.label, "score": round(d.score, 3), "box": [round(x, 1) for x in d.box]}
            for d in detections
        ]

        pd = PhotoDetection(
            filename=filename,
            filepath=str(photo_path),
            objects=objects,
            summary=", ".join(f"{o['label']}({o['score']})" for o in objects),
        )
        all_detections.append(pd)

        # Update object index
        for obj in objects:
            label = obj["label"]
            if label not in object_index:
                object_index[label] = ObjectInstance(label=label)
            oi = object_index[label]
            oi.photos.append(filename)
            oi.total_count += 1
            if obj["score"] > oi.best_score:
                oi.best_score = obj["score"]
                oi.best_photo = filename

        # Print detections for this photo
        if objects:
            labels = ", ".join(f"{o['label']}({o['score']})" for o in objects)
            log.info("  Found: %s", labels)
        else:
            log.info("  No detections")

    return all_detections, object_index


def compute_photo_connections(
    all_detections: List[PhotoDetection],
) -> List[Dict]:
    """Compute connections between photos based on shared detected objects.

    TASK 2 — labels are normalized to canonical store tokens before the Jaccard
    overlap so "milk bottle" and "milk carton" count as the same shared object
    instead of splitting one zone into two.
    """
    from server.config import normalize_label
    edges = []
    for i, det_a in enumerate(all_detections):
        labels_a = set(normalize_label(o["label"]) for o in det_a.objects)
        for j in range(i + 1, min(i + 4, len(all_detections))):
            det_b = all_detections[j]
            labels_b = set(normalize_label(o["label"]) for o in det_b.objects)
            shared = labels_a & labels_b
            if shared:
                sim = len(shared) / len(labels_a | labels_b)
                edges.append({
                    "from": det_a.filename,
                    "to": det_b.filename,
                    "similarity": round(sim, 3),
                    "shared_objects": sorted(shared),
                })
    return edges


def save_results(
    all_detections: List[PhotoDetection],
    object_index: Dict[str, ObjectInstance],
    edges: List[Dict],
    goal: str,
    output_path: str,
) -> None:
    """Save detection results to JSON."""
    data = {
        "goal": goal,
        "total_photos": len(all_detections),
        "total_objects_found": len(object_index),
        "photos": [
            {
                "filename": d.filename,
                "objects": d.objects,
                "summary": d.summary,
            }
            for d in all_detections
        ],
        "object_index": {
            label: {
                "photos": list(set(oi.photos)),
                "total_detections": oi.total_count,
                "best_score": oi.best_score,
                "best_photo": oi.best_photo,
            }
            for label, oi in sorted(object_index.items(), key=lambda x: -x[1].total_count)
        },
        "photo_connections": edges,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Results saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Batch environment mapper using GroundingDINO (UniGoal perception)"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Folder containing photos to process")
    parser.add_argument("--goal", "-g", required=True,
                        help="What to find, e.g. 'find refrigerator'")
    parser.add_argument("--output", "-o", default="detections.json",
                        help="Output JSON path (default: detections.json)")
    parser.add_argument("--extra-classes", nargs="*", default=None,
                        help="Additional object classes to detect")
    parser.add_argument("--map", "-m", action="store_true",
                        help="Auto-generate topological map after detection")
    parser.add_argument("--start", "-s", default=None,
                        help="Start photo filename for map navigation (default: first photo)")
    parser.add_argument("--map-output", default="topological_map",
                        help="Map output filename prefix (default: topological_map)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build detection classes from goal
    classes = build_detect_classes(args.goal)
    if args.extra_classes:
        classes = list(dict.fromkeys(classes + args.extra_classes))
    log.info("Goal: %s", args.goal)
    log.info("Detecting: %s", ", ".join(classes))

    # Run GroundingDINO detection on all photos
    all_detections, object_index = process_photos_groundingdino(args.input, classes)
    edges = compute_photo_connections(all_detections)

    # Save results
    save_results(all_detections, object_index, edges, args.goal, args.output)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.goal}")
    print(f"{'='*60}")
    print(f"Processed {len(all_detections)} photos")
    print(f"Found {len(object_index)} unique object types")
    print(f"Computed {len(edges)} photo-to-photo connections")

    # Show goal-related objects
    goal_words = [w.lower() for w in args.goal.split() if len(w) > 2]
    print(f"\nGoal-related detections:")
    for label, oi in object_index.items():
        if any(w in label.lower() for w in goal_words):
            photos = sorted(set(oi.photos))
            print(f"  {label}: found in {len(photos)} photos "
                  f"(best: {oi.best_photo} @ {oi.best_score:.3f})")
            for p in photos:
                print(f"    - {p}")

    print(f"\nAll detected objects:")
    for label, oi in sorted(object_index.items(), key=lambda x: -x[1].total_count):
        print(f"  {label}: {len(set(oi.photos))} photos, {oi.total_count} instances")

    # Auto-generate topological map if requested
    if args.map:
        print(f"\n{'='*60}")
        print("GENERATING TOPOLOGICAL MAP")
        print(f"{'='*60}")
        from generate_topomap import (
            load_detections, cluster_photos_into_zones, build_zone_info,
            generate_zone_label, build_edges, find_goal_zones, parse_goals,
            find_start_zone, find_navigation_path, render_map,
        )
        import networkx as nx

        goals = parse_goals(args.goal)

        photos = load_detections(args.output)
        zones = cluster_photos_into_zones(photos)
        zone_info = build_zone_info(zones)
        goal_zones = find_goal_zones(zone_info, goals)
        start_zone = find_start_zone(zone_info, args.start)

        G = nx.Graph()
        used_names = set()
        for zid, info in zone_info.items():
            label = generate_zone_label(zid, info, used_names)
            G.add_node(zid, label=label)
        for u, v, shared in build_edges(zones, zone_info):
            G.add_edge(u, v, shared=shared)

        path_edges = find_navigation_path(G, start_zone, goal_zones)
        render_map(G, zone_info, goal_zones, start_zone, path_edges,
                   goals, args.map_output)
        print(f"Map saved: {args.map_output}.png, {args.map_output}_hires.png")


if __name__ == "__main__":
    main()
