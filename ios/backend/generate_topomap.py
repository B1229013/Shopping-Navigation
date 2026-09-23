"""Fully automatic topological map generator.

Reads GroundingDINO detection results and automatically:
1. Clusters photos into zones based on object similarity
2. Determines edges between zones from shared objects
3. Lays out the graph automatically
4. Highlights goal objects and navigation paths
5. Shows ALL detected objects with confidence scores per zone

No hardcoded zones, edges, or positions — everything is derived
from GroundingDINO detections.

Usage:
    python generate_topomap.py                                          # defaults
    python generate_topomap.py --goal "refrigerator,fire extinguisher"  # multiple goals
    python generate_topomap.py --start IMG_1434.jpg                     # set start photo
"""
import argparse
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


# ── Step 1: Load & deduplicate ──────────────────────────────────────────

def load_detections(path: str) -> list:
    """Load detection JSON and return deduplicated photo list."""
    with open(path, 'r') as f:
        data = json.load(f)
    seen = set()
    photos = []
    for p in data['photos']:
        if p['filename'] not in seen:
            seen.add(p['filename'])
            photos.append(p)
    photos.sort(key=lambda p: p['filename'])
    return photos


# ── Step 2: Compute similarity between photos ──────────────────────────

def photo_object_set(photo: dict) -> set:
    """Get the set of unique object labels detected in a photo."""
    return set(obj['label'] for obj in photo['objects'])


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Step 3: Auto-cluster photos into zones ──────────────────────────────

def cluster_photos_into_zones(photos: list, sim_threshold: float = 0.25,
                               min_zone_size: int = 2) -> list:
    """Cluster consecutive photos into zones based on object similarity."""
    if not photos:
        return []

    zones = [[photos[0]]]
    for i in range(1, len(photos)):
        prev_set = photo_object_set(photos[i - 1])
        curr_set = photo_object_set(photos[i])
        sim = jaccard(prev_set, curr_set)
        if sim < sim_threshold:
            zones.append([photos[i]])
        else:
            zones[-1].append(photos[i])

    # Merge small zones into nearest neighbor
    def zone_object_set(zone):
        labels = set()
        for p in zone:
            labels |= photo_object_set(p)
        return labels

    changed = True
    while changed:
        changed = False
        new_zones = []
        i = 0
        while i < len(zones):
            zone = zones[i]
            if len(zone) < min_zone_size:
                prev_sim = 0
                next_sim = 0
                if new_zones:
                    prev_sim = jaccard(zone_object_set(new_zones[-1]), zone_object_set(zone))
                if i + 1 < len(zones):
                    next_sim = jaccard(zone_object_set(zone), zone_object_set(zones[i + 1]))
                if prev_sim >= next_sim and prev_sim > 0.05:
                    new_zones[-1].extend(zone)
                    changed = True
                    i += 1
                    continue
                elif next_sim > 0.05 and i + 1 < len(zones):
                    zones[i + 1] = zone + zones[i + 1]
                    changed = True
                    i += 1
                    continue
            new_zones.append(zone)
            i += 1
        zones = new_zones

    return zones


# ── Step 4: Build zone metadata ─────────────────────────────────────────

def build_zone_info(zones: list) -> dict:
    """Build per-zone metadata from clustered photos."""
    zone_info = {}
    for zid, zone_photos in enumerate(zones):
        obj_best_score = {}
        obj_count = {}
        # Track per-photo detections for detail
        photo_detections = {}
        for photo in zone_photos:
            photo_objs = []
            for obj in photo['objects']:
                label = obj['label']
                score = obj['score']
                photo_objs.append((label, score))
                if label not in obj_best_score or score > obj_best_score[label]:
                    obj_best_score[label] = score
                obj_count[label] = obj_count.get(label, 0) + 1
            photo_detections[photo['filename']] = photo_objs

        # ALL objects sorted by best score (not just top 5)
        all_objects = sorted(obj_best_score.items(), key=lambda x: -x[1])
        top_objects = all_objects[:5]
        all_labels = set(obj_best_score.keys())

        zone_info[zid] = {
            'photos': [p['filename'] for p in zone_photos],
            'top_objects': top_objects,
            'all_objects': all_objects,
            'all_labels': all_labels,
            'obj_best_score': obj_best_score,
            'obj_count': obj_count,
            'photo_detections': photo_detections,
            'photo_range': f"{zone_photos[0]['filename']}-{zone_photos[-1]['filename']}",
        }
    return zone_info


# ── Step 5: Auto-generate zone labels ───────────────────────────────────

def generate_zone_label(zid: int, info: dict, used_names: set = None) -> str:
    """Generate a descriptive label for a zone based on its top objects."""
    if used_names is None:
        used_names = set()

    generic = {'door', 'sign', 'cabinet', 'wall', 'fire extinguisher'}
    scored = []
    for obj, score in info['top_objects']:
        is_generic = obj.lower() in generic
        scored.append((not is_generic, score, obj))
    scored.sort(reverse=True)

    main_obj = None
    for _, _, obj in scored:
        name = obj.title()
        if name not in used_names:
            main_obj = name
            used_names.add(name)
            break

    n_photos = len(info['photos'])
    if main_obj:
        label = f"Zone {zid}\n{main_obj} Area\n({n_photos} photos)"
    else:
        label = f"Zone {zid}\n({n_photos} photos)"
    return label


# ── Step 6: Auto-build edges between zones ──────────────────────────────

def build_edges(zones: list, zone_info: dict) -> list:
    """Build edges between zones based on shared detected objects."""
    n = len(zones)
    edges = []
    seen_edges = set()

    def add_edge(a, b, shared):
        key = (min(a, b), max(a, b))
        if key not in seen_edges and a != b:
            seen_edges.add(key)
            edges.append((a, b, shared))

    # Sequential connections
    for i in range(n - 1):
        labels_a = zone_info[i]['all_labels']
        labels_b = zone_info[i + 1]['all_labels']
        shared = sorted(labels_a & labels_b)
        add_edge(i, i + 1, shared if shared else ['adjacent'])

    # Non-adjacent connections via distinctive shared objects
    label_zone_count = Counter()
    for zid, info in zone_info.items():
        for label in info['all_labels']:
            label_zone_count[label] += 1

    ubiquitous = {label for label, count in label_zone_count.items()
                  if count > 0.6 * n}

    for i in range(n):
        for j in range(i + 2, n):
            labels_i = zone_info[i]['all_labels'] - ubiquitous
            labels_j = zone_info[j]['all_labels'] - ubiquitous
            shared = sorted(labels_i & labels_j)
            if len(shared) >= 2:
                sim = jaccard(labels_i, labels_j)
                if sim >= 0.35:
                    add_edge(i, j, shared)

    return edges


# ── Step 7: Find goal-related zones (supports multiple goals) ───────────

def parse_goals(goal_str: str) -> list:
    """Parse comma-separated goal string into list of goal keywords."""
    goals = []
    for part in goal_str.split(','):
        part = part.strip().lower()
        # Remove filler words
        words = [w for w in part.split() if w not in
                 {'find', 'the', 'a', 'an', 'all', 'every', 'where', 'is', 'are'}]
        if words:
            goals.append(' '.join(words))
    return goals


def find_goal_zones(zone_info: dict, goals: list) -> dict:
    """Find zones containing any goal object, with confidence scores.

    Returns dict: zone_id -> {goal_label: score, ...}
    """
    goal_zones = {}

    for zid, info in zone_info.items():
        zone_goals = {}
        for label, score in info['obj_best_score'].items():
            for goal in goals:
                # Match if any goal word appears in the label
                goal_words = goal.split()
                if any(w in label.lower() for w in goal_words if len(w) > 2):
                    if goal not in zone_goals or score > zone_goals[goal]:
                        zone_goals[goal] = score
        if zone_goals:
            goal_zones[zid] = zone_goals

    # Filter out weak detections per goal
    for goal in goals:
        scores = [(zid, gz[goal]) for zid, gz in goal_zones.items() if goal in gz]
        if scores:
            best = max(s for _, s in scores)
            for zid, s in scores:
                if s < max(0.45, best * 0.4):
                    del goal_zones[zid][goal]
                    if not goal_zones[zid]:
                        del goal_zones[zid]

    return goal_zones


def find_start_zone(zone_info: dict, start_photo: str = None) -> int:
    if start_photo:
        for zid, info in zone_info.items():
            if start_photo in info['photos']:
                return zid
    return 0


def find_navigation_path(G: nx.Graph, start: int, goal_zones: dict) -> set:
    path_edges = set()
    for target in goal_zones:
        try:
            path = nx.shortest_path(G, start, target)
            for i in range(len(path) - 1):
                path_edges.add((min(path[i], path[i+1]), max(path[i], path[i+1])))
        except nx.NetworkXNoPath:
            continue
    return path_edges


# ── Step 8: Render the detailed map ─────────────────────────────────────

def sequential_layout(G: nx.Graph, zones_per_row: int = 6) -> dict:
    """Snake layout mirroring the physical walk order (TASK 15).

    Nodes are laid left-to-right, wrapping every ``zones_per_row`` into a new
    row whose direction alternates (snake), so the path reads like aisles walked
    in sequence. Deterministic and bounded — unlike kamada_kawai it never
    produces NaN/huge coordinates that crash the Agg renderer on tiny graphs.
    """
    pos = {}
    nodes = sorted(G.nodes())
    for i, node in enumerate(nodes):
        row = i // zones_per_row
        col = i % zones_per_row
        if row % 2 == 1:  # reverse every other row so the walk snakes
            col = zones_per_row - 1 - col
        pos[node] = (col * 2.4, -row * 2.6)
    return pos



# Color palette for multiple goals
GOAL_COLORS = [
    ('#58a6ff', '#0d1d30', '#1f6feb'),  # blue
    ('#3fb950', '#0d2d0d', '#238636'),  # green
    ('#d2a8ff', '#1c1333', '#8957e5'),  # purple
    ('#f0883e', '#2d1a0d', '#db6d28'),  # orange
]


def render_map(
    G: nx.Graph,
    zone_info: dict,
    goal_zones: dict,
    start_zone: int,
    path_edges: set,
    goals: list,
    output_prefix: str = 'topological_map',
):
    """Render the detailed topological map as a PNG."""

    fig, ax = plt.subplots(1, 1, figsize=(24, 20))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Layout: a snake/sequential layout that mirrors how the store is physically
    # walked (TASK 15). kamada_kawai is aesthetic-only and degenerates (NaN/huge
    # coords -> renderer crash) on small or path-like graphs, so we avoid it.
    pos = sequential_layout(G, zones_per_row=6)

    # Separate edges
    regular_edges = []
    nav_path_edges = []
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        if key in path_edges:
            nav_path_edges.append((u, v))
        else:
            regular_edges.append((u, v))

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, ax=ax,
                           edge_color='#484f58', width=2, style='solid', alpha=0.5)
    if nav_path_edges:
        nx.draw_networkx_edges(G, pos, edgelist=nav_path_edges, ax=ax,
                               edge_color='#58a6ff', width=3.5, style='dashed', alpha=0.9)

    # Node colors and sizes
    node_colors = []
    node_sizes = []
    node_edge_colors = []
    for n in G.nodes():
        if n == start_zone:
            node_colors.append('#f85149')
            node_sizes.append(5000)
            node_edge_colors.append('#ff7b72')
        elif n in goal_zones:
            node_colors.append('#1f6feb')
            node_sizes.append(4500)
            node_edge_colors.append('#58a6ff')
        else:
            node_colors.append('#21262d')
            node_sizes.append(3800)
            node_edge_colors.append('#484f58')

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           edgecolors=node_edge_colors, linewidths=2.5)

    # Node labels (zone name + photo range)
    for n, (x, y) in pos.items():
        label = G.nodes[n].get('label', f'Zone {n}')
        is_special = (n == start_zone or n in goal_zones)
        color = '#ffffff' if is_special else '#c9d1d9'
        fontsize = 8.5 if is_special else 7.5
        weight = 'bold' if is_special else 'normal'
        ax.text(x, y + 0.05, label, ha='center', va='center', fontsize=fontsize,
                color=color, fontweight=weight, linespacing=1.3)

    # Edge labels — show ALL shared objects (not just 3)
    for u, v, d in G.edges(data=True):
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]
        length = max(0.01, (dx**2 + dy**2)**0.5)
        ox, oy = -dy / length * 0.2, dx / length * 0.2
        shared_list = d.get('shared', [])
        shared_text = ', '.join(shared_list[:5])
        if len(shared_list) > 5:
            shared_text += f' +{len(shared_list)-5}'
        key = (min(u, v), max(u, v))
        edge_color = '#58a6ff' if key in path_edges else '#8b949e'
        ax.text(x + ox, y + oy, shared_text, ha='center', va='center',
                fontsize=5.5, color=edge_color, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#161b22',
                          edgecolor='#30363d', alpha=0.9))

    # ── DETAILED: ALL detected objects per zone (below node) ──
    for zid, info in zone_info.items():
        if zid not in pos:
            continue
        x, y = pos[zid]

        # Show ALL objects with count and confidence
        obj_lines = []
        for name, score in info['all_objects']:
            count = info['obj_count'][name]
            obj_lines.append(f'{name}: {score:.2f} (x{count})')
        obj_text = '  |  '.join(obj_lines)

        ax.text(x, y - 0.48, obj_text, ha='center', va='top', fontsize=5,
                color='#d2a8ff', family='monospace',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#1c2333',
                          edgecolor='#30363d', alpha=0.85))

    # ── DETAILED: Photo filenames per zone (further below) ──
    for zid, info in zone_info.items():
        if zid not in pos:
            continue
        x, y = pos[zid]
        photo_text = ', '.join(info['photos'])
        ax.text(x, y - 0.7, photo_text, ha='center', va='top', fontsize=4.5,
                color='#6e7681', family='monospace',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='#0d1117',
                          edgecolor='#21262d', alpha=0.8))

    # ── Goal confidence labels above goal zones ──
    for zid, goal_scores in goal_zones.items():
        if zid not in pos:
            continue
        x, y = pos[zid]
        # Build multi-goal label
        labels = []
        for gi, goal in enumerate(goals):
            if goal in goal_scores:
                color_idx = gi % len(GOAL_COLORS)
                labels.append((goal, goal_scores[goal], GOAL_COLORS[color_idx]))

        for i, (goal_name, score, (text_color, bg_color, _)) in enumerate(labels):
            offset_y = 0.55 + i * 0.35
            ax.text(x, y + offset_y, f'{goal_name}: {score:.3f}',
                    ha='center', va='bottom', fontsize=9, color=text_color,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color,
                              edgecolor=text_color, alpha=0.95))

    # "YOU ARE HERE" label on start zone
    if start_zone in pos:
        x, y = pos[start_zone]
        n_goal_labels = len(goal_zones.get(start_zone, {}))
        offset = 0.55 + n_goal_labels * 0.35
        ax.text(x, y + offset, 'YOU ARE HERE', ha='center', va='bottom',
                fontsize=10, color='#f85149', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#2d0d0d',
                          edgecolor='#f85149', alpha=0.95))

    # Title
    n_photos = sum(len(info['photos']) for info in zone_info.values())
    n_obj_types = len(set(label for info in zone_info.values() for label in info['all_labels']))
    goal_str = ' + '.join(goals)
    ax.set_title(
        f'UniGoal Topological Map — Fully Automatic from GroundingDINO\n'
        f'Goals: "{goal_str}" | {n_photos} photos → {len(zone_info)} zones → '
        f'{G.number_of_edges()} edges | {n_obj_types} unique object types detected\n'
        f'Zones auto-clustered by Jaccard similarity | Edges = shared GroundingDINO detections | '
        f'Purple = all objects(confidence)(count)',
        fontsize=13, color='#58a6ff', fontweight='bold', pad=20)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#f85149', edgecolor='#ff7b72',
                       label='Start Position (YOU ARE HERE)'),
    ]
    for gi, goal in enumerate(goals):
        color_idx = gi % len(GOAL_COLORS)
        text_c, _, node_c = GOAL_COLORS[color_idx]
        legend_elements.append(
            mpatches.Patch(facecolor=node_c, edgecolor=text_c,
                           label=f'Goal: {goal}'))
    legend_elements += [
        mpatches.Patch(facecolor='#21262d', edgecolor='#484f58', label='Other Zone'),
        plt.Line2D([0], [0], color='#58a6ff', linewidth=3, linestyle='--',
                   label='Navigation Path'),
        plt.Line2D([0], [0], color='#484f58', linewidth=2,
                   label='Connected (shared objects)'),
        mpatches.Patch(facecolor='#1c2333', edgecolor='#30363d',
                       label='All detections: obj: conf (xcount)'),
        mpatches.Patch(facecolor='#0d1117', edgecolor='#21262d',
                       label='Photo filenames per zone (gray)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # Stats box
    # Collect global object stats
    global_obj_count = Counter()
    global_obj_best = {}
    for info in zone_info.values():
        for label, score in info['obj_best_score'].items():
            global_obj_count[label] += info['obj_count'][label]
            if label not in global_obj_best or score > global_obj_best[label]:
                global_obj_best[label] = score

    stats_lines = [
        f"GroundingDINO SwinT OGC | box_thresh=0.30 | text_thresh=0.25",
        f"Total: {n_photos} photos, {n_obj_types} object types, "
        f"{sum(global_obj_count.values())} detections",
        f"Top objects globally:",
    ]
    for label, count in global_obj_count.most_common(8):
        best = global_obj_best[label]
        stats_lines.append(f"  {label}: {count} detections, best conf={best:.3f}")

    stats_text = '\n'.join(stats_lines)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=6,
            color='#8b949e', verticalalignment='bottom', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#161b22',
                      edgecolor='#30363d', alpha=0.9))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=150, facecolor='#0d1117', bbox_inches='tight')
    print(f'Saved: {output_prefix}.png')
    plt.savefig(f'{output_prefix}_hires.png', dpi=300, facecolor='#0d1117', bbox_inches='tight')
    print(f'Saved: {output_prefix}_hires.png')
    plt.close()


# ── Main pipeline ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fully automatic topological map from GroundingDINO detections"
    )
    parser.add_argument('--input', '-i', default='detections_groundingdino.json',
                        help='Detection JSON from batch_mapper.py')
    parser.add_argument('--goal', '-g', default='refrigerator',
                        help='Goal objects, comma-separated (e.g. "refrigerator,fire extinguisher")')
    parser.add_argument('--start', '-s', default=None,
                        help='Start photo filename (default: first photo)')
    parser.add_argument('--output', '-o', default='topological_map',
                        help='Output filename prefix (default: topological_map)')
    parser.add_argument('--sim-threshold', type=float, default=0.25,
                        help='Jaccard similarity threshold for zone boundaries (default: 0.25)')
    args = parser.parse_args()

    # Step 1: Load detections
    print(f'Loading detections from {args.input}...')
    photos = load_detections(args.input)
    print(f'  {len(photos)} unique photos')

    # Step 2: Auto-cluster into zones
    print(f'Clustering photos into zones (threshold={args.sim_threshold})...')
    zones = cluster_photos_into_zones(photos, sim_threshold=args.sim_threshold)
    print(f'  {len(zones)} zones created')

    # Step 3: Build zone metadata
    zone_info = build_zone_info(zones)
    for zid, info in zone_info.items():
        all_objs = ', '.join(f'{o}({s:.2f})' for o, s in info['all_objects'])
        print(f'  Zone {zid}: {len(info["photos"])} photos [{info["photo_range"]}]')
        print(f'    Objects: {all_objs}')

    # Step 4: Parse goals, find goal zones and start zone
    goals = parse_goals(args.goal)
    print(f'\nGoals: {goals}')
    goal_zones = find_goal_zones(zone_info, goals)
    start_zone = find_start_zone(zone_info, args.start)
    for zid, gscores in goal_zones.items():
        print(f'  Zone {zid}: {gscores}')
    print(f'Start zone: {start_zone}')

    # Step 5: Build graph
    print('\nBuilding graph...')
    edges = build_edges(zones, zone_info)
    G = nx.Graph()
    used_names = set()
    for zid, info in zone_info.items():
        label = generate_zone_label(zid, info, used_names)
        G.add_node(zid, label=label)
    for u, v, shared in edges:
        G.add_edge(u, v, shared=shared)
    print(f'  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

    # Step 6: Find navigation path
    path_edges = find_navigation_path(G, start_zone, goal_zones)
    if path_edges:
        print(f'  Navigation path edges: {path_edges}')
    else:
        print('  No navigation path found (start may already be at goal)')

    # Step 7: Render
    print('\nRendering map...')
    render_map(G, zone_info, goal_zones, start_zone, path_edges,
               goals, args.output)
    print('Done!')


if __name__ == '__main__':
    main()
