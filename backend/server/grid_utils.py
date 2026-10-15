"""3×3 grid utilities for object spatial analysis.

Computes grid cell coverage, piece count, depth class, and cell-to-cell
adjacency pairs from normalized [0,1] bounding boxes.
"""
from __future__ import annotations

from typing import List, Tuple

ROWS = ["top", "mid", "bottom"]
COLS = ["left", "center", "right"]
GRID_SIZE = 3

CELL_COORDS = {
    f"{r}-{c}": (ri, ci)
    for ri, r in enumerate(ROWS)
    for ci, c in enumerate(COLS)
}

COORD_TO_CELL = {v: k for k, v in CELL_COORDS.items()}

_DIR_NAMES = {
    (-1, -1): "above-left",
    (-1,  0): "above",
    (-1,  1): "above-right",
    ( 0, -1): "left",
    ( 0,  0): "overlap",
    ( 0,  1): "right",
    ( 1, -1): "below-left",
    ( 1,  0): "below",
    ( 1,  1): "below-right",
}


def compute_grid_cells(
    bbox: List[float],
    grid_size: int = GRID_SIZE,
) -> Tuple[str, List[str], int, str]:
    """Compute grid coverage from a normalized [0,1] bounding box.

    Returns (grid_cell, grid_cells, piece_count, depth_class).
    """
    x1, y1, x2, y2 = bbox
    col_start = min(int(x1 * grid_size), grid_size - 1)
    col_end = min(int(x2 * grid_size), grid_size - 1)
    row_start = min(int(y1 * grid_size), grid_size - 1)
    row_end = min(int(y2 * grid_size), grid_size - 1)

    rows = ROWS if grid_size == 3 else [str(i) for i in range(grid_size)]
    cols = COLS if grid_size == 3 else [str(i) for i in range(grid_size)]

    cells = []
    for r in range(row_start, row_end + 1):
        for c in range(col_start, col_end + 1):
            cells.append(f"{rows[r]}-{cols[c]}")

    cx_col = min(int(((x1 + x2) / 2) * grid_size), grid_size - 1)
    cx_row = min(int(((y1 + y2) / 2) * grid_size), grid_size - 1)
    grid_cell = f"{rows[cx_row]}-{cols[cx_col]}"

    piece_count = len(cells)
    if piece_count == 1:
        depth_class = "far"
    elif piece_count <= 3:
        depth_class = "mid"
    else:
        depth_class = "near"

    return grid_cell, cells, piece_count, depth_class


def compute_cell_pairs(
    cells_a: List[str],
    cells_b: List[str],
) -> List[dict]:
    """Compute all cell-to-cell adjacency pairs between two objects.

    Only pairs where |row_diff| <= 1 AND |col_diff| <= 1 are included.
    Returns list of {"a": cell_name, "b": cell_name, "dir": direction}.
    """
    pairs = []
    for ca in cells_a:
        ra, col_a = CELL_COORDS.get(ca, (None, None))
        if ra is None:
            continue
        for cb in cells_b:
            rb, col_b = CELL_COORDS.get(cb, (None, None))
            if rb is None:
                continue
            dr = rb - ra
            dc = col_b - col_a
            if abs(dr) <= 1 and abs(dc) <= 1:
                direction = _DIR_NAMES.get((dr, dc), "unknown")
                pairs.append({"a": ca, "b": cb, "dir": direction})
    return pairs


def are_adjacent(cells_a: List[str], cells_b: List[str]) -> bool:
    """Check if two objects have any adjacent or overlapping cells."""
    for ca in cells_a:
        ra, col_a = CELL_COORDS.get(ca, (None, None))
        if ra is None:
            continue
        for cb in cells_b:
            rb, col_b = CELL_COORDS.get(cb, (None, None))
            if rb is None:
                continue
            if abs(rb - ra) <= 1 and abs(col_b - col_a) <= 1:
                return True
    return False


def depth_diff(da: str, db: str) -> str:
    """Compare two depth classes: same_level / closer / farther."""
    order = {"far": 0, "mid": 1, "near": 2}
    va = order.get(da, 1)
    vb = order.get(db, 1)
    if va == vb:
        return "same_level"
    return "closer" if vb > va else "farther"
