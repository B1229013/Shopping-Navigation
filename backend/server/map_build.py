"""Map-building workspace: Sensor Lab exports → <workspace>/<set>/ photos + topomap.json.

Shared by build_map_v2_topomap.py (command line) and the builder's 「路線」 dialog
(/topomap-sources, /topomap-walks). Pure logic lives in server/map_v2.py.

Every store has its own workspace, MAP_BUILD_ROOT/<店名>/ (workspace_for), so building a
second store never touches the first one's photos, alignment or confirmed map.

A walk keeps its set id ("01", "02", …) for as long as it stays in the workspace, and ids are
never reused (topomap.json "next_set"), so the builder's saved edits — keyed by "set02_wp5" —
always land on the same waypoint.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from server.map_v2 import (PHOTO_RE, build_topomap, extract_waypoints, parse_sensor_log,
                           walk_stats, walk_track)

CSV_NAME = "sensor_log.csv"
IMPORTS_DIR = "imports"            # zips uploaded from the builder, inside the workspace
_HEADER_DATE = re.compile(r"Export\s*[—-]\s*(\d{8})_(\d{6})")


_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def place_folder(place: str) -> str:
    """Folder name for a store: its name with characters Windows forbids replaced by "_"
    ("A7家樂福（9/16)" → "A7家樂福（9_16)"); the real name stays inside topomap.json."""
    name = _UNSAFE_CHARS.sub("_", place or "").strip().rstrip(". ")
    if not name or set(name) <= {"."}:
        raise ValueError("店名不能是空的")
    return name


def workspace_for(place: str, root: Path) -> Path:
    return Path(root) / place_folder(place)


def list_places(root: Path) -> List[dict]:
    """Stores with a workspace under root (a folder holding topomap.json), newest first."""
    out = []
    for d in Path(root).iterdir() if Path(root).is_dir() else []:
        topo_path = d / "topomap.json"
        if not topo_path.is_file():
            continue
        try:
            topo = json.loads(topo_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        confirmed = d / "topomap_confirmed.json"
        out.append({"place": topo.get("place") or d.name, "folder": d.name,
                    "nodes": len(topo.get("nodes", [])),
                    "walks": len(topo.get("walks") or topo.get("stats", {}).get("sets") or []),
                    "confirmed": confirmed.is_file(),
                    "updated": max(p.stat().st_mtime for p in (topo_path, confirmed) if p.exists())})
    return sorted(out, key=lambda p: -p["updated"])


def _is_photo(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg"))


def _sync_photos(photo_dir: Path, items: Iterable[tuple]) -> List[str]:
    """Write (name, size, read_bytes) items into photo_dir, skipping files already there with the
    same size, and remove photos that are no longer part of this walk."""
    photo_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for name, size, read in items:
        dst = photo_dir / name
        if not (dst.is_file() and dst.stat().st_size == size):
            dst.write_bytes(read())
        names.append(name)
    keep = set(names)
    for p in photo_dir.iterdir():
        if p.is_file() and _is_photo(p.name) and p.name not in keep:
            p.unlink()
    return names


def read_walk(src: Path, photo_dir: Path) -> Tuple[List[dict], List[str]]:
    """One Sensor Lab export (.zip, or a folder with sensor_log.csv + photos/) → its CSV rows,
    with the photos copied into photo_dir. Raises ValueError when it is not one."""
    if src.is_file() and zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as zf:
            csv_member = next((n for n in zf.namelist() if n.endswith(CSV_NAME)), None)
            if csv_member is None:
                raise ValueError(f"{src.name}: zip 裡沒有 {CSV_NAME}")
            rows = parse_sensor_log(zf.read(csv_member).decode("utf-8-sig"))
            infos = [i for i in zf.infolist() if "/photos/" in f"/{i.filename}" and _is_photo(i.filename)]
            names = _sync_photos(photo_dir, ((Path(i.filename).name, i.file_size,
                                              (lambda i=i: zf.read(i))) for i in infos))
    elif src.is_dir():
        csv_path = next(src.rglob(CSV_NAME), None)
        if csv_path is None:
            raise ValueError(f"{src.name}: 資料夾裡沒有 {CSV_NAME}")
        rows = parse_sensor_log(csv_path.read_text(encoding="utf-8-sig"))
        files = [p for p in src.rglob("*") if p.parent.name == "photos" and _is_photo(p.name)]
        names = _sync_photos(photo_dir, ((p.name, p.stat().st_size, p.read_bytes) for p in files))
    else:
        raise ValueError(f"{src}: 不是 Sensor Lab 的 zip 或資料夾")
    if not rows:
        raise ValueError(f"{src.name}: {CSV_NAME} 讀不到資料列")
    return rows, names


def build_workspace(place: str, walks: Sequence[Tuple[str, Path]], ws: Path,
                    next_set: Optional[int] = None) -> Tuple[dict, List[str]]:
    """Read every (set_id, source) walk into ws and write ws/topomap.json; returns it plus
    report lines. Photo folders of walks that are no longer used are removed."""
    report: List[str] = []
    built, info = [], {}
    for set_id, src in walks:
        rows, names = read_walk(src, ws / set_id)
        wps, warnings = extract_waypoints(rows, names)
        stats = walk_stats(rows)
        report.append(f"[{set_id}] {src.name}: {len(wps)} 個拍照節點、{stats['steps']} 步、"
                      f"{stats['distance_m']} m")
        report += [f"     ⚠ {w}" for w in warnings]
        built.append((set_id, wps))
        info[set_id] = {"source": str(src), "stats": stats, "track": walk_track(rows, wps)}
    topo = build_topomap(place, built, info)
    used = {s for s, _ in walks}
    topo["next_set"] = max([next_set or 0] + [int(s) + 1 for s in used if s.isdigit()])
    for d in ws.iterdir() if ws.exists() else []:
        if d.is_dir() and re.fullmatch(r"\d{2,}", d.name) and d.name not in used:
            shutil.rmtree(d)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "topomap.json").write_text(json.dumps(topo, ensure_ascii=False, indent=1), encoding="utf-8")
    return topo, report


def assign_sets(current: Optional[dict], sources: Sequence[Path]) -> Tuple[List[Tuple[str, Path]], int]:
    """Set ids for the chosen sources: a walk already in topomap.json keeps its id, a new one
    gets the next id never used in this workspace. Returns ([(set_id, source)] by id, next_set)."""
    current = current or {}
    had = {str(Path(w["source"]).resolve()): w["set"] for w in current.get("walks", []) if w.get("source")}
    used_ids = [int(w["set"]) for w in current.get("walks", []) if str(w.get("set", "")).isdigit()]
    next_id = max([current.get("next_set") or 0, max(used_ids, default=0) + 1, 1])
    out = []
    for src in sources:
        key = str(Path(src).resolve())
        if key in had:
            out.append((had[key], Path(src)))
        else:
            out.append((f"{next_id:02d}", Path(src)))
            next_id += 1
    return sorted(out, key=lambda t: t[0]), next_id


def _source_entry(path: Path, kind: str, root: Path) -> Optional[dict]:
    """A selectable Sensor Lab export with a quick summary, or None if it is not one."""
    try:
        if kind == "zip":
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                csv_member = next((n for n in names if n.endswith(CSV_NAME)), None)
                if csv_member is None:
                    return None
                with zf.open(csv_member) as f:
                    first = f.readline().decode("utf-8-sig", "ignore")
                photos = [Path(n).name for n in names if _is_photo(n)]
        else:
            csv_path = path / CSV_NAME
            with csv_path.open(encoding="utf-8-sig", errors="ignore") as f:
                first = f.readline()
            photos = [p.name for p in (path / "photos").glob("*")] if (path / "photos").is_dir() else []
    except (OSError, zipfile.BadZipFile):
        return None
    wps = {m.group(1) for m in (PHOTO_RE.match(n) for n in photos) if m}
    m = _HEADER_DATE.search(first)
    when = f"{m.group(1)[:4]}/{m.group(1)[4:6]}/{m.group(1)[6:]} {m.group(2)[:2]}:{m.group(2)[2:4]}" if m else ""
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = path.name
    return {"path": str(path.resolve()), "name": path.stem if kind == "zip" else path.name,
            "rel": rel, "kind": kind, "waypoints": len(wps), "photos": len(photos), "recorded": when}


def list_sources(roots: Sequence[Path], ws: Optional[Path]) -> List[dict]:
    """Sensor Lab exports found in the source folders (folders holding sensor_log.csv, up to
    two levels down, and .zip files) plus zips uploaded into the store's workspace (if any)."""
    found: Dict[str, dict] = {}
    for root in [*roots, *([ws / IMPORTS_DIR] if ws else [])]:
        if not root.is_dir():
            continue
        candidates = [root, *root.iterdir()]
        candidates += [c for d in root.iterdir() if d.is_dir() for c in d.iterdir()]
        for p in candidates:
            if p.is_dir() and (p / CSV_NAME).is_file():
                e = _source_entry(p, "folder", root)
            elif p.is_file() and p.suffix.lower() == ".zip":
                e = _source_entry(p, "zip", root)
            else:
                continue
            if e:
                found.setdefault(e["path"], e)
    return sorted(found.values(), key=lambda e: (e["recorded"] or "", e["name"]), reverse=True)
