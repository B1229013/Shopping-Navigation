"""Session state and in-memory store."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from server.topomap import TopoMap


@dataclass
class Session:
    id: str
    goal: str
    goal_objects: List[str]
    place: Optional[str] = None  # which map/place this session navigates in
    # Product words only (goal_objects minus section/landmark context). This is
    # what the arrival gate, crop-verify and "goal visible" checks match against;
    # goal_objects (the superset) is still what the detector is prompted with.
    target_objects: List[str] = field(default_factory=list)
    # The rest of goal_objects: section / landmark words shown to the VLM as
    # "clues near the target", never as the target itself.
    context_objects: List[str] = field(default_factory=list)
    topomap: TopoMap = field(default_factory=TopoMap)
    history: List[dict] = field(default_factory=list)
    pending_question: Optional[str] = None
    pending_is_confirm: bool = False  # is pending_question our arrival-confirm question?
    last_planned_action: Optional[str] = None
    arrived: bool = False
    pending_arrival: bool = False
    goal_node: Optional[int] = None
    last_node_id: Optional[int] = None
    last_detections: List[dict] = field(default_factory=list)
    last_photo_path: Optional[str] = None
    last_ocr_summary: Optional[str] = None
    last_ocr_matches: List[str] = field(default_factory=list)
    last_img_w: int = 0
    last_img_h: int = 0
    corrections: List[str] = field(default_factory=list)
    false_positive_nodes: List[int] = field(default_factory=list)
    repeated_misidentification_count: int = 0
    last_corrected_nid: Optional[int] = None   # Neo4j reference node from visual localization
    created_at: datetime = field(default_factory=datetime.utcnow)

    # ── Heading / orientation ──────────────────────────────────────
    user_heading: Optional[float] = None          # estimated heading in degrees [0, 360)
    heading_slot: Optional[str] = None            # "front" / "right" / "back" / "left"
    heading_confidence: float = 0.0               # 0.0–1.0

    # ── Multi-target route planning ─────────────────────────────────
    # All target node IDs the user wants to visit (resolved from goal_objects)
    target_nodes: List[int] = field(default_factory=list)
    # Targets already visited (removed from routing but kept for history)
    visited_targets: Set[int] = field(default_factory=set)
    # The current optimised route plan (from path_planner)
    route_plan: Optional[object] = None   # server.path_planner.RoutePlan
    # Index into route_plan.legs — which leg the user is currently on
    current_leg_index: int = 0
    # Fixed terminal nodes
    checkout_node: Optional[int] = None
    exit_node: Optional[int] = None

    @property
    def remaining_targets(self) -> List[int]:
        """Targets not yet visited, in their current planned order."""
        return [t for t in self.target_nodes if t not in self.visited_targets]

    def mark_target_visited(self, node: int) -> None:
        """Mark a target as reached."""
        self.visited_targets.add(node)

    def add_target(self, node: int) -> None:
        """Add a new target mid-route."""
        if node not in self.target_nodes:
            self.target_nodes.append(node)

    def remove_target(self, node: int) -> None:
        """Remove a target mid-route (user changed their mind)."""
        self.target_nodes = [t for t in self.target_nodes if t != node]
        self.visited_targets.discard(node)

    @property
    def all_targets_visited(self) -> bool:
        return len(self.remaining_targets) == 0


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def create(self, goal: str, goal_objects: List[str], place: Optional[str] = None,
               target_objects: Optional[List[str]] = None,
               context_objects: Optional[List[str]] = None) -> Session:
        sid = uuid.uuid4().hex[:8]
        s = Session(id=sid, goal=goal, goal_objects=goal_objects, place=place,
                    target_objects=list(target_objects if target_objects is not None else goal_objects),
                    context_objects=list(context_objects or []))
        self._sessions[sid] = s
        return s

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)
