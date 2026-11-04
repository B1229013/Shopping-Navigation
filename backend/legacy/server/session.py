"""Session state and in-memory store."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from server.topomap import TopoMap
from server.topomap_v2 import TopoGraphV2


@dataclass
class SubGoal:
    """One item in a multi-goal shopping list."""
    name: str
    goal_objects: List[str] = field(default_factory=list)
    goal_photo_ids: List[int] = field(default_factory=list)
    arrived: bool = False
    arrived_node: Optional[int] = None
    # Auto-appended checkout/exit destinations. These are navigated to like any
    # other goal but are excluded from the user-facing "N goals" count.
    is_terminal: bool = False


@dataclass
class Session:
    id: str
    goal: str
    goal_objects: List[str]
    topomap: TopoMap = field(default_factory=TopoMap)
    history: List[dict] = field(default_factory=list)
    pending_question: Optional[str] = None
    pending_is_confirm: bool = False
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
    created_at: datetime = field(default_factory=datetime.utcnow)
    # ── 地圖導航模式（從 Neo4j 載入 TopoGraphV2）──
    topo_v2: Optional[TopoGraphV2] = None
    place_name: str = ""
    nav_mode: str = "explore"
    goal_photo_ids: List[int] = field(default_factory=list)
    planned_path: List[int] = field(default_factory=list)
    last_route_guidance: Optional[str] = None
    last_route_waypoints: List[dict] = field(default_factory=list)
    last_matched_direction: str = ""
    # ── 多目標 TSP ──
    sub_goals: List[SubGoal] = field(default_factory=list)
    tsp_order: List[int] = field(default_factory=list)
    current_goal_idx: int = 0
    # True while the multi-goal visit order is still provisional and must be
    # (re)solved from the user's real localized position on the first successful
    # localization. See sensor_nav._setup_map_goals / _reorder_goals_from_start.
    tsp_pending_reorder: bool = False

    @property
    def current_sub_goal(self) -> Optional[SubGoal]:
        if not self.sub_goals or not self.tsp_order:
            return None
        if self.current_goal_idx >= len(self.tsp_order):
            return None
        return self.sub_goals[self.tsp_order[self.current_goal_idx]]

    @property
    def all_sub_goals_arrived(self) -> bool:
        return bool(self.sub_goals) and all(sg.arrived for sg in self.sub_goals)

    @property
    def shopping_goal_count(self) -> int:
        """Number of user shopping goals, excluding auto-appended checkout/exit."""
        return sum(1 for sg in self.sub_goals if not sg.is_terminal)

    @property
    def shopping_progress_index(self) -> int:
        """1-based position of the current goal among shopping goals in the visit
        order. Counts shopping goals up to and including the current one; while at
        a terminal (checkout/exit) it returns the shopping-goal total."""
        count = 0
        for j, order_idx in enumerate(self.tsp_order):
            if 0 <= order_idx < len(self.sub_goals) and not self.sub_goals[order_idx].is_terminal:
                count += 1
            if j == self.current_goal_idx:
                break
        return count

    def advance_to_next_goal(self) -> Optional[SubGoal]:
        """Advance to the next unfinished sub-goal. Returns it, or None if all done."""
        self.current_goal_idx += 1
        sg = self.current_sub_goal
        if sg is not None:
            self.goal_photo_ids = sg.goal_photo_ids
            self.goal_objects = sg.goal_objects
            self.planned_path = []
            self.last_route_guidance = None
            self.last_route_waypoints = []
        return sg


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def create(self, goal: str, goal_objects: List[str]) -> Session:
        sid = uuid.uuid4().hex[:8]
        s = Session(id=sid, goal=goal, goal_objects=goal_objects)
        self._sessions[sid] = s
        return s

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)
