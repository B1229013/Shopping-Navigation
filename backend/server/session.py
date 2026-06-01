"""Session state and in-memory store."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from server.topomap import TopoMap


@dataclass
class Session:
    id: str
    goal: str
    goal_objects: List[str]
    topomap: TopoMap = field(default_factory=TopoMap)
    history: List[dict] = field(default_factory=list)
    pending_question: Optional[str] = None
    pending_is_confirm: bool = False  # is pending_question our arrival-confirm question?
    last_planned_action: Optional[str] = None
    arrived: bool = False
    goal_node: Optional[int] = None
    last_node_id: Optional[int] = None
    last_detections: List[dict] = field(default_factory=list)
    last_photo_path: Optional[str] = None
    last_ocr_summary: Optional[str] = None
    last_ocr_matches: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


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
