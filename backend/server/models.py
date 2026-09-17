"""Pydantic request/response models and internal dataclasses."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class VLMAction(str, Enum):
    ARRIVED = "ARRIVED"
    MOVE = "MOVE"
    ASK = "ASK"


class Detection(BaseModel):
    label: str
    box: List[float] = Field(..., description="[x1,y1,x2,y2] absolute pixels")
    score: float


class VLMResponse(BaseModel):
    action: VLMAction
    guidance: str
    question: Optional[str] = None
    vlm_summary: str = ""


class StartSessionRequest(BaseModel):
    goal: str
    place: Optional[str] = None  # which map/place to navigate in


class StartSessionResponse(BaseModel):
    session_id: str
    guidance: str
    action: Literal["TAKE_PHOTO"] = "TAKE_PHOTO"
    goal_objects: List[str]
    place: Optional[str] = None  # the resolved place used for this session


class AnswerRequest(BaseModel):
    answer: str


class ConfirmArrivalRequest(BaseModel):
    # "repeated_misidentification": the user has rejected an ARRIVED claim for
    # this same goal multiple times in a row — unlike false_positive/wrong_instance
    # (which just resume the same MOVE search), this tells the backend the current
    # strategy keeps failing, so it should change approach (e.g. ask the user a
    # clarifying question about the item's appearance instead of just retrying).
    kind: Literal["confirmed", "false_positive", "wrong_instance", "repeated_misidentification"]


class TurnResponse(BaseModel):
    action: VLMAction
    guidance: str
    question: Optional[str] = None
    node_id: int
    annotated_photo_url: Optional[str] = None
    # Neo4j visual localization (position correction)
    corrected_node_id: Optional[int] = None       # reference map node ID
    corrected_confidence: Optional[float] = None   # 0.0–1.0
    corrected_location: Optional[str] = None       # human-readable location name
    # Heading estimation from directional photo matching
    heading_deg: Optional[float] = None            # estimated user heading [0, 360)
    heading_slot: Optional[str] = None             # "front"/"right"/"back"/"left"
    heading_confidence: Optional[float] = None     # 0.0–1.0
    # Route-aware navigation context
    next_instruction: Optional[str] = None         # e.g. "右轉，走向第3區"
    remaining_targets: Optional[int] = None        # how many targets left


class NodeJSON(BaseModel):
    id: int
    photo: str
    detected: List[str]
    summary: str
    timestamp: str


class EdgeJSON(BaseModel):
    from_id: int = Field(..., alias="from")
    to: int
    action: str

    class Config:
        populate_by_name = True


class MapJSON(BaseModel):
    nodes: List[NodeJSON]
    edges: List[EdgeJSON]
    current_node: Optional[int]
    goal_node: Optional[int]


class ErrorResponse(BaseModel):
    error: str
    detail: str


# ── Multi-target route planning ─────────────────────────────────────────

class PlanRouteRequest(BaseModel):
    """Request to plan an optimised multi-target route."""
    targets: List[str] = Field(..., description="List of target queries (product names, locations)")
    start_node: Optional[int] = Field(None, description="Starting node ID (from visual localization)")
    checkout_node: Optional[int] = Field(None, description="Checkout/counter node ID")
    exit_node: Optional[int] = Field(None, description="Exit node ID")
    place: Optional[str] = Field(None, description="Map/place name")


class RouteLegResponse(BaseModel):
    from_node: int = Field(..., alias="from")
    to: int
    path: List[int]
    cost: float
    actions: List[str]
    purpose: str  # "target" | "checkout" | "exit"

    class Config:
        populate_by_name = True


class PlanRouteResponse(BaseModel):
    visit_order: List[int] = Field(..., description="Target nodes in optimal visit order")
    total_cost: float
    full_path: List[int]
    legs: List[RouteLegResponse]
    resolved_targets: List[dict] = Field(default_factory=list, description="Query → node resolution results")
    checkout_node: Optional[int] = None
    exit_node: Optional[int] = None


class ModifyRouteRequest(BaseModel):
    """Add or remove targets mid-route."""
    add: List[str] = Field(default_factory=list, description="Targets to add")
    remove: List[str] = Field(default_factory=list, description="Targets to remove")
