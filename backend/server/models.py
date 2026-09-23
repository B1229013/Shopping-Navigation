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
    place_name: str = ""


class SubGoalInfo(BaseModel):
    name: str
    arrived: bool = False
    # True for auto-appended checkout/exit destinations (excluded from the
    # user-facing goal count; the client can render them separately).
    is_terminal: bool = False

class StartSessionResponse(BaseModel):
    session_id: str
    guidance: str
    action: Literal["TAKE_PHOTO"] = "TAKE_PHOTO"
    goal_objects: List[str]
    nav_mode: str = "explore"
    goal_photo_ids: List[int] = []
    place_name: str = ""
    sub_goals: List[SubGoalInfo] = []
    tsp_order: List[int] = []
    current_goal_idx: int = 0
    current_goal_name: str = ""
    total_goals: int = 1


class AnswerRequest(BaseModel):
    answer: str


class ConfirmArrivalRequest(BaseModel):
    kind: Literal["confirmed", "false_positive", "wrong_instance"]


class TurnResponse(BaseModel):
    action: VLMAction
    guidance: str
    question: Optional[str] = None
    node_id: int
    annotated_photo_url: Optional[str] = None
    localized_photo_id: Optional[int] = None
    localization_score: Optional[float] = None
    localization_candidates: List[dict] = []
    goal_photo_ids: List[int] = []
    planned_route: List[int] = []
    route_target_photo_id: Optional[int] = None
    route_distance: Optional[float] = None
    route_hops: Optional[int] = None
    route_guidance: Optional[str] = None
    route_waypoints: List[dict] = []
    sub_goals: List[SubGoalInfo] = []
    current_goal_idx: int = 0
    current_goal_name: str = ""
    total_goals: int = 1


class ConfirmLocationRequest(BaseModel):
    photo_id: int


class ConfirmLocationResponse(BaseModel):
    guidance: str
    localized_photo_id: int
    goal_photo_ids: List[int] = []
    route_guidance: Optional[str] = None
    route_waypoints: List[dict] = []
    route_distance: Optional[float] = None
    route_hops: Optional[int] = None
    sub_goals: List[SubGoalInfo] = []
    current_goal_idx: int = 0
    current_goal_name: str = ""
    total_goals: int = 1


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


class VLMDetectionItem(BaseModel):
    label: str
    bbox: List[float] = Field(..., description="[x1,y1,x2,y2] absolute pixels")
    score: float
    position: str = ""


class VLMOCRItem(BaseModel):
    text: str
    bbox: List[List[float]] = Field(..., description="[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]")
    score: float
    position: str = ""


class AisleSignItem(BaseModel):
    aisle_number: Optional[int] = None
    categories: List[str] = []
    position: str = ""

class VLMPerceptionResponse(BaseModel):
    detections: List[VLMDetectionItem] = []
    ocr_texts: List[VLMOCRItem] = []
    aisle_signs: List[AisleSignItem] = []
    scene_description: str = ""


class ErrorResponse(BaseModel):
    error: str
    detail: str
