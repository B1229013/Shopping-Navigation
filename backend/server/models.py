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


class StartSessionResponse(BaseModel):
    session_id: str
    guidance: str
    action: Literal["TAKE_PHOTO"] = "TAKE_PHOTO"
    goal_objects: List[str]


class AnswerRequest(BaseModel):
    answer: str


class TurnResponse(BaseModel):
    action: VLMAction
    guidance: str
    question: Optional[str] = None
    node_id: int
    annotated_photo_url: Optional[str] = None


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
