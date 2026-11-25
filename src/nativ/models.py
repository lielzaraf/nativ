"""Shared data types for resident requests and their analysis."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


IncidentType = Literal["lighting", "sanitation", "roads", "water", "parks", "other"]
Urgency = Literal["low", "medium", "high"]
MissingField = Literal["address", "incident_type", "details"]


@dataclass(frozen=True)
class Request:
    """A resident's submitted report.

    Attributes:
        text: The free-form text submitted by the resident.
        received_at: The time the request was received, preferably timezone-aware.
    """

    text: str
    received_at: datetime


class RequestAnalysis(BaseModel):
    """Structured summary produced from a resident request."""

    address: str | None
    incident_type: IncidentType | None
    summary: str | None
    urgency: Urgency | None
    urgency_reason: str | None
    missing_fields: list[MissingField]
