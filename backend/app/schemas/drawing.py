from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ReferenceResponse(BaseModel):
    id: int
    filename: str
    drawing_type: str
    created_at: datetime

    class Config:
        from_attributes = True


class SubmissionResponse(BaseModel):
    id: int
    reference_id: int
    student_id: str
    filename: str
    status: str

    class Config:
        from_attributes = True


class ProcessResponse(BaseModel):
    job_id: str
    status: str
    total: int


class JobResponse(BaseModel):
    id: str
    reference_id: int
    status: str
    processed: int
    total: int
    error: str | None = None

    class Config:
        from_attributes = True


class ResultResponse(BaseModel):
    id: int
    submission_id: int
    student_id: str
    filename: str
    score: float
    percentage: float
    max_marks: float
    feedback: str
    errors: list[Any]
    metrics: dict[str, Any]
    overlay_url: str | None = None
    heatmap_url: str | None = None
    created_at: datetime
