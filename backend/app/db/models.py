from __future__ import annotations

from datetime import datetime

from app.db.session import Base
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class ReferenceDrawing(Base):
    __tablename__ = "reference_drawings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    drawing_type: Mapped[str] = mapped_column(String(64), default="orthographic")
    file_path: Mapped[str] = mapped_column(Text)
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    submissions: Mapped[list["StudentSubmission"]] = relationship(back_populates="reference")


class StudentSubmission(Base):
    __tablename__ = "student_submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    reference_id: Mapped[int] = mapped_column(ForeignKey("reference_drawings.id"))
    student_id: Mapped[str] = mapped_column(String(128), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    reference: Mapped[ReferenceDrawing] = relationship(back_populates="submissions")
    result: Mapped["EvaluationResult"] = relationship(back_populates="submission", uselist=False)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    reference_id: Mapped[int] = mapped_column(ForeignKey("reference_drawings.id"))
    status: Mapped[str] = mapped_column(String(32), default="queued")
    processed: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("student_submissions.id"), unique=True)
    score: Mapped[float] = mapped_column(Float)
    max_marks: Mapped[float] = mapped_column(Float, default=100)
    feedback: Mapped[str] = mapped_column(Text)
    errors: Mapped[list] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    features: Mapped[dict] = mapped_column(JSON)
    overlay_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    heatmap_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    submission: Mapped[StudentSubmission] = relationship(back_populates="result")
