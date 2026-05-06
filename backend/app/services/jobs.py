from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.db.models import EvaluationResult, ProcessingJob, ReferenceDrawing, StudentSubmission
from app.db.session import SessionLocal
from app.services.evaluator import analyze_file, evaluate_submission
from sqlalchemy.orm import Session


def create_job(db: Session, reference_id: int) -> ProcessingJob:
    total = db.query(StudentSubmission).filter(StudentSubmission.reference_id == reference_id).count()
    job = ProcessingJob(id=uuid4().hex, reference_id=reference_id, total=total, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def run_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(ProcessingJob, job_id)
        if not job:
            return
        job.status = "processing"
        job.updated_at = datetime.utcnow()
        db.commit()

        reference = db.get(ReferenceDrawing, job.reference_id)
        if reference is None:
            raise RuntimeError("Reference drawing not found")
        if not reference.features:
            reference_features, reference_processed = analyze_file(reference.file_path)
            reference.features = reference_features
            db.commit()
        else:
            reference_features, reference_processed = analyze_file(reference.file_path)

        submissions = (
            db.query(StudentSubmission)
            .filter(StudentSubmission.reference_id == job.reference_id)
            .order_by(StudentSubmission.id.asc())
            .all()
        )
        job.total = len(submissions)
        db.commit()

        for submission in submissions:
            submission.status = "processing"
            db.commit()
            payload = evaluate_submission(reference_features, reference_processed, submission.file_path, submission.id)
            existing = db.query(EvaluationResult).filter(EvaluationResult.submission_id == submission.id).one_or_none()
            if existing:
                for key, value in payload.items():
                    setattr(existing, key, value)
            else:
                db.add(EvaluationResult(submission_id=submission.id, **payload))
            submission.status = "completed"
            job.processed += 1
            job.updated_at = datetime.utcnow()
            db.commit()

        job.status = "completed"
        job.updated_at = datetime.utcnow()
        db.commit()
    except Exception as exc:
        if "job" in locals() and job:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()
