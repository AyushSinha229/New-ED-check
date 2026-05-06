from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.db.models import EvaluationResult, ProcessingJob, ReferenceDrawing, StudentSubmission
from app.db.session import get_db
from app.schemas.drawing import JobResponse, ProcessResponse, ReferenceResponse, ResultResponse, SubmissionResponse
from app.services.evaluator import analyze_file
from app.services.exporter import result_rows, write_csv, write_excel
from app.services.jobs import create_job, run_job
from app.services.storage import output_path, output_url, save_upload
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/upload-reference", response_model=ReferenceResponse)
def upload_reference(
    file: UploadFile = File(...),
    drawing_type: str = Form("orthographic"),
    db: Session = Depends(get_db),
):
    path = save_upload(file, "references")
    try:
        features, _ = analyze_file(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not analyze reference drawing: {exc}") from exc
    reference = ReferenceDrawing(
        filename=file.filename or Path(path).name,
        drawing_type=drawing_type,
        file_path=str(path),
        features=features,
    )
    db.add(reference)
    db.commit()
    db.refresh(reference)
    return reference


@router.post("/upload-students", response_model=list[SubmissionResponse])
def upload_students(
    reference_id: int = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    reference = db.get(ReferenceDrawing, reference_id)
    if reference is None:
        raise HTTPException(status_code=404, detail="Reference drawing not found")
    submissions = []
    for index, file in enumerate(files, start=1):
        path = save_upload(file, f"students/reference_{reference_id}")
        student_id = Path(file.filename or f"student_{index}").stem
        submission = StudentSubmission(
            reference_id=reference_id,
            student_id=student_id,
            filename=file.filename or Path(path).name,
            file_path=str(path),
            status="uploaded",
        )
        db.add(submission)
        submissions.append(submission)
    db.commit()
    for submission in submissions:
        db.refresh(submission)
    return submissions


@router.post("/process", response_model=ProcessResponse)
def process(reference_id: int = Form(...), background_tasks: BackgroundTasks = None, db: Session = Depends(get_db)):
    reference = db.get(ReferenceDrawing, reference_id)
    if reference is None:
        raise HTTPException(status_code=404, detail="Reference drawing not found")
    submission_count = db.query(StudentSubmission).filter(StudentSubmission.reference_id == reference_id).count()
    if submission_count == 0:
        raise HTTPException(status_code=400, detail="Upload at least one student drawing before processing")
    job = create_job(db, reference_id)
    if background_tasks is not None:
        background_tasks.add_task(run_job, job.id)
    return ProcessResponse(job_id=job.id, status=job.status, total=job.total)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/results", response_model=list[ResultResponse])
def results(reference_id: int | None = None, db: Session = Depends(get_db)):
    query = db.query(EvaluationResult, StudentSubmission).join(
        StudentSubmission, EvaluationResult.submission_id == StudentSubmission.id
    )
    if reference_id is not None:
        query = query.filter(StudentSubmission.reference_id == reference_id)
    rows = query.order_by(EvaluationResult.score.desc()).all()
    return [
        ResultResponse(
            id=result.id,
            submission_id=result.submission_id,
            student_id=submission.student_id,
            filename=submission.filename,
            score=result.score,
            percentage=round(result.score / 100.0, 4),
            max_marks=result.max_marks,
            feedback=result.feedback,
            errors=result.errors,
            metrics=result.metrics,
            overlay_url=output_url(result.overlay_path),
            heatmap_url=output_url(result.heatmap_path),
            created_at=result.created_at,
        )
        for result, submission in rows
    ]


@router.get("/results/{result_id}", response_model=ResultResponse)
def result_detail(result_id: int, db: Session = Depends(get_db)):
    row = (
        db.query(EvaluationResult, StudentSubmission)
        .join(StudentSubmission, EvaluationResult.submission_id == StudentSubmission.id)
        .filter(EvaluationResult.id == result_id)
        .one_or_none()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Result not found")
    result, submission = row
    return ResultResponse(
        id=result.id,
        submission_id=result.submission_id,
        student_id=submission.student_id,
        filename=submission.filename,
        score=result.score,
        percentage=round(result.score / 100.0, 4),
        max_marks=result.max_marks,
        feedback=result.feedback,
        errors=result.errors,
        metrics=result.metrics,
        overlay_url=output_url(result.overlay_path),
        heatmap_url=output_url(result.heatmap_path),
        created_at=result.created_at,
    )


@router.get("/export/csv")
def export_csv(db: Session = Depends(get_db)):
    destination = output_path("drawing_evaluation_results.csv")
    write_csv(result_rows(db), destination)
    return FileResponse(destination, media_type="text/csv", filename=destination.name)


@router.get("/export/excel")
def export_excel(db: Session = Depends(get_db)):
    destination = output_path("drawing_evaluation_results.xlsx")
    write_excel(result_rows(db), destination)
    return FileResponse(
        destination,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=destination.name,
    )


@router.get("/references", response_model=list[ReferenceResponse])
def references(db: Session = Depends(get_db)):
    return db.query(ReferenceDrawing).order_by(ReferenceDrawing.created_at.desc()).all()


@router.get("/submissions", response_model=list[SubmissionResponse])
def submissions(reference_id: int | None = None, db: Session = Depends(get_db)):
    query = db.query(StudentSubmission)
    if reference_id is not None:
        query = query.filter(StudentSubmission.reference_id == reference_id)
    return query.order_by(StudentSubmission.created_at.desc()).all()
