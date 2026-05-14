from __future__ import annotations

import csv
from pathlib import Path

from app.db.models import EvaluationResult, StudentSubmission
from openpyxl import Workbook
from sqlalchemy.orm import Session


def result_rows(db: Session) -> list[dict]:
    rows = (
        db.query(EvaluationResult, StudentSubmission)
        .join(StudentSubmission, EvaluationResult.submission_id == StudentSubmission.id)
        .order_by(StudentSubmission.student_id.asc())
        .all()
    )
    return [
        {
            "Student_ID": submission.student_id,
            "File": submission.filename,
            "Marks": result.score,
            "Feedback": result.feedback,
            "Errors": "; ".join(result.errors),
        }
        for result, submission in rows
    ]


def write_csv(rows: list[dict], destination: Path) -> Path:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Student_ID", "File", "Marks", "Feedback", "Errors"])
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_excel(rows: list[dict], destination: Path) -> Path:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Results"
    headers = ["Student_ID", "File", "Marks", "Feedback", "Errors"]
    sheet.append(headers)
    for row in rows:
        sheet.append([row[key] for key in headers])
    for column in sheet.columns:
        max_len = max(len(str(cell.value or "")) for cell in column)
        sheet.column_dimensions[column[0].column_letter].width = min(max_len + 3, 80)
    workbook.save(destination)
    return destination
