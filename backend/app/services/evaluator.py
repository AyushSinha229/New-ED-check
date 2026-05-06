from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.cv.features import extract_features
from app.cv.io import first_page
from app.cv.preprocessing import preprocess
from app.services.comparison import compare_drawings
from app.services.feedback import draw_heatmap, draw_overlay, textual_feedback
from app.services.storage import output_path


def analyze_file(path: str | Path) -> tuple[dict, dict]:
    image = first_page(Path(path))
    processed = preprocess(image)
    features = extract_features(processed)
    return features, processed


def evaluate_submission(reference_features: dict, reference_processed: dict, student_path: str | Path, submission_id: int) -> dict:
    student_features, student_processed = analyze_file(student_path)
    comparison = compare_drawings(reference_processed, student_processed)
    overlay = output_path(f"overlay_submission_{submission_id}.png")
    heatmap = output_path(f"heatmap_submission_{submission_id}.png")
    draw_overlay(student_processed["original"], student_features, comparison["errors"], overlay)
    draw_heatmap(reference_processed["edges"], student_processed["edges"], heatmap)
    feedback = textual_feedback(comparison["score"], comparison["errors"], comparison["metrics"])
    return {
        "score": comparison["score"],
        "errors": comparison["errors"],
        "metrics": comparison["metrics"],
        "features": student_features,
        "feedback": feedback,
        "overlay_path": str(overlay),
        "heatmap_path": str(heatmap),
    }
