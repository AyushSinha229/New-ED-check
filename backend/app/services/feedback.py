from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def textual_feedback(score: float, errors: list[str], metrics: dict) -> str:
    structure = metrics.get("structure", {})
    if structure.get("rejected"):
        reasons = "; ".join(structure.get("rejection_reasons", []))
        return f"Drawing structure does not match reference. Overlap: {structure.get('overlap_score', 0):.2f}. Edge SSIM: {structure.get('edge_ssim', 0):.2f}. Rejection reason: {reasons}"

    lines = []
    lines.append(f"Line overlap: {structure.get('overlap_score', 0):.2f}")
    lines.append(f"Edge-only SSIM: {structure.get('edge_ssim', 0):.2f}")

    feature_metrics = metrics.get("feature_matching", {})
    if feature_metrics:
        lines.append(
            f"Feature matches: {feature_metrics.get('good_matches', 0)} "
            f"of {min(feature_metrics.get('reference_keypoints', 0), feature_metrics.get('student_keypoints', 0))} "
            f"(ratio: {feature_metrics.get('match_ratio', 0):.2f})"
        )

    shape_metrics = metrics.get("shapes", {})
    ref_counts = shape_metrics.get("reference_counts", {})
    stu_counts = shape_metrics.get("student_counts", {})
    if ref_counts or stu_counts:
        lines.append(f"Detected shapes: expected {ref_counts}, found {stu_counts}")

    if errors:
        lines.extend(errors[:3])
    return ". ".join(lines)


def draw_overlay(image: np.ndarray, features: dict, errors: list[str], destination: Path) -> str:
    canvas = image.copy()
    for line in features.get("lines", []):
        cv2.line(canvas, (line["x1"], line["y1"]), (line["x2"], line["y2"]), (44, 160, 44), 2)
        cv2.circle(canvas, (int(line["midpoint"][0]), int(line["midpoint"][1])), 3, (255, 180, 0), -1)
    for shape in features.get("shapes", []):
        x, y, w, h = shape["bbox"]
        color = (80, 120, 255) if shape["type"] != "circle" else (255, 90, 120)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(canvas, shape["type"], (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    y = 28
    for error in errors[:5]:
        cv2.putText(canvas, error[:82], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 220), 2, cv2.LINE_AA)
        y += 25
    cv2.imwrite(str(destination), canvas)
    return str(destination)


def draw_heatmap(reference_edges: np.ndarray, student_edges: np.ndarray, destination: Path) -> str:
    h = min(reference_edges.shape[0], student_edges.shape[0])
    w = min(reference_edges.shape[1], student_edges.shape[1])
    ref = cv2.resize(reference_edges, (w, h))
    stu = cv2.resize(student_edges, (w, h))
    missing = cv2.subtract(ref, stu)
    extra = cv2.subtract(stu, ref)
    heat = np.zeros((h, w, 3), dtype=np.uint8)
    heat[:, :, 2] = missing
    heat[:, :, 0] = extra
    heat = cv2.GaussianBlur(heat, (9, 9), 0)
    base = cv2.cvtColor(stu, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(base, 0.55, heat, 0.95, 0)
    cv2.imwrite(str(destination), blended)
    return str(destination)
