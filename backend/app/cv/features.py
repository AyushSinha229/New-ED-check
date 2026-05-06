from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class LineFeature:
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    length: float
    angle: float
    midpoint: tuple[float, float]


def _normalize_angle(angle: float) -> float:
    while angle < 0:
        angle += 180
    while angle >= 180:
        angle -= 180
    return angle


def _line_angle(x1: int, y1: int, x2: int, y2: int) -> float:
    return _normalize_angle(math.degrees(math.atan2(y2 - y1, x2 - x1)))


def _merge_similar_lines(lines: list[LineFeature], distance_threshold: float = 18.0, angle_threshold: float = 3.0) -> list[LineFeature]:
    merged: list[LineFeature] = []
    for line in sorted(lines, key=lambda item: item.length, reverse=True):
        duplicate = False
        for existing in merged:
            angle_delta = min(abs(line.angle - existing.angle), 180 - abs(line.angle - existing.angle))
            midpoint_delta = np.linalg.norm(np.array(line.midpoint) - np.array(existing.midpoint))
            if angle_delta <= angle_threshold and midpoint_delta <= distance_threshold:
                duplicate = True
                break
        if not duplicate:
            line.id = f"L{len(merged) + 1}"
            merged.append(line)
    return merged


def extract_lines(edges: np.ndarray) -> list[dict]:
    min_len = max(30, int(min(edges.shape[:2]) * 0.06))
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(40, int(min(edges.shape[:2]) * 0.045)),
        minLineLength=min_len,
        maxLineGap=max(8, int(min(edges.shape[:2]) * 0.018)),
    )
    features: list[LineFeature] = []
    if raw is not None:
        for item in raw[:, 0]:
            x1, y1, x2, y2 = [int(v) for v in item]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_len:
                continue
            angle = _line_angle(x1, y1, x2, y2)
            features.append(
                LineFeature(
                    id="",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    length=length,
                    angle=angle,
                    midpoint=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                )
            )
    return [asdict(line) for line in _merge_similar_lines(features)[:80]]


def _intersection(a: dict, b: dict) -> tuple[float, float] | None:
    x1, y1, x2, y2 = a["x1"], a["y1"], a["x2"], a["y2"]
    x3, y3, x4, y4 = b["x1"], b["y1"], b["x2"], b["y2"]
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    margin = 12
    if (
        min(x1, x2) - margin <= px <= max(x1, x2) + margin
        and min(y1, y2) - margin <= py <= max(y1, y2) + margin
        and min(x3, x4) - margin <= px <= max(x3, x4) + margin
        and min(y3, y4) - margin <= py <= max(y3, y4) + margin
    ):
        return float(px), float(py)
    return None


def extract_angles_and_intersections(lines: list[dict]) -> tuple[list[dict], list[dict]]:
    angles: list[dict] = []
    intersections: list[dict] = []
    for i, line_a in enumerate(lines):
        for line_b in lines[i + 1 :]:
            point = _intersection(line_a, line_b)
            if point is None:
                continue
            delta = abs(line_a["angle"] - line_b["angle"])
            angle = min(delta, 180 - delta)
            if angle < 8:
                continue
            intersections.append(
                {"lines": [line_a["id"], line_b["id"]], "point": [round(point[0], 2), round(point[1], 2)]}
            )
            angles.append(
                {
                    "id": f"A{len(angles) + 1}",
                    "lines": [line_a["id"], line_b["id"]],
                    "value": round(float(angle), 2),
                    "vertex": [round(point[0], 2), round(point[1], 2)],
                }
            )
    return angles[:120], intersections[:120]


def _circle_edge_support(edges: np.ndarray, x: int, y: int, r: int) -> float:
    samples = 96
    hits = 0
    h, w = edges.shape[:2]
    for theta in np.linspace(0, 2 * math.pi, samples, endpoint=False):
        px = int(round(x + math.cos(theta) * r))
        py = int(round(y + math.sin(theta) * r))
        if 0 <= px < w and 0 <= py < h:
            patch = edges[max(0, py - 1) : min(h, py + 2), max(0, px - 1) : min(w, px + 2)]
            if np.any(patch > 0):
                hits += 1
    return hits / samples


def _is_duplicate_shape(candidate: dict, shapes: list[dict]) -> bool:
    cx, cy = candidate["center"]
    cw, ch = candidate["bbox"][2], candidate["bbox"][3]
    for shape in shapes:
        sx, sy = shape["center"]
        sw, sh = shape["bbox"][2], shape["bbox"][3]
        center_delta = np.linalg.norm(np.array([cx, cy]) - np.array([sx, sy]))
        size_delta = abs(cw - sw) + abs(ch - sh)
        if shape["type"] == candidate["type"] and center_delta < 18 and size_delta < 35:
            return True
    return False


def extract_shapes(gray: np.ndarray, edges: np.ndarray) -> list[dict]:
    shapes: list[dict] = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        image_area = gray.shape[0] * gray.shape[1]
        if area < 250 or area > image_area * 0.35:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        if w < 12 or h < 12:
            continue
        label = None
        if len(approx) == 3:
            label = "triangle"
        elif len(approx) == 4:
            ratio = w / float(h)
            label = "square" if 0.85 <= ratio <= 1.15 else "rectangle"
        elif len(approx) >= 8:
            circularity = 4 * math.pi * area / (perimeter * perimeter + 1e-6)
            if circularity > 0.62:
                label = "circle"
        if label:
            candidate = {
                "id": f"S{len(shapes) + 1}",
                "type": label,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": round(float(area), 2),
                "center": [round(x + w / 2, 2), round(y + h / 2, 2)],
            }
            if not _is_duplicate_shape(candidate, shapes):
                shapes.append(candidate)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=90,
        param2=28,
        minRadius=max(18, min(gray.shape[:2]) // 28),
        maxRadius=max(18, min(gray.shape[:2]) // 8),
    )
    if circles is not None:
        for x, y, r in np.round(circles[0, :]).astype("int")[:20]:
            if _circle_edge_support(edges, int(x), int(y), int(r)) < 0.42:
                continue
            candidate = {
                "id": f"S{len(shapes) + 1}",
                "type": "circle",
                "bbox": [int(x - r), int(y - r), int(2 * r), int(2 * r)],
                "area": round(float(math.pi * r * r), 2),
                "center": [int(x), int(y)],
                "radius": int(r),
            }
            if not _is_duplicate_shape(candidate, shapes):
                shapes.append(candidate)
    return shapes[:80]


def extract_features(preprocessed: dict[str, np.ndarray]) -> dict:
    gray = preprocessed["gray"]
    edges = preprocessed["edges"]
    lines = extract_lines(edges)
    angles, intersections = extract_angles_and_intersections(lines)
    shapes = extract_shapes(gray, edges)
    h, w = gray.shape[:2]
    return {
        "image_size": {"width": int(w), "height": int(h)},
        "lines": lines,
        "angles": angles,
        "shapes": shapes,
        "intersections": intersections,
    }
