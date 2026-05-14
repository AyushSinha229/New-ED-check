from __future__ import annotations

import math
from collections import Counter
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize

COMPARE_SIZE = (900, 900)
WEIGHTS = {"overlap": 0.40, "edge_ssim": 0.25, "shape": 0.20, "features": 0.15}


def _line_mask(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        10,
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    image_area = binary.shape[0] * binary.shape[1]
    for idx in range(1, components):
        area = stats[idx, cv2.CC_STAT_AREA]
        width = stats[idx, cv2.CC_STAT_WIDTH]
        height = stats[idx, cv2.CC_STAT_HEIGHT]
        if 12 <= area <= image_area * 0.2 and max(width, height) >= 8:
            cleaned[labels == idx] = 255
    return cleaned


def _content_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        return 0, 0, mask.shape[1], mask.shape[0]
    x, y, w, h = cv2.boundingRect(points)
    mx = max(8, int(w * 0.05))
    my = max(8, int(h * 0.05))
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(mask.shape[1], x + w + mx)
    y2 = min(mask.shape[0], y + h + my)
    if x2 - x1 < 30 or y2 - y1 < 30:
        return 0, 0, mask.shape[1], mask.shape[0]
    return x1, y1, x2 - x1, y2 - y1


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return image[y : y + h, x : x + w]


def _skeleton(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(mask, COMPARE_SIZE, interpolation=cv2.INTER_NEAREST)
    resized = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    skeleton = skeletonize(resized > 0)
    return (skeleton.astype(np.uint8) * 255)


def _prepare_structure(gray: np.ndarray) -> dict[str, np.ndarray]:
    mask = _line_mask(gray)
    bbox = _content_bbox(mask)
    cropped_mask = _crop(mask, bbox)
    resized_mask = cv2.resize(cropped_mask, COMPARE_SIZE, interpolation=cv2.INTER_NEAREST)
    edge = cv2.Canny(resized_mask, 40, 120)
    skeleton = _skeleton(cropped_mask)
    structure = cv2.bitwise_or(edge, skeleton)
    return {"mask": resized_mask, "edge": structure, "skeleton": skeleton}


def _overlap_score(ref_edge: np.ndarray, stu_edge: np.ndarray) -> float:
    kernel = np.ones((25, 25), np.uint8)
    ref = ref_edge > 0
    stu = stu_edge > 0
    ref_dilated = cv2.dilate(ref_edge, kernel, iterations=1) > 0
    stu_dilated = cv2.dilate(stu_edge, kernel, iterations=1) > 0
    ref_hits = np.logical_and(ref, stu_dilated).sum() / max(1, ref.sum())
    stu_hits = np.logical_and(stu, ref_dilated).sum() / max(1, stu.sum())
    return float(max(0.0, min(1.0, (ref_hits + stu_hits) / 2.0)))


def _edge_ssim(ref_edge: np.ndarray, stu_edge: np.ndarray) -> float:
    return float(max(0.0, min(1.0, structural_similarity(ref_edge, stu_edge, data_range=255))))


def _detect_shapes(mask: np.ndarray) -> list[dict[str, Any]]:
    work = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = mask.shape[0] * mask.shape[1]
    shapes: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.001 or area > image_area * 0.90:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 35 or h < 35:
            continue
        approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter + 1e-6)
        shape_type = ""
        if len(approx) == 3:
            shape_type = "triangle"
        elif len(approx) == 4:
            ratio = w / float(h)
            shape_type = "square" if 0.82 <= ratio <= 1.18 else "rectangle"
        elif len(approx) >= 7 and circularity >= 0.55:
            shape_type = "circle"
        if not shape_type:
            continue
        shapes.append(
            {
                "type": shape_type,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area_ratio": round(float(area / image_area), 5),
                "aspect_ratio": round(float(w / h), 4),
                "vertices": int(len(approx)),
                "circularity": round(float(circularity), 4),
            }
        )
    return sorted(shapes, key=lambda item: item["area_ratio"], reverse=True)[:30]


def _major_contour_count(mask: np.ndarray) -> int:
    work = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = mask.shape[0] * mask.shape[1]
    return sum(1 for contour in contours if image_area * 0.001 <= cv2.contourArea(contour) <= image_area * 0.90)


def _shape_score(ref_mask: np.ndarray, stu_mask: np.ndarray) -> tuple[float, dict[str, Any], bool, list[str]]:
    ref_shapes = _detect_shapes(ref_mask)
    stu_shapes = _detect_shapes(stu_mask)
    ref_counts = Counter(shape["type"] for shape in ref_shapes)
    stu_counts = Counter(shape["type"] for shape in stu_shapes)
    all_types = sorted(set(ref_counts) | set(stu_counts))
    mismatch_count = sum(abs(ref_counts.get(item, 0) - stu_counts.get(item, 0)) for item in all_types)
    expected = max(1, sum(ref_counts.values()))
    shape_mismatch = mismatch_count > 0
    if not ref_shapes and not stu_shapes:
        score = 1.0
    elif not ref_shapes or not stu_shapes:
        score = 0.0
        shape_mismatch = True
    else:
        count_score = max(0.0, 1.0 - mismatch_count / expected)
        ratio_errors = []
        used: set[int] = set()
        for ref_shape in ref_shapes:
            best_index = None
            best_error = float("inf")
            for idx, stu_shape in enumerate(stu_shapes):
                if idx in used or stu_shape["type"] != ref_shape["type"]:
                    continue
                error = abs(ref_shape["area_ratio"] - stu_shape["area_ratio"]) * 5.0
                error += abs(ref_shape["aspect_ratio"] - stu_shape["aspect_ratio"]) * 0.25
                ref_x, ref_y, ref_w, ref_h = ref_shape["bbox"]
                stu_x, stu_y, stu_w, stu_h = stu_shape["bbox"]
                position_error = (
                    abs(ref_x - stu_x) / max(1, ref_mask.shape[1])
                    + abs(ref_y - stu_y) / max(1, ref_mask.shape[0])
                    + abs((ref_x + ref_w) - (stu_x + stu_w)) / max(1, ref_mask.shape[1])
                    + abs((ref_y + ref_h) - (stu_y + stu_h)) / max(1, ref_mask.shape[0])

                )
                error += position_error * 1.5
                if error < best_error:
                    best_error = error
                    best_index = idx
            if best_index is not None:
                used.add(best_index)
                ratio_errors.append(min(1.0, best_error))
            else:
                ratio_errors.append(1.0)
        geometry_score = 1.0 - (float(np.mean(ratio_errors)) if ratio_errors else 1.0)
        score = max(0.0, min(1.0, count_score * 0.65 + geometry_score * 0.35))
    errors = []
    for shape_type in all_types:
        expected_count = ref_counts.get(shape_type, 0)
        found_count = stu_counts.get(shape_type, 0)
        if expected_count != found_count:
            errors.append(f"Shape mismatch: expected {expected_count} {shape_type}, found {found_count}")
    return score, {
        "reference_shapes": ref_shapes,
        "student_shapes": stu_shapes,
        "reference_counts": dict(ref_counts),
        "student_counts": dict(stu_counts),
        "reference_major_contours": _major_contour_count(ref_mask),
        "student_major_contours": _major_contour_count(stu_mask),
        "shape_mismatch": shape_mismatch,
    }, shape_mismatch, errors


def _feature_score(ref_edge: np.ndarray, stu_edge: np.ndarray) -> tuple[float, dict[str, Any], list[str]]:
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=8, fastThreshold=5)
    ref_kp, ref_des = orb.detectAndCompute(ref_edge, None)
    stu_kp, stu_des = orb.detectAndCompute(stu_edge, None)
    if ref_des is None or stu_des is None or not ref_kp or not stu_kp:
        return 0.0, {
            "reference_keypoints": len(ref_kp or []),
            "student_keypoints": len(stu_kp or []),
            "good_matches": 0,
            "match_ratio": 0.0,
        }, ["No reliable edge feature matches were found"]
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ref_des, stu_des, k=2)
    good = []
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance:
            good.append(pair[0])
    denominator = max(1, min(len(ref_kp), len(stu_kp)))
    raw_ratio = len(good) / denominator
    ratio = raw_ratio * 3.5
    ratio = max(0.0, min(1.0, ratio))
    errors = []
    if ratio < 0.10:
        errors.append(f"Edge feature match ratio is low ({ratio:.2f})")
    return ratio, {
        "reference_keypoints": len(ref_kp),
        "student_keypoints": len(stu_kp),
        "good_matches": len(good),
        "match_ratio": round(ratio, 4),
    }, errors


def compare_drawings(reference_processed: dict[str, np.ndarray], student_processed: dict[str, np.ndarray]) -> dict[str, Any]:
    ref = _prepare_structure(reference_processed["gray"])
    stu = _prepare_structure(student_processed["gray"])
    overlap = _overlap_score(ref["edge"], stu["edge"])
    edge_ssim = _edge_ssim(ref["edge"], stu["edge"])
    shape_score, shape_metrics, shape_mismatch, shape_errors = _shape_score(ref["mask"], stu["mask"])
    feature_score, feature_metrics, feature_errors = _feature_score(ref["edge"], stu["edge"])

    score = (
    overlap * 40.0
    + edge_ssim * 25.0
    + shape_score * 20.0
    + feature_score * 15.0
    )
    hard_difference = (
    overlap < 0.15
    and shape_score < 0.30
    and feature_score < 0.10
   )

    errors: list[str] = []

    if overlap < 0.30:
        errors.append(f"Drawing line overlap is low ({overlap:.2f})")

    if edge_ssim < 0.45:
        errors.append(f"Edge-only SSIM is low ({edge_ssim:.2f})")

    errors.extend(shape_errors)
    errors.extend(feature_errors)

# Same / almost same image
    if (
        overlap > 0.97
        and edge_ssim > 0.97
        and shape_score > 0.97
        and feature_score > 0.90
    ):
        score = 100.0

    elif hard_difference:
        score = 0.0
        errors.insert(0, "Drawing structure does not match reference")

    else:
        # Soft penalties instead of destroying score

        if shape_mismatch:
            score *= 0.75

        if overlap < 0.35:
            score *= 0.70

        if feature_score < 0.12:
            score *= 0.75

    metrics = {
        "weights": WEIGHTS,
        "category_scores": {
            "overlap": round(overlap * 100.0, 2),
            "edge_ssim": round(edge_ssim * 100.0, 2),
            "shapes": round(shape_score * 100.0, 2),
            "features": round(feature_score * 100.0, 2),
        },
        "structure": {
            "overlap_score": round(overlap, 4),
            "edge_ssim": round(edge_ssim, 4),
            "rejected": hard_difference,
            "rejection_reasons": ["overlap < 0.2 with shape mismatch or weak features"] if hard_difference else [],
            "shape_mismatch": shape_mismatch,
            "reference_line_pixels": int(cv2.countNonZero(ref["edge"])),
            "student_line_pixels": int(cv2.countNonZero(stu["edge"])),
        },
        "shapes": shape_metrics,
        "feature_matching": feature_metrics,
    }
    final_score = round(max(0.0, min(100.0, score)), 2)
    return {
        "score": final_score,
        "percentage": round(final_score / 100.0, 4),
        "errors": errors[:30],
        "metrics": metrics,
    }


def compare_features(reference: dict, student: dict, angle_tolerance: float, length_tolerance: float) -> dict[str, Any]:
    raise RuntimeError("compare_features was replaced by structure-only compare_drawings")
