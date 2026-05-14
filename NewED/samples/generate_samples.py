from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "samples" / "drawings"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def canvas() -> np.ndarray:
    image = np.full((900, 700, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 35), (665, 865), (20, 20, 20), 3)
    cv2.putText(image, "ENGINEERING DRAWING SHEET", (62, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
    return image


def draw_reference(image: np.ndarray, offset: tuple[int, int] = (0, 0), scale: float = 1.0, missing: bool = False, extra: bool = False) -> np.ndarray:
    ox, oy = offset
    def p(x: int, y: int) -> tuple[int, int]:
        return int(ox + x * scale), int(oy + y * scale)

    cv2.line(image, p(155, 710), p(540, 710), (0, 0, 0), 4)
    cv2.line(image, p(155, 710), p(155, 345), (0, 0, 0), 4)
    cv2.line(image, p(155, 345), p(350, 215), (0, 0, 0), 4)
    cv2.line(image, p(350, 215), p(540, 345), (0, 0, 0), 4)
    cv2.line(image, p(540, 345), p(540, 710), (0, 0, 0), 4)
    cv2.line(image, p(155, 345), p(540, 345), (0, 0, 0), 3)
    cv2.line(image, p(350, 215), p(350, 710), (0, 0, 0), 3)
    if not missing:
        cv2.circle(image, p(465, 455), int(48 * scale), (0, 0, 0), 3)
    cv2.rectangle(image, p(225, 555), p(310, 640), (0, 0, 0), 3)
    if extra:
        cv2.line(image, p(175, 780), p(535, 260), (0, 0, 0), 2)
        cv2.line(image, p(200, 490), p(500, 620), (0, 0, 0), 2)
    return image


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def save(name: str, image: np.ndarray) -> None:
    cv2.imwrite(str(SAMPLE_DIR / name), image)


def main() -> None:
    save("reference_orthographic.png", draw_reference(canvas()))
    save("student_001_good.png", draw_reference(canvas(), offset=(4, 2), scale=1.0))
    save("student_002_tilted.png", rotate(draw_reference(canvas(), offset=(-8, 3), scale=1.0), 5.5))
    save("student_003_incomplete.png", draw_reference(canvas(), missing=True))
    save("student_004_extra_lines.png", draw_reference(canvas(), extra=True))
    distorted = draw_reference(canvas(), offset=(20, -5), scale=0.93)
    pts1 = np.float32([[0, 0], [700, 0], [0, 900], [700, 900]])
    pts2 = np.float32([[18, 28], [670, 0], [0, 885], [690, 860]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    save("student_005_perspective.png", cv2.warpPerspective(distorted, matrix, (700, 900), borderValue=(255, 255, 255)))
    print(f"Generated samples in {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
