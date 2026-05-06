from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_pages(path: Path) -> list[np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = convert_from_path(str(path), dpi=220)
        return [cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) for page in pages]
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        image = Image.open(path).convert("RGB")
        return [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)]
    raise ValueError(f"Unsupported file type: {suffix}")


def first_page(path: Path) -> np.ndarray:
    pages = load_pages(path)
    if not pages:
        raise ValueError("No pages found in uploaded file")
    return pages[0]
