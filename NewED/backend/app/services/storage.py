from __future__ import annotations

import re
import shutil
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from fastapi import UploadFile


def safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("_") or "upload"
    suffix = Path(name).suffix.lower()
    return f"{stem}_{uuid4().hex[:10]}{suffix}"


def save_upload(file: UploadFile, folder: str) -> Path:
    target_dir = settings.upload_dir / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / safe_name(file.filename or "upload.bin")
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return destination


def output_path(name: str) -> Path:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings.output_dir / name


def output_url(path: str | None) -> str | None:
    if not path:
        return None
    return f"/outputs/{Path(path).name}"
