from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./drawing_eval.db"
    storage_dir: Path = Path("../storage")
    upload_dir: Path = Path("../storage/uploads")
    output_dir: Path = Path("../storage/outputs")
    max_upload_mb: int = 50
    angle_tolerance_deg: float = 3.0
    length_tolerance_ratio: float = 0.08
    cors_origins: list[str] = ["*"];

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def ensure_dirs(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
