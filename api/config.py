# api/config.py - Configuration settings

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # API Settings
    app_name: str = "Chess Board to FEN API"
    app_version: str = "1.0.0"
    app_description: str = "Convert chess board images to FEN notation using computer vision and deep learning"

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    debug: bool = Field(default=False)
    reload: bool = Field(default=False)

    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    # CORS
    cors_origins: List[str] = Field(default=["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Database
    database_url: str = Field(default="sqlite:///./chess_predictions.db")
    sqlite_check_same_thread: bool = False
    db_pool_size: int = Field(default=5)
    db_max_overflow: int = Field(default=10)

    # Model
    model_path: str = Field(default="outputs/piece_classifier/final_chess_piece_classifier.keras")
    model_input_width: int = Field(default=224)
    model_input_height: int = Field(default=224)

    piece_classes: List[str] = Field(default=[
        'black_bishop', 'black_king', 'black_knight', 'black_pawn',
        'black_queen', 'black_rook', 'white_bishop', 'white_king',
        'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ])

    # Image Processing
    max_image_size_mb: float = Field(default=10.0)
    supported_image_formats: List[str] = Field(default=['jpg', 'jpeg', 'png', 'bmp'])
    min_image_dimension: int = Field(default=100)
    image_storage_quality: int = Field(default=85)

    # Retraining
    retrain_correction_threshold: int = Field(default=1000)
    retrain_enabled: bool = Field(default=True)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Security
    rate_limit_enabled: bool = Field(default=False)
    rate_limit_requests_per_minute: int = Field(default=100)

    @property
    def model_input_size(self) -> tuple:
        return (self.model_input_width, self.model_input_height)

    @property
    def max_image_size_bytes(self) -> int:
        return int(self.max_image_size_mb * 1024 * 1024)

    @property
    def database_connect_args(self) -> dict:
        if "sqlite" in self.database_url.lower():
            return {"check_same_thread": self.sqlite_check_same_thread}
        return {}

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.database_url.lower()

    @property
    def absolute_model_path(self) -> Path:
        model_path = Path(self.model_path)
        if model_path.is_absolute():
            return model_path
        return Path(__file__).parent.parent / model_path

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DevelopmentSettings(Settings):
    debug: bool = True
    reload: bool = True
    log_level: str = "DEBUG"


class ProductionSettings(Settings):
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"
    cors_origins: List[str] = []
    rate_limit_enabled: bool = True


def get_settings() -> Settings:
    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment in ("development", "dev"):
        return DevelopmentSettings()
    elif environment in ("production", "prod"):
        return ProductionSettings()
    else:
        return Settings()


settings = get_settings()