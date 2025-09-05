# api/config.py - Configuration settings

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Immutable defaults (tuples) to avoid "mutable default" IDE warnings
DEFAULT_CORS_ORIGINS = ("*",)
DEFAULT_CORS_METHODS = ("*",)
DEFAULT_CORS_HEADERS = ("*",)
DEFAULT_PIECE_CLASSES = (
    "empty",
    "black-bishop", "black-king", "black-knight", "black-pawn",
    "black-queen", "black-rook", "white-bishop", "white-king",
    "white-knight", "white-pawn", "white-queen", "white-rook",
)
DEFAULT_IMAGE_FORMATS = ("jpg", "jpeg", "png", "bmp")


class Settings(BaseSettings):
    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Settings
    app_name: str = "Chess Board to FEN API"
    app_version: str = "1.0.0"
    app_description: str = (
        "Convert chess board images to FEN notation using computer vision and deep learning"
    )

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    debug: bool = Field(default=False)
    reload: bool = Field(default=False)

    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    # CORS (use tuples for immutability; convert to list at usage sites if required)
    cors_origins: tuple[str, ...] = Field(default=DEFAULT_CORS_ORIGINS)
    cors_allow_credentials: bool = True
    cors_allow_methods: tuple[str, ...] = Field(default=DEFAULT_CORS_METHODS)
    cors_allow_headers: tuple[str, ...] = Field(default=DEFAULT_CORS_HEADERS)

    # Database
    database_url: str = Field(default="sqlite:///./chess_predictions.db")
    sqlite_check_same_thread: bool = False
    db_pool_size: int = Field(default=5)
    db_max_overflow: int = Field(default=10)

    # Model - Updated to support URLs
    # model_path: str = Field(
    #     default="https://storage.googleapis.com/chess_board_cllassification_model/final_light_quick_20250903.keras")
    model_path: str = Field(
        default="https://storage.googleapis.com/chess_board_cllassification_model/final_light_quick_20250905.keras")

    model_input_width: int = Field(default=256)
    model_input_height: int = Field(default=256)

    # Model caching settings
    model_cache_dir: str = Field(default="./model_cache")
    model_cache_enabled: bool = Field(default=True)
    model_download_timeout: int = Field(default=300)  # 5 minutes

    piece_classes: tuple[str, ...] = Field(default=DEFAULT_PIECE_CLASSES)

    # Image Processing
    max_image_size_mb: float = Field(default=10.0)
    supported_image_formats: tuple[str, ...] = Field(default=DEFAULT_IMAGE_FORMATS)
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
    def model_input_size(self) -> tuple[int, int]:
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
        """For backward compatibility - returns path for local files"""
        if self.is_model_url:
            # Return cached model path
            return self.model_cache_path

        model_path = Path(self.model_path)
        if model_path.is_absolute():
            return model_path
        return Path(__file__).parent.parent / model_path

    @property
    def is_model_url(self) -> bool:
        """Check if model_path is a URL"""
        return self.model_path.startswith(('http://', 'https://'))

    @property
    def model_cache_path(self) -> Path:
        """Path where the downloaded model will be cached"""
        cache_dir = Path(self.model_cache_dir)
        if self.is_model_url:
            # Generate filename from URL
            import hashlib
            url_hash = hashlib.md5(self.model_path.encode()).hexdigest()[:8]
            model_filename = f"cached_model_{url_hash}.keras"
            return cache_dir / model_filename
        return cache_dir / "local_model.keras"


class DevelopmentSettings(Settings):
    debug: bool = True
    reload: bool = True
    log_level: str = "DEBUG"


class ProductionSettings(Settings):
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"
    cors_origins: tuple[str, ...] = Field(default=())  # lock down in prod
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