# api/config.py - Updated configuration settings for modular chess prediction services

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
    app_version: str = "2.0.0"  # Updated version for modular architecture
    app_description: str = (
        "Convert chess board images to FEN notation using computer vision and deep learning. "
        "Supports both end-to-end models and multi-model pipelines."
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

    # === CHESS PREDICTION SERVICE CONFIGURATION ===

    # Service Type Selection
    chess_service_type: str = Field(
        default="end_to_end",
        description="Type of chess prediction service: 'end_to_end' or 'multi_model_pipeline'"
    )

    # End-to-End Model Configuration
    end_to_end_model_path: str = Field(
        default="https://storage.googleapis.com/chess_board_cllassification_model/final_light_quick_20250905.keras"
    )

    # Multi-Model Pipeline Configuration
    segmentation_model_path: str = Field(
        default="https://storage.googleapis.com/chess_board_cllassification_model/yolo_chess_board_segmentation_sept_2025.pt",
        description="Path to YOLO segmentation model for board detection"
    )
    pieces_model_path: str = Field(
        default="https://storage.googleapis.com/chess_board_cllassification_model/yolo_chess_piece_detector_sept_2025.pt",
        description="Path to YOLO object detection model for piece detection"
    )

    # Model Input/Output Settings
    model_input_width: int = Field(default=256)
    model_input_height: int = Field(default=256)
    piece_classes: tuple[str, ...] = Field(default=DEFAULT_PIECE_CLASSES)

    # Service-Specific Confidence Thresholds
    # End-to-End Model Thresholds
    end_to_end_piece_confidence: float = Field(default=0.3, description="Confidence threshold for piece predictions")
    end_to_end_empty_confidence: float = Field(default=0.5,
                                               description="Confidence threshold for empty square predictions")

    # Multi-Model Pipeline Thresholds
    segmentation_confidence: float = Field(default=0.1, description="Confidence threshold for board segmentation")
    piece_detection_confidence: float = Field(default=0.05, description="Confidence threshold for piece detection")
    iou_threshold: float = Field(default=0.2, description="IoU threshold for non-maximum suppression")

    # Model caching settings
    model_cache_dir: str = Field(default="./model_cache")
    model_cache_enabled: bool = Field(default=True)
    model_download_timeout: int = Field(default=300)  # 5 minutes

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

    # === SERVICE CONFIGURATION METHODS ===

    def get_service_config(self) -> dict:
        """Get configuration for the selected chess prediction service"""
        base_config = {
            'piece_classes': self.piece_classes,
            'model_input_size': self.model_input_size
        }

        if self.chess_service_type == "end_to_end":
            return {
                **base_config,
                'model_path': self.end_to_end_model_path,
                'piece_confidence_threshold': self.end_to_end_piece_confidence,
                'empty_confidence_threshold': self.end_to_end_empty_confidence
            }
        elif self.chess_service_type == "multi_model_pipeline":
            return {
                **base_config,
                'segmentation_model_path': self.segmentation_model_path,
                'pieces_model_path': self.pieces_model_path,
                'segmentation_confidence': self.segmentation_confidence,
                'piece_confidence': self.piece_detection_confidence,
                'iou_threshold': self.iou_threshold
            }
        else:
            raise ValueError(f"Unknown service type: {self.chess_service_type}")

    # === LEGACY PROPERTIES FOR BACKWARD COMPATIBILITY ===

    @property
    def model_path(self) -> str:
        """Legacy property - returns end_to_end_model_path for backward compatibility"""
        return self.end_to_end_model_path

    @property
    def absolute_model_path(self) -> Path:
        """Legacy property for backward compatibility"""
        if self.is_model_url:
            return self.model_cache_path

        model_path = Path(self.end_to_end_model_path)
        if model_path.is_absolute():
            return model_path
        return Path(__file__).parent.parent / model_path

    @property
    def is_model_url(self) -> bool:
        """Check if end_to_end_model_path is a URL"""
        return self.end_to_end_model_path.startswith(('http://', 'https://'))

    @property
    def model_cache_path(self) -> Path:
        """Path where the downloaded model will be cached"""
        cache_dir = Path(self.model_cache_dir)
        if self.is_model_url:
            # Generate filename from URL
            import hashlib
            url_hash = hashlib.md5(self.end_to_end_model_path.encode()).hexdigest()[:8]
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