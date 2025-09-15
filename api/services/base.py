# api/services/base.py - Abstract base class for chess prediction services

from abc import ABC, abstractmethod
from typing import Optional
import logging
from pathlib import Path

# Import existing components
from api._helpers import PredictionResult

logger = logging.getLogger(__name__)


class ChessPredictionService(ABC):
    """
    Abstract base class for chess prediction services.
    Supports both end-to-end models and multi-model pipelines.
    """

    def __init__(self, config: dict):
        """
        Initialize the chess prediction service

        Args:
            config: Configuration dictionary containing model paths and parameters
        """
        self.config = config
        self.service_loaded = False
        self.service_type = self.__class__.__name__

    @abstractmethod
    def _load_models(self) -> bool:
        """
        Load the required models for this service.

        Returns:
            bool: True if all models loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def predict_from_image(self, image_bytes: bytes) -> PredictionResult:
        """
        Main prediction method: convert image bytes to FEN notation

        Args:
            image_bytes: Raw image bytes

        Returns:
            PredictionResult: Structured prediction result
        """
        pass

    def is_ready(self) -> bool:
        """Check if the service is ready to make predictions"""
        return self.service_loaded

    def get_service_info(self) -> dict:
        """Get information about this service"""
        return {
            "service_type": self.service_type,
            "loaded": self.service_loaded,
            "config": {k: v for k, v in self.config.items() if not k.startswith('_')}
        }


class ServiceFactory:
    """Factory class for creating chess prediction services"""

    @staticmethod
    def create_service(service_type: str, config: dict) -> ChessPredictionService:
        """
        Create a chess prediction service based on type

        Args:
            service_type: Type of service ("end_to_end" or "multi_model_pipeline")
            config: Service configuration

        Returns:
            ChessPredictionService: Configured service instance

        Raises:
            ValueError: If service_type is not supported
        """
        if service_type == "end_to_end":
            from api.services.end_to_end import EndToEndPipelineService
            return EndToEndPipelineService(config)
        elif service_type == "multi_model_pipeline":
            from api.services.multi_model import MultiModelPipelineService
            return MultiModelPipelineService(config)
        else:
            raise ValueError(f"Unsupported service type: {service_type}")

    @staticmethod
    def get_available_services() -> list:
        """Get list of available service types"""
        return ["end_to_end", "multi_model_pipeline"]