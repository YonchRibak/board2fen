# api/schemas.py - Pydantic Request/Response Models

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# ============================================================================
# PREDICTION SCHEMAS
# ============================================================================

class PredictionResponse(BaseModel):
    """Response model for chess board predictions"""
    success: bool = Field(..., description="Whether prediction was successful")
    prediction_id: Optional[int] = Field(None, description="Database ID of the prediction")
    fen: Optional[str] = Field(None, description="Predicted FEN notation")
    board_matrix: Optional[List[List[str]]] = Field(None, description="8x8 board matrix representation")
    confidence_score: Optional[float] = Field(None, description="Prediction confidence (0-1)")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    board_detected: Optional[bool] = Field(None, description="Whether chess board was detected")
    message: Optional[str] = Field(None, description="Additional information or error message")

class PredictionDetailResponse(BaseModel):
    """Detailed response for specific prediction retrieval"""
    id: int
    predicted_fen: Optional[str]
    predicted_matrix: Optional[List[List[str]]]
    corrected_fen: Optional[str]
    corrected_matrix: Optional[List[List[str]]]
    device_identifier: str
    confidence_score: Optional[float]
    processing_time_ms: Optional[int]
    processing_time_seconds: Optional[float]
    created_at: Optional[str]
    corrected_at: Optional[str]
    board_detected: Optional[bool]
    prediction_successful: Optional[bool]
    has_correction: bool
    is_successful: bool

# ============================================================================
# CORRECTION SCHEMAS
# ============================================================================

class CorrectionRequest(BaseModel):
    """Request model for submitting corrections"""
    prediction_id: int = Field(..., description="ID of the prediction to correct")
    corrected_fen: str = Field(..., description="Manually corrected FEN notation")

class CorrectionResponse(BaseModel):
    """Response model for corrections"""
    success: bool
    message: str
    prediction_id: int
    corrected_fen: str

class RecentCorrection(BaseModel):
    """Model for recent correction data"""
    id: int
    predicted_fen: Optional[str]
    corrected_fen: Optional[str]
    confidence_score: Optional[float]
    device_identifier: Optional[str]  # Partially anonymized
    created_at: Optional[str]
    corrected_at: Optional[str]

class RecentCorrectionsResponse(BaseModel):
    """Response model for recent corrections"""
    count: int
    corrections: List[RecentCorrection]

# ============================================================================
# STATISTICS SCHEMAS
# ============================================================================

class StatsResponse(BaseModel):
    """Response model for API statistics"""
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    corrections_submitted: int
    average_processing_time_ms: float
    average_confidence: float

class DetailedStatsResponse(BaseModel):
    """Extended statistics response"""
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    corrections_submitted: int
    board_detection_success_rate: float
    prediction_success_rate: float
    correction_rate: float
    average_processing_time_ms: float
    average_confidence_score: float
    unique_devices: int

# ============================================================================
# MODEL STATUS SCHEMAS
# ============================================================================

class ModelStatusResponse(BaseModel):
    """Response model for model status"""
    model_version: str
    model_loaded: bool
    total_predictions: int
    accuracy_estimate: Optional[float]
    last_retrain: Optional[str]

class RetrainingStatusResponse(BaseModel):
    """Response model for retraining status"""
    total_corrections: int
    corrections_since_last_model: int
    retrain_threshold: int
    needs_retraining: bool
    corrections_until_retrain: int
    last_model_version: Optional[str]

# ============================================================================
# SYSTEM SCHEMAS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_ready: bool
    database_ready: bool
    timestamp: str

class ConfigInfoResponse(BaseModel):
    """Configuration information response (non-sensitive)"""
    environment: str
    app_version: str
    model_input_size: tuple
    max_image_size_mb: float
    supported_formats: List[str]
    retrain_threshold: int
    retrain_enabled: bool
    database_type: str
    debug_mode: bool
    cors_origins: List[str]
    rate_limiting_enabled: bool

# ============================================================================
# VALIDATION SCHEMAS
# ============================================================================

class FENValidationRequest(BaseModel):
    """Request model for FEN validation"""
    fen: str = Field(..., description="FEN notation string to validate")

class FENValidationResponse(BaseModel):
    """Response model for FEN validation"""
    valid: bool
    fen: Optional[str] = None
    board_matrix: Optional[List[List[str]]] = None
    error: Optional[str] = None

# ============================================================================
# UTILITY SCHEMAS
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response model"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    status_code: Optional[int] = None

class SuccessResponse(BaseModel):
    """Standard success response model"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None

# ============================================================================
# FILE UPLOAD SCHEMAS
# ============================================================================

class ImageValidationResponse(BaseModel):
    """Response model for image validation"""
    valid: bool
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None
    error: Optional[str] = None

# ============================================================================
# API INFO SCHEMAS
# ============================================================================

class RootResponse(BaseModel):
    """Root endpoint response"""
    message: str
    version: str
    docs: str
    health: str
    environment: str

class APIInfo(BaseModel):
    """API information model"""
    name: str
    version: str
    description: str
    docs_url: str
    health_url: str
    environment: str
    uptime: Optional[str] = None