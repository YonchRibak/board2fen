# api/main.py - FastAPI Chess Board to FEN Service

import os
import time
import logging
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
import uvicorn

# Add the parent directory (project root) to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "api" else current_dir
sys.path.insert(0, str(project_root))

# Import configuration
from api.config import settings

# Import database components
from api.database import get_db, check_database_connection, health_check as db_health_check, get_database_info

# Import our chess pipeline components
from api._helpers import (
    FENValidator,
    ChessPipelineService,
    validate_uploaded_image,
    resize_image_for_model
)
# Import database cnn_models
from api.models import (
    ChessPrediction,
    get_database_statistics,
    check_retraining_threshold,
    get_corrected_predictions,
    get_model_metrics,
    get_current_active_model
)
# Import Pydantic schemas
from api.schemas import (
    PredictionResponse,
    CorrectionRequest,
    CorrectionResponse,
    StatsResponse,
    ModelStatusResponse,
    HealthResponse,
    ConfigInfoResponse,
    FENValidationResponse,
    RootResponse,
    RecentCorrectionsResponse,
    RetrainingStatusResponse, ModelMetricsResponse, ModelMetrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Initialize chess pipeline service
chess_pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize the chess pipeline service on startup"""
    global chess_pipeline

    try:
        # Check database connection
        if not check_database_connection():
            raise Exception("Database connection failed")

        # Initialize chess pipeline with configured model path
        chess_pipeline = ChessPipelineService(model_path=str(settings.absolute_model_path))
        logger.info(f"♟️ Chess pipeline service initialized with model: {settings.absolute_model_path}")

        # Log configuration info
        logger.info(f"🔧 Environment: {os.getenv('ENVIRONMENT', 'development')}")
        logger.info(f"🗄️ Database: {settings.database_url}")
        logger.info(f"🖼️ Max image size: {settings.max_image_size_mb}MB")
        logger.info(f"🔄 Retrain threshold: {settings.retrain_correction_threshold} corrections")

    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise e


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_client_ip(request: Request) -> str:
    """Extract client IP address from request"""
    # Check for forwarded IP (if behind proxy)
    forwarded_ip = request.headers.get("X-Forwarded-For")
    if forwarded_ip:
        return forwarded_ip.split(",")[0].strip()

    # Check for real IP (if behind proxy)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct client IP
    return request.client.host


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=RootResponse)
async def root():
    """Root endpoint with API information"""
    return RootResponse(
        message=settings.app_name,
        version=settings.app_version,
        docs=settings.docs_url,
        health="/health",
        environment=os.getenv("ENVIRONMENT", "development")
    )


@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""

    # Use database health check function
    db_health = db_health_check()

    return HealthResponse(
        status="healthy" if (chess_pipeline and db_health["database_connected"]) else "unhealthy",
        model_ready=chess_pipeline is not None,
        database_ready=db_health["database_connected"],
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_chess_position(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="Chess board image file"),
        db: Session = Depends(get_db)
):
    """
    Main prediction endpoint: Upload chess board image and get FEN notation
    """

    if not chess_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Chess pipeline service not available"
        )

    start_time = time.time()
    device_ip = get_client_ip(request)

    try:
        # Validate uploaded file
        image_bytes = await file.read()
        validation_result = validate_uploaded_image(image_bytes, file.filename)

        if not validation_result["valid"]:
            return PredictionResponse(
                success=False,
                message=validation_result["error"]
            )

        # Resize image to model expectations and convert to bytes
        resized_image_bytes = resize_image_for_model(image_bytes)

        logger.info(f"🔍 Processing image: {file.filename} from IP: {device_ip}")

        # Process the image through our chess pipeline
        prediction_result = chess_pipeline.predict_from_image(image_bytes)

        processing_time = int((time.time() - start_time) * 1000)

        # Save prediction to database
        db_prediction = ChessPrediction(
            image=resized_image_bytes,
            predicted_fen=prediction_result.fen if prediction_result.success else None,
            device_identifier=device_ip,
            confidence_score=prediction_result.confidence,
            processing_time_ms=processing_time,
            board_detected=prediction_result.board_detected,
            prediction_successful=prediction_result.success
        )

        # Set predicted matrix using the model method
        if prediction_result.board_matrix:
            db_prediction.set_predicted_matrix(prediction_result.board_matrix)

        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        if prediction_result.success:
            logger.info(f"✅ Prediction successful in {processing_time}ms, saved as ID: {db_prediction.id}")

            return PredictionResponse(
                success=True,
                prediction_id=db_prediction.id,
                fen=prediction_result.fen,
                board_matrix=prediction_result.board_matrix,
                confidence_score=prediction_result.confidence,
                processing_time_ms=processing_time,
                board_detected=prediction_result.board_detected,
                message="Prediction completed successfully"
            )
        else:
            logger.warning(f"⚠️ Prediction failed: {prediction_result.error_message}")

            return PredictionResponse(
                success=False,
                prediction_id=db_prediction.id,
                processing_time_ms=processing_time,
                board_detected=prediction_result.board_detected,
                message=prediction_result.error_message or "Prediction failed"
            )

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"❌ Prediction error: {str(e)}")

        return PredictionResponse(
            success=False,
            processing_time_ms=processing_time,
            message=f"Internal error: {str(e)}"
        )


@app.post("/predict/correct", response_model=CorrectionResponse)
async def submit_correction(
        correction: CorrectionRequest,
        db: Session = Depends(get_db)
):
    """
    Submit a correction for a previous prediction
    """
    try:
        # Validate the corrected FEN
        if not FENValidator.validate_fen(correction.corrected_fen):
            raise HTTPException(
                status_code=400,
                detail="Invalid FEN notation provided"
            )

        # Find the prediction in database
        db_prediction = db.query(ChessPrediction).filter(
            ChessPrediction.id == correction.prediction_id
        ).first()

        if not db_prediction:
            raise HTTPException(
                status_code=404,
                detail="Prediction not found"
            )

        # Update with correction
        db_prediction.corrected_fen = correction.corrected_fen
        db_prediction.set_corrected_matrix(FENValidator.fen_to_board_matrix(correction.corrected_fen))
        db_prediction.corrected_at = datetime.utcnow()

        db.commit()

        logger.info(f"📝 Correction submitted for prediction ID: {correction.prediction_id}")

        return CorrectionResponse(
            success=True,
            message="Correction saved successfully",
            prediction_id=correction.prediction_id,
            corrected_fen=correction.corrected_fen
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error saving correction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save correction: {str(e)}"
        )


@app.get("/predict/{prediction_id}")
async def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Get details of a specific prediction"""

    db_prediction = db.query(ChessPrediction).filter(
        ChessPrediction.id == prediction_id
    ).first()

    if not db_prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return db_prediction.to_dict()


@app.get("/stats", response_model=StatsResponse)
async def get_api_stats(db: Session = Depends(get_db)):
    """Get API usage statistics"""

    try:
        stats = get_database_statistics(db)

        return StatsResponse(
            total_predictions=stats["total_predictions"],
            successful_predictions=stats["successful_predictions"],
            failed_predictions=stats["failed_predictions"],
            corrections_submitted=stats["corrections_submitted"],
            average_processing_time_ms=stats["average_processing_time_ms"],
            average_confidence=stats["average_confidence_score"]
        )

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@app.get("/corrections/recent", response_model=RecentCorrectionsResponse)
async def get_recent_corrections(limit: int = 10, db: Session = Depends(get_db)):
    """Get recent corrections for monitoring purposes"""

    try:
        recent_corrections = get_corrected_predictions(db, limit=limit)

        corrections_list = [
            {
                "id": pred.id,
                "predicted_fen": pred.predicted_fen,
                "corrected_fen": pred.corrected_fen,
                "confidence_score": pred.confidence_score,
                "device_identifier": pred.device_identifier[:8] + "..." if pred.device_identifier else None,
                # Partial IP for privacy
                "created_at": pred.created_at.isoformat() if pred.created_at else None,
                "corrected_at": pred.corrected_at.isoformat() if pred.corrected_at else None,
            }
            for pred in recent_corrections
        ]

        return RecentCorrectionsResponse(
            count=len(recent_corrections),
            corrections=corrections_list
        )

    except Exception as e:
        logger.error(f"Error getting recent corrections: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recent corrections")


@app.get("/corrections/count", response_model=RetrainingStatusResponse)
async def get_corrections_count(db: Session = Depends(get_db)):
    """Get count of corrections for model retraining threshold"""

    try:
        retrain_info = check_retraining_threshold(db, threshold=settings.retrain_correction_threshold)
        return RetrainingStatusResponse(**retrain_info)

    except Exception as e:
        logger.error(f"Error checking retraining threshold: {e}")
        raise HTTPException(status_code=500, detail="Failed to check retraining status")


@app.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status(db: Session = Depends(get_db)):
    """Get current model status and statistics"""

    try:
        # Get database statistics for accuracy estimation
        stats = get_database_statistics(db)

        return ModelStatusResponse(
            model_version=settings.app_version,  # TODO: Get from actual model metadata
            model_loaded=chess_pipeline is not None,
            total_predictions=stats["total_predictions"],
            accuracy_estimate=stats["prediction_success_rate"] if stats["total_predictions"] > 0 else None,
            last_retrain=None  # TODO: Get from ModelVersion table
        )

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model status")


@app.get("/model/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(db: Session = Depends(get_db)):
    """Get comprehensive metrics for all model versions/generations"""

    try:
        from models import ModelVersion, ChessPrediction
        from datetime import datetime

        # Get all model versions
        model_versions = db.query(ModelVersion).order_by(ModelVersion.created_at.asc()).all()

        if not model_versions:
            return ModelMetricsResponse(
                total_model_versions=0,
                current_active_version=None,
                metrics=[]
            )

        # Get current active model
        active_model = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        current_active_version = active_model.version_number if active_model else None

        # Simple metrics calculation inline
        metrics_list = []

        for model in model_versions:
            # Basic metrics for now
            total_predictions = db.query(ChessPrediction).count()
            successful_predictions = db.query(ChessPrediction).filter(
                ChessPrediction.prediction_successful == True
            ).count()

            model_metrics = ModelMetrics(
                version_id=model.id,
                version_number=model.version_number,
                created_at=model.created_at.isoformat() if model.created_at else "",
                is_active=model.is_active,
                training_data_count=model.training_data_count or 0,
                validation_accuracy=model.validation_accuracy,
                performance_metrics=model.get_performance_metrics(),
                total_predictions=total_predictions,
                successful_predictions=successful_predictions,
                failed_predictions=total_predictions - successful_predictions,
                success_rate=successful_predictions / total_predictions if total_predictions > 0 else 0.0,
                corrections_received=0,  # Simplified for now
                correction_rate=0.0,
                average_confidence=None,
                average_processing_time_ms=None,
                active_duration_days=0,
                predictions_per_day=None,
                notes=model.notes
            )

            metrics_list.append(model_metrics)

        return ModelMetricsResponse(
            total_model_versions=len(metrics_list),
            current_active_version=current_active_version,
            metrics=metrics_list
        )

    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model metrics: {str(e)}")


# Replace the debug endpoint in main.py with this corrected version

@app.get("/debug/model")
async def debug_model_loading():
    """Debug endpoint to check model loading status"""

    # Import everything we need in this function
    from pathlib import Path
    import os

    debug_info = {
        "pipeline_exists": chess_pipeline is not None,
        "model_loaded": chess_pipeline.model_loaded if chess_pipeline else False,
        "model_path": str(settings.absolute_model_path),
        "model_exists": settings.absolute_model_path.exists(),
        "model_size_mb": None,
        "errors": [],
        "tensorflow_version": None,
        "current_directory": str(Path.cwd()),
        "files_in_model_dir": [],
        "settings_model_path": settings.model_path,
        "python_path": os.environ.get("PYTHONPATH", "Not set")
    }

    # Check TensorFlow
    try:
        import tensorflow as tf
        debug_info["tensorflow_version"] = tf.__version__
    except Exception as e:
        debug_info["errors"].append(f"TensorFlow import error: {e}")

    # Check model file
    model_path = settings.absolute_model_path
    debug_info["model_path_absolute"] = str(model_path.absolute())

    if model_path.exists():
        debug_info["model_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2)
    else:
        debug_info["errors"].append(f"Model file not found: {model_path}")

        # List files in the directory
        parent_dir = model_path.parent
        if parent_dir.exists():
            debug_info["files_in_model_dir"] = [f.name for f in parent_dir.glob("*")]
            debug_info["parent_dir_exists"] = True
        else:
            debug_info["errors"].append(f"Model directory doesn't exist: {parent_dir}")
            debug_info["parent_dir_exists"] = False

            # Try to find any .keras files
            debug_info["keras_files_found"] = []
            try:
                for keras_file in Path("/app").rglob("*.keras"):
                    debug_info["keras_files_found"].append(str(keras_file))
            except:
                pass

    # Try to manually load the model
    try:
        if model_path.exists():
            import tensorflow as tf
            test_model = tf.keras.models.load_model(str(model_path))
            debug_info["manual_load_success"] = True
            debug_info["model_input_shape"] = str(test_model.input_shape)
            debug_info["model_output_shape"] = str(test_model.output_shape)
        else:
            debug_info["manual_load_success"] = False
            debug_info["errors"].append("Cannot test manual load - file doesn't exist")
    except Exception as e:
        debug_info["manual_load_success"] = False
        debug_info["errors"].append(f"Manual model loading failed: {e}")

    # Test pipeline initialization
    try:
        from _helpers import ChessPipelineService
        test_pipeline = ChessPipelineService(str(model_path))
        debug_info["test_pipeline_success"] = test_pipeline.model_loaded
        if not test_pipeline.model_loaded:
            debug_info["errors"].append("Test pipeline failed to load model")
    except Exception as e:
        debug_info["test_pipeline_success"] = False
        debug_info["errors"].append(f"Test pipeline error: {e}")
        import traceback
        debug_info["test_pipeline_traceback"] = traceback.format_exc()

    # Check if the global chess_pipeline has any detailed error info
    if chess_pipeline and hasattr(chess_pipeline, 'model_path'):
        debug_info["global_pipeline_model_path"] = str(chess_pipeline.model_path)
        debug_info["global_pipeline_model_exists"] = chess_pipeline.model_path.exists()

    return debug_info

@app.get("/config/info", response_model=ConfigInfoResponse)
async def get_config_info():
    """Get current configuration information (non-sensitive)"""

    return ConfigInfoResponse(
        environment=os.getenv("ENVIRONMENT", "development"),
        app_version=settings.app_version,
        model_input_size=settings.model_input_size,
        max_image_size_mb=settings.max_image_size_mb,
        supported_formats=settings.supported_image_formats,
        retrain_threshold=settings.retrain_correction_threshold,
        retrain_enabled=settings.retrain_enabled,
        database_type="sqlite" if settings.is_sqlite else "external",
        debug_mode=settings.debug,
        cors_origins=settings.cors_origins if settings.debug else ["***"],  # Hide in production
        rate_limiting_enabled=settings.rate_limit_enabled,
    )


@app.get("/database/info")
async def get_database_info_endpoint():
    """Get database connection information"""

    try:
        db_info = get_database_info()
        db_health = db_health_check()

        return {
            **db_info,
            "health": db_health,
            "connection_status": "connected" if db_health["database_connected"] else "disconnected"
        }

    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get database information")


@app.get("/validate/fen", response_model=FENValidationResponse)
async def validate_fen_notation(fen: str):
    """Validate FEN notation and convert to board matrix"""

    try:
        is_valid = FENValidator.validate_fen(fen)

        if is_valid:
            board_matrix = FENValidator.fen_to_board_matrix(fen)
            return FENValidationResponse(
                valid=True,
                fen=fen,
                board_matrix=board_matrix
            )
        else:
            return FENValidationResponse(
                valid=False,
                error="Invalid FEN notation"
            )

    except Exception as e:
        return FENValidationResponse(
            valid=False,
            error=f"FEN validation error: {str(e)}"
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(413)
async def file_too_large_handler(request, exc):
    """Handle file size too large errors"""
    return JSONResponse(
        status_code=413,
        content={
            "success": False,
            "message": f"File size too large. Maximum size is {settings.max_image_size_mb}MB."
        }
    )


@app.exception_handler(415)
async def unsupported_media_type_handler(request, exc):
    """Handle unsupported file type errors"""
    supported_formats = ", ".join(settings.supported_image_formats).upper()
    return JSONResponse(
        status_code=415,
        content={
            "success": False,
            "message": f"Unsupported file type. Please upload {supported_formats} images."
        }
    )


# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    import logging

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=settings.log_format
    )

    # For development only
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )