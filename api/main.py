# api/main.py - Updated FastAPI Chess Board to FEN Service with Modular Architecture

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
import threading
# Add the parent directory (project root) to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "api" else current_dir
sys.path.insert(0, str(project_root))

# Import configuration
from api.config import settings

# Import database components
from api.database import get_db, check_database_connection, health_check as db_health_check, get_database_info

# Import our chess service components (NEW MODULAR ARCHITECTURE)
from api.services.base import ServiceFactory, ChessPredictionService
from api._helpers import (
    FENValidator,
    validate_uploaded_image,
    resize_image_for_model, ImageProcessor
)

# Import database models
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
    RetrainingStatusResponse,
    ModelMetricsResponse,
    ModelMetrics, ServiceSwitchResponse, ServiceSwitchRequest, CurrentServiceResponse
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

# Initialize chess prediction service (NEW MODULAR APPROACH)
chess_service: ChessPredictionService = None
current_service_type: str = ""
# Thread safety for service switching
service_lock = threading.Lock()


def initialize_chess_service(service_type: str = None) -> tuple[bool, str]:
    """
    Initialize or switch the chess prediction service

    Args:
        service_type: Service type to initialize. If None, uses config default.

    Returns:
        tuple: (success: bool, message: str)
    """
    global chess_service, current_service_type

    try:
        if service_type is None:
            service_type = settings.chess_service_type

        # Validate service type
        available_services = ServiceFactory.get_available_services()
        if service_type not in available_services:
            return False, f"Invalid service type: {service_type}. Available: {available_services}"

        # Get service configuration
        if service_type == "end_to_end":
            service_config = {
                'model_path': settings.end_to_end_model_path,
                'piece_classes': settings.piece_classes,
                'model_input_size': settings.model_input_size,
                'piece_confidence_threshold': settings.end_to_end_piece_confidence,
                'empty_confidence_threshold': settings.end_to_end_empty_confidence
            }
        elif service_type == "multi_model_pipeline":
            service_config = {
                'segmentation_model_path': settings.segmentation_model_path,
                'pieces_model_path': settings.pieces_model_path,
                'piece_classes': [
                    'white-king', 'white-queen', 'white-rook', 'white-bishop', 'white-knight', 'white-pawn',
                    'black-king', 'black-queen', 'black-rook', 'black-bishop', 'black-knight', 'black-pawn'
                ],
                'segmentation_confidence': settings.segmentation_confidence,
                'piece_confidence': settings.piece_detection_confidence,
                'iou_threshold': settings.iou_threshold
            }
        else:
            return False, f"Unknown service type: {service_type}"

        # Create new service
        new_service = ServiceFactory.create_service(service_type, service_config)

        if new_service.is_ready():
            # Replace old service
            chess_service = new_service
            current_service_type = service_type

            logger.info(f"Chess prediction service initialized successfully")
            logger.info(f"   Service type: {chess_service.service_type}")
            logger.info(f"   Service loaded: {chess_service.service_loaded}")

            return True, f"Successfully initialized {service_type} service"
        else:
            return False, f"Service {service_type} failed to initialize properly"

    except Exception as e:
        logger.error(f"Failed to initialize chess service: {e}")
        return False, f"Failed to initialize service: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Initialize the chess prediction service on startup"""
    global chess_service

    try:
        # DEBUG: Print configuration info
        logger.info(f"🔧 Service type: {settings.chess_service_type}")
        logger.info(f"🔧 Service config: {settings.get_service_config()}")

        # Check database connection
        if not check_database_connection():
            raise Exception("Database connection failed")

        # Create the appropriate chess prediction service
        service_config = settings.get_service_config()
        chess_service = ServiceFactory.create_service(
            service_type=settings.chess_service_type,
            config=service_config
        )

        if chess_service.is_ready():
            logger.info(f"♟️ Chess prediction service initialized successfully")
            logger.info(f"   Service type: {chess_service.service_type}")
            logger.info(f"   Service loaded: {chess_service.service_loaded}")
        else:
            raise Exception(f"Chess service failed to initialize: {chess_service.service_type}")

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
        status="healthy" if (
                    chess_service and chess_service.is_ready() and db_health["database_connected"]) else "unhealthy",
        model_ready=chess_service is not None and chess_service.is_ready(),
        database_ready=db_health["database_connected"],
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    )


@app.post("/service/switch", response_model=ServiceSwitchResponse)
async def switch_service(request: ServiceSwitchRequest):
    """
    Switch between different chess prediction services dynamically

    Available service types:
    - end_to_end: Single Keras model for direct FEN prediction
    - multi_model_pipeline: YOLO segmentation + object detection pipeline
    """
    global chess_service, current_service_type

    # Thread safety - only one service switch at a time
    with service_lock:
        try:
            # Validate service type
            available_services = ServiceFactory.get_available_services()
            if request.service_type not in available_services:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid service type: {request.service_type}. Available: {available_services}"
                )

            # Check if already using this service
            if request.service_type == current_service_type:
                return ServiceSwitchResponse(
                    success=True,
                    message=f"Already using {request.service_type} service",
                    previous_service=current_service_type,
                    new_service=request.service_type,
                    timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                )

            previous_service = current_service_type

            # Initialize new service
            success, message = initialize_chess_service(request.service_type)

            if success:
                logger.info(f"Service switched from {previous_service} to {current_service_type}")
                return ServiceSwitchResponse(
                    success=True,
                    message=f"Successfully switched to {request.service_type} service",
                    previous_service=previous_service,
                    new_service=current_service_type,
                    timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to switch service: {message}"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Service switch error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal error during service switch: {str(e)}"
            )


@app.get("/service/current", response_model=CurrentServiceResponse)
async def get_current_service():
    """Get information about the currently active service"""
    if not chess_service:
        return CurrentServiceResponse(
            service_type=None,
            service_loaded=False,
            service_info=None,
            available_services=ServiceFactory.get_available_services(),
            message="No service currently loaded"
        )

    return CurrentServiceResponse(
        service_type=current_service_type,
        service_loaded=chess_service.is_ready(),
        service_info=chess_service.get_service_info(),
        available_services=ServiceFactory.get_available_services(),
        message=f"Currently using {current_service_type} service"
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
    Uses modular service architecture supporting both end-to-end and multi-model pipelines
    """

    if not chess_service or not chess_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Chess prediction service not available (type: {settings.chess_service_type})"
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

        logger.info(f"📸 Processing image: {file.filename} from IP: {device_ip} using {chess_service.service_type}")

        # Process the image through our chess service (MODULAR APPROACH)
        prediction_result = chess_service.predict_from_image(image_bytes)

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
    """Submit a correction for a previous prediction"""
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
            model_version=settings.app_version + f"-{settings.chess_service_type}",
            model_loaded=chess_service is not None and chess_service.is_ready(),
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
        from api.models import ModelVersion, ChessPrediction

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
# DEBUG ENDPOINTS
# ============================================================================

@app.get("/debug/service-info")
async def debug_service_info():
    """Debug endpoint to get information about the current service"""
    if not chess_service:
        return {"error": "No chess service initialized"}

    return {
        "service_info": chess_service.get_service_info(),
        "available_services": ServiceFactory.get_available_services(),
        "selected_service_type": settings.chess_service_type,
        "service_config": settings.get_service_config()
    }


@app.get("/debug/raw-prediction")
async def debug_raw_prediction():
    """Debug raw model predictions (only works for end-to-end service)"""
    import numpy as np

    if not chess_service or not chess_service.is_ready():
        return {"error": "Service not loaded"}

    if chess_service.service_type != "EndToEndPipelineService":
        return {
            "error": f"Raw prediction debug only available for end-to-end service, current: {chess_service.service_type}"}

    try:
        # Create a test input that should produce clear predictions
        test_input = np.random.random((1, 256, 256, 3)).astype(np.float32)

        # Get raw model output (access the model directly for end-to-end service)
        raw_prediction = chess_service.model.predict(test_input, verbose=0)

        # Analyze the output
        debug_info = {
            "model_output_shape": raw_prediction.shape,
            "output_range": [float(raw_prediction.min()), float(raw_prediction.max())],
            "output_mean": float(raw_prediction.mean()),
            "output_std": float(raw_prediction.std()),
            "piece_classes": chess_service.piece_classes,
            "num_classes": len(chess_service.piece_classes),
            "squares_analysis": []
        }

        # Analyze first 8 squares in detail
        for i in range(min(8, 64)):
            square_probs = raw_prediction[0, i]  # Shape should be (13,)
            predicted_class = int(np.argmax(square_probs))
            max_confidence = float(square_probs[predicted_class])

            # Get class name
            class_name = chess_service.piece_classes[predicted_class] if predicted_class < len(
                chess_service.piece_classes) else "unknown"

            square_analysis = {
                "square_index": i,
                "square_position": f"{chr(ord('a') + (i % 8))}{8 - (i // 8)}",  # e.g., 'a8', 'b8', etc.
                "predicted_class_index": predicted_class,
                "predicted_class_name": class_name,
                "confidence": max_confidence,
                "all_probabilities": [float(p) for p in square_probs],
                "piece_symbol": chess_service._piece_name_to_symbol(class_name) if hasattr(chess_service,
                                                                                           '_piece_name_to_symbol') else "N/A"
            }

            debug_info["squares_analysis"].append(square_analysis)

        return debug_info

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/debug/predict-detailed")
async def debug_predict_detailed(file: UploadFile = File(...)):
    """Upload an image and get detailed prediction analysis"""

    if not chess_service or not chess_service.is_ready():
        return {"error": "Service not loaded"}

    try:
        image_bytes = await file.read()

        # Get basic prediction first
        prediction_result = chess_service.predict_from_image(image_bytes)

        debug_info = {
            "service_type": chess_service.service_type,
            "service_config": chess_service.config,
            "prediction_result": {
                "success": prediction_result.success,
                "fen": prediction_result.fen,
                "confidence": prediction_result.confidence,
                "board_detected": prediction_result.board_detected,
                "board_matrix": prediction_result.board_matrix,
                "error_message": prediction_result.error_message,
                "processing_steps": prediction_result.processing_steps
            }
        }

        return debug_info

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/debug/multi-model-detailed")
async def debug_multi_model_detailed(file: UploadFile = File(...)):
    """Debug endpoint specifically for multi-model pipeline issues"""

    if not chess_service or chess_service.service_type != "MultiModelPipelineService":
        return {"error": "Multi-model service not active"}

    try:
        image_bytes = await file.read()
        image = ImageProcessor.load_image_from_bytes(image_bytes)

        if image is None:
            return {"error": "Failed to load image"}

        debug_info = {
            "image_shape": image.shape,
            "service_config": chess_service.config
        }

        # Step 1: Board detection
        corners, seg_confidence, mask = chess_service._detect_board_with_segmentation(image)
        debug_info["board_detection"] = {
            "corners_found": corners is not None,
            "segmentation_confidence": seg_confidence,
            "corners": corners
        }

        if not corners:
            return {"error": "Board not detected", "debug_info": debug_info}

        # Step 2: Piece detection (CRITICAL STEP)
        original_pieces = chess_service._detect_pieces(image)
        debug_info["piece_detection"] = {
            "pieces_found": len(original_pieces),
            "pieces_details": original_pieces[:10]  # First 10 pieces for debugging
        }

        # Step 3: Board warping
        warped_board, transform_matrix, final_corners = chess_service._warp_board_from_mask(
            image, mask, corners
        )

        debug_info["board_warping"] = {
            "warp_successful": warped_board is not None,
            "warped_shape": warped_board.shape if warped_board is not None else None,
            "transform_matrix_exists": transform_matrix is not None
        }

        if warped_board is None:
            return {"error": "Board warping failed", "debug_info": debug_info}

        # Step 4: Transform pieces to warped space
        warped_pieces = chess_service._transform_pieces_to_warped_space(
            original_pieces, transform_matrix, warped_board.shape[:2]
        )

        debug_info["coordinate_transformation"] = {
            "original_pieces": len(original_pieces),
            "warped_pieces": len(warped_pieces),
            "warped_pieces_details": warped_pieces[:10]
        }

        # Step 5: Assign pieces to squares
        board_matrix = chess_service._assign_pieces_to_squares(warped_pieces)

        pieces_on_board = sum(1 for row in board_matrix for cell in row if cell != '')
        debug_info["piece_assignment"] = {
            "pieces_assigned": pieces_on_board,
            "board_matrix": board_matrix
        }

        # Test with lower thresholds
        debug_info["threshold_analysis"] = {}

        # Try with very low confidence threshold
        low_confidence_pieces = []
        try:
            results = chess_service.pieces_model(image, conf=0.1, iou=0.3)  # Much lower thresholds
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        piece_info = {
                            'x_center': float(box.xywh[0][0]),
                            'y_center': float(box.xywh[0][1]),
                            'confidence': float(box.conf[0]),
                            'category_id': int(box.cls[0])
                        }
                        if hasattr(result, 'names') and piece_info['category_id'] in result.names:
                            piece_info['class_name'] = result.names[piece_info['category_id']]
                        low_confidence_pieces.append(piece_info)

            debug_info["threshold_analysis"]["low_threshold_pieces"] = {
                "count": len(low_confidence_pieces),
                "pieces": low_confidence_pieces[:15]  # First 15
            }
        except Exception as e:
            debug_info["threshold_analysis"]["error"] = str(e)

        return debug_info

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

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