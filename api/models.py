# api/cnn_models.py - Database Models for Chess Board to FEN API

from sqlalchemy import Column, Integer, String, Text, LargeBinary, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import json
from typing import List, Optional, Dict, Any

Base = declarative_base()


class ChessPrediction(Base):
    """
    Main table for chess board predictions and corrections

    This single table handles:
    - Original predictions from uploaded images
    - User corrections for improving the model
    - Device tracking via IP address
    - Retraining data collection
    """
    __tablename__ = "chess_predictions"

    # Primary identifier
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Image data (resized to model expectations)
    image = Column(LargeBinary, nullable=False,
                   comment="Resized image data for model input (224x224)")

    # Prediction results
    predicted_fen = Column(String(200), nullable=True,
                           comment="AI-generated FEN notation")
    predicted_matrix = Column(Text, nullable=True,
                              comment="AI-generated 8x8 board matrix as JSON string")

    # User corrections (for model improvement)
    corrected_fen = Column(String(200), nullable=True,
                           comment="User-corrected FEN notation")
    corrected_matrix = Column(Text, nullable=True,
                              comment="User-corrected 8x8 board matrix as JSON string")

    # Device and performance tracking
    device_identifier = Column(String(100), nullable=False, index=True,
                               comment="Client IP address for device identification")
    confidence_score = Column(Float, nullable=True,
                              comment="Model confidence score (0-1)")
    processing_time_ms = Column(Integer, nullable=True,
                                comment="Processing time in milliseconds")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True,
                        comment="When prediction was made")
    corrected_at = Column(DateTime, nullable=True, index=True,
                          comment="When correction was submitted")

    # Status tracking
    board_detected = Column(Boolean, nullable=True,
                            comment="Whether chess board was successfully detected")
    prediction_successful = Column(Boolean, nullable=True,
                                   comment="Whether FEN generation was successful")

    def __repr__(self):
        return f"<ChessPrediction(id={self.id}, device={self.device_identifier}, created={self.created_at})>"

    @property
    def has_correction(self) -> bool:
        """Check if this prediction has been corrected"""
        return self.corrected_fen is not None

    @property
    def is_successful(self) -> bool:
        """Check if prediction was successful"""
        return self.predicted_fen is not None and self.prediction_successful is True

    @property
    def processing_time_seconds(self) -> Optional[float]:
        """Get processing time in seconds"""
        if self.processing_time_ms is None:
            return None
        return self.processing_time_ms / 1000.0

    def get_predicted_matrix(self) -> Optional[List[List[str]]]:
        """Parse predicted matrix from JSON string"""
        if not self.predicted_matrix:
            return None
        try:
            return json.loads(self.predicted_matrix)
        except (json.JSONDecodeError, TypeError):
            return None

    def set_predicted_matrix(self, matrix: List[List[str]]) -> None:
        """Set predicted matrix as JSON string"""
        if matrix:
            self.predicted_matrix = json.dumps(matrix)
        else:
            self.predicted_matrix = None

    def get_corrected_matrix(self) -> Optional[List[List[str]]]:
        """Parse corrected matrix from JSON string"""
        if not self.corrected_matrix:
            return None
        try:
            return json.loads(self.corrected_matrix)
        except (json.JSONDecodeError, TypeError):
            return None

    def set_corrected_matrix(self, matrix: List[List[str]]) -> None:
        """Set corrected matrix as JSON string"""
        if matrix:
            self.corrected_matrix = json.dumps(matrix)
        else:
            self.corrected_matrix = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses"""
        return {
            "id": self.id,
            "predicted_fen": self.predicted_fen,
            "predicted_matrix": self.get_predicted_matrix(),
            "corrected_fen": self.corrected_fen,
            "corrected_matrix": self.get_corrected_matrix(),
            "device_identifier": self.device_identifier,
            "confidence_score": self.confidence_score,
            "processing_time_ms": self.processing_time_ms,
            "processing_time_seconds": self.processing_time_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "corrected_at": self.corrected_at.isoformat() if self.corrected_at else None,
            "board_detected": self.board_detected,
            "prediction_successful": self.prediction_successful,
            "has_correction": self.has_correction,
            "is_successful": self.is_successful
        }

    def to_training_sample(self) -> Optional[Dict[str, Any]]:
        """
        Convert to training sample format for model retraining
        Returns None if not suitable for training
        """
        # Only use corrected samples for training
        if not self.has_correction:
            return None

        # Must have image data
        if not self.image:
            return None

        return {
            "id": self.id,
            "image_data": self.image,
            "fen": self.corrected_fen,
            "board_matrix": self.get_corrected_matrix(),
            "original_prediction": self.predicted_fen,
            "confidence_score": self.confidence_score,
            "device_identifier": self.device_identifier,
            "created_at": self.created_at,
            "corrected_at": self.corrected_at
        }


class ModelVersion(Base):
    """
    Track different versions of the trained model
    (Optional - for future model versioning)
    """
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_number = Column(String(50), nullable=False, unique=True,
                            comment="Model version identifier (e.g., 'v1.0.0')")
    model_path = Column(String(500), nullable=False,
                        comment="Path to the model file")
    training_data_count = Column(Integer, nullable=False, default=0,
                                 comment="Number of samples used for training")
    validation_accuracy = Column(Float, nullable=True,
                                 comment="Validation accuracy score")
    performance_metrics = Column(Text, nullable=True,
                                 comment="JSON string of detailed performance metrics")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=False, nullable=False,
                       comment="Whether this is the currently active model")
    notes = Column(Text, nullable=True,
                   comment="Additional notes about this model version")

    def __repr__(self):
        return f"<ModelVersion(version={self.version_number}, active={self.is_active})>"

    def get_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """Parse performance metrics from JSON string"""
        if not self.performance_metrics:
            return None
        try:
            return json.loads(self.performance_metrics)
        except (json.JSONDecodeError, TypeError):
            return None

    def set_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Set performance metrics as JSON string"""
        if metrics:
            self.performance_metrics = json.dumps(metrics)
        else:
            self.performance_metrics = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses"""
        return {
            "id": self.id,
            "version_number": self.version_number,
            "model_path": self.model_path,
            "training_data_count": self.training_data_count,
            "validation_accuracy": self.validation_accuracy,
            "performance_metrics": self.get_performance_metrics(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
            "notes": self.notes
        }


# ============================================================================
# HELPER FUNCTIONS FOR DATABASE QUERIES
# ============================================================================

def get_recent_predictions(db_session, limit: int = 100, device_identifier: str = None):
    """Get recent predictions, optionally filtered by device"""
    query = db_session.query(ChessPrediction).order_by(ChessPrediction.created_at.desc())

    if device_identifier:
        query = query.filter(ChessPrediction.device_identifier == device_identifier)

    return query.limit(limit).all()


def get_successful_predictions(db_session, limit: int = None):
    """Get all successful predictions"""
    query = db_session.query(ChessPrediction).filter(
        ChessPrediction.prediction_successful == True,
        ChessPrediction.predicted_fen.isnot(None)
    ).order_by(ChessPrediction.created_at.desc())

    if limit:
        query = query.limit(limit)

    return query.all()


def get_corrected_predictions(db_session, limit: int = None):
    """Get all predictions that have been corrected"""
    query = db_session.query(ChessPrediction).filter(
        ChessPrediction.corrected_fen.isnot(None)
    ).order_by(ChessPrediction.corrected_at.desc())

    if limit:
        query = query.limit(limit)

    return query.all()


def get_training_data(db_session, min_corrections: int = 100):
    """Get data suitable for model retraining"""
    return db_session.query(ChessPrediction).filter(
        ChessPrediction.corrected_fen.isnot(None),
        ChessPrediction.image.isnot(None)
    ).order_by(ChessPrediction.corrected_at.desc()).limit(min_corrections * 10).all()


def get_database_statistics(db_session) -> Dict[str, Any]:
    """Get comprehensive database statistics"""

    total_predictions = db_session.query(ChessPrediction).count()

    successful_predictions = db_session.query(ChessPrediction).filter(
        ChessPrediction.prediction_successful == True
    ).count()

    failed_predictions = total_predictions - successful_predictions

    corrections_count = db_session.query(ChessPrediction).filter(
        ChessPrediction.corrected_fen.isnot(None)
    ).count()

    board_detection_success = db_session.query(ChessPrediction).filter(
        ChessPrediction.board_detected == True
    ).count()

    # Calculate averages
    avg_processing_time = db_session.query(ChessPrediction.processing_time_ms).filter(
        ChessPrediction.processing_time_ms.isnot(None)
    ).all()
    avg_processing_time = sum(x[0] for x in avg_processing_time) / len(
        avg_processing_time) if avg_processing_time else 0

    avg_confidence = db_session.query(ChessPrediction.confidence_score).filter(
        ChessPrediction.confidence_score.isnot(None)
    ).all()
    avg_confidence = sum(x[0] for x in avg_confidence) / len(avg_confidence) if avg_confidence else 0

    # Unique devices
    unique_devices = db_session.query(ChessPrediction.device_identifier).distinct().count()

    return {
        "total_predictions": total_predictions,
        "successful_predictions": successful_predictions,
        "failed_predictions": failed_predictions,
        "corrections_submitted": corrections_count,
        "board_detection_success_rate": board_detection_success / total_predictions if total_predictions > 0 else 0,
        "prediction_success_rate": successful_predictions / total_predictions if total_predictions > 0 else 0,
        "correction_rate": corrections_count / total_predictions if total_predictions > 0 else 0,
        "average_processing_time_ms": avg_processing_time,
        "average_confidence_score": avg_confidence,
        "unique_devices": unique_devices
    }


def check_retraining_threshold(db_session, threshold: int = 1000) -> Dict[str, Any]:
    """Check if model retraining threshold has been reached"""

    corrections_count = db_session.query(ChessPrediction).filter(
        ChessPrediction.corrected_fen.isnot(None)
    ).count()

    # Get the last model version to see how many corrections since then
    last_model = db_session.query(ModelVersion).filter(
        ModelVersion.is_active == True
    ).first()

    corrections_since_last_model = corrections_count
    if last_model and last_model.created_at:
        corrections_since_last_model = db_session.query(ChessPrediction).filter(
            ChessPrediction.corrected_fen.isnot(None),
            ChessPrediction.corrected_at > last_model.created_at
        ).count()

    needs_retraining = corrections_since_last_model >= threshold

    return {
        "total_corrections": corrections_count,
        "corrections_since_last_model": corrections_since_last_model,
        "retrain_threshold": threshold,
        "needs_retraining": needs_retraining,
        "corrections_until_retrain": max(0, threshold - corrections_since_last_model),
        "last_model_version": last_model.version_number if last_model else None
    }


def get_model_metrics(db_session) -> List[Dict[str, Any]]:
    """
    Calculate comprehensive metrics for each model version
    """
    from sqlalchemy import func
    from datetime import datetime, timedelta

    # Get all model versions
    model_versions = db_session.query(ModelVersion).order_by(ModelVersion.created_at.asc()).all()

    if not model_versions:
        return []

    metrics_list = []

    for i, model in enumerate(model_versions):
        # Determine the time period this model was active
        start_date = model.created_at

        # End date is either the next model's start date or now if it's the current model
        if i + 1 < len(model_versions):
            end_date = model_versions[i + 1].created_at
        else:
            end_date = datetime.utcnow()

        # Get predictions made during this model's active period
        predictions_query = db_session.query(ChessPrediction).filter(
            ChessPrediction.created_at >= start_date,
            ChessPrediction.created_at < end_date
        )

        total_predictions = predictions_query.count()

        # Calculate success metrics
        successful_predictions = predictions_query.filter(
            ChessPrediction.prediction_successful == True
        ).count()

        failed_predictions = total_predictions - successful_predictions
        success_rate = successful_predictions / total_predictions if total_predictions > 0 else 0.0

        # Calculate correction metrics
        corrections_received = predictions_query.filter(
            ChessPrediction.corrected_fen.isnot(None)
        ).count()

        correction_rate = corrections_received / total_predictions if total_predictions > 0 else 0.0

        # Calculate average confidence
        confidence_scores = db_session.query(ChessPrediction.confidence_score).filter(
            ChessPrediction.created_at >= start_date,
            ChessPrediction.created_at < end_date,
            ChessPrediction.confidence_score.isnot(None)
        ).all()

        average_confidence = None
        if confidence_scores:
            average_confidence = sum(score[0] for score in confidence_scores) / len(confidence_scores)

        # Calculate average processing time
        processing_times = db_session.query(ChessPrediction.processing_time_ms).filter(
            ChessPrediction.created_at >= start_date,
            ChessPrediction.created_at < end_date,
            ChessPrediction.processing_time_ms.isnot(None)
        ).all()

        average_processing_time = None
        if processing_times:
            average_processing_time = sum(time[0] for time in processing_times) / len(processing_times)

        # Calculate time-based metrics
        active_duration = end_date - start_date
        active_duration_days = active_duration.days

        predictions_per_day = None
        if active_duration_days > 0:
            predictions_per_day = total_predictions / active_duration_days

        # Compile metrics
        model_metrics = {
            "version_id": model.id,
            "version_number": model.version_number,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "is_active": model.is_active,
            "training_data_count": model.training_data_count,
            "validation_accuracy": model.validation_accuracy,
            "performance_metrics": model.get_performance_metrics(),
            "total_predictions": total_predictions,
            "successful_predictions": successful_predictions,
            "failed_predictions": failed_predictions,
            "success_rate": round(success_rate, 4),
            "corrections_received": corrections_received,
            "correction_rate": round(correction_rate, 4),
            "average_confidence": round(average_confidence, 4) if average_confidence else None,
            "average_processing_time_ms": round(average_processing_time, 2) if average_processing_time else None,
            "active_duration_days": active_duration_days,
            "predictions_per_day": round(predictions_per_day, 2) if predictions_per_day else None,
            "notes": model.notes
        }

        metrics_list.append(model_metrics)

    return metrics_list


def get_current_active_model(db_session) -> Optional[ModelVersion]:
    """Get the currently active model version"""
    return db_session.query(ModelVersion).filter(
        ModelVersion.is_active == True
    ).first()