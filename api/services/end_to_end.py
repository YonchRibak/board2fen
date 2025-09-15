# api/services/end_to_end.py - End-to-end chess prediction service

import time
import logging
from typing import Optional, List, Tuple
import numpy as np
import tensorflow as tf

from api.services.base import ChessPredictionService
from api._helpers import (
    PredictionResult,
    ImageProcessor,
    FENValidator,
    ModelDownloader
)
from api.config import settings

logger = logging.getLogger(__name__)


class EndToEndPipelineService(ChessPredictionService):
    """
    Chess prediction service using a single end-to-end model.
    This is the original approach from your current implementation.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.actual_model_path = None
        self.piece_classes = list(config.get('piece_classes', settings.piece_classes))

        # Load models on initialization
        if self._load_models():
            self.service_loaded = True
        else:
            logger.error("Failed to load end-to-end model")

    def _load_models(self) -> bool:
        """Load the end-to-end model from local path or URL"""
        try:
            model_path_or_url = self.config.get('model_path', settings.model_path)

            # Get the local path for the model (downloading if necessary)
            local_model_path = ModelDownloader.get_model_path(model_path_or_url)

            if local_model_path is None:
                logger.error(f"Failed to get model from: {model_path_or_url}")
                return False

            self.actual_model_path = local_model_path

            # Check if model file exists
            if not local_model_path.exists():
                logger.error(f"Model file not found: {local_model_path}")
                return False

            # Load the model
            logger.info(f"Loading end-to-end model from: {local_model_path}")
            self.model = tf.keras.models.load_model(str(local_model_path))

            logger.info(f"End-to-end chess model loaded successfully")
            logger.info(f"   Source: {model_path_or_url}")
            logger.info(f"   Local path: {local_model_path}")
            logger.info(f"   Model input shape: {self.model.input_shape}")
            logger.info(f"   Model output shape: {self.model.output_shape}")

            return True

        except Exception as e:
            logger.error(f"Failed to load end-to-end model: {e}")
            return False

    def predict_from_image(self, image_bytes: bytes) -> PredictionResult:
        """
        Main prediction method using end-to-end model

        For end-to-end model:
        1. Load and preprocess image
        2. Run inference with the model
        3. Convert model output to board matrix
        4. Generate FEN notation
        """

        if not self.service_loaded:
            return PredictionResult(
                success=False,
                error_message="End-to-end model not loaded properly"
            )

        processing_steps = {}
        start_time = time.time()

        try:
            # Step 1: Load and preprocess image
            step_start = time.time()
            image = ImageProcessor.load_image_from_bytes(image_bytes)
            if image is None:
                return PredictionResult(
                    success=False,
                    error_message="Failed to load image"
                )

            # Preprocess for model
            processed_image = ImageProcessor.preprocess_for_model(
                image,
                target_size=settings.model_input_size
            )
            processing_steps['image_loading'] = time.time() - step_start

            # Step 2: Run model inference
            step_start = time.time()
            model_output = self._run_inference(processed_image)
            processing_steps['model_inference'] = time.time() - step_start

            if model_output is None:
                return PredictionResult(
                    success=False,
                    error_message="Model inference failed"
                )

            # Step 3: Convert model output to board matrix
            step_start = time.time()
            board_matrix, confidence = self._convert_output_to_board(model_output)
            processing_steps['output_conversion'] = time.time() - step_start

            # Step 4: Generate FEN notation
            step_start = time.time()
            fen = FENValidator.board_matrix_to_fen(board_matrix)
            processing_steps['fen_generation'] = time.time() - step_start

            if fen is None:
                return PredictionResult(
                    success=False,
                    board_detected=True,
                    error_message="Failed to generate valid FEN",
                    processing_steps=processing_steps
                )

            processing_steps['total_time'] = time.time() - start_time

            return PredictionResult(
                success=True,
                fen=fen,
                board_matrix=board_matrix,
                confidence=float(confidence),
                board_detected=True,
                processing_steps=processing_steps
            )

        except Exception as e:
            logger.error(f"End-to-end pipeline error: {e}")
            return PredictionResult(
                success=False,
                error_message=f"End-to-end pipeline error: {str(e)}",
                processing_steps=processing_steps
            )

    def _run_inference(self, processed_image: np.ndarray) -> Optional[np.ndarray]:
        """Run inference on the preprocessed image"""
        try:
            # Add batch dimension
            batch_input = np.expand_dims(processed_image, axis=0)

            # Run inference
            prediction = self.model.predict(batch_input, verbose=0)

            return prediction[0]  # Remove batch dimension

        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return None

    def _convert_output_to_board(self, model_output: np.ndarray) -> Tuple[List[List[str]], float]:
        """
        Convert model output to 8x8 board matrix

        This function handles different output formats:
        - If output is (64, num_classes): each position is a class prediction
        - If output is (8, 8, num_classes): already in board format
        - If output is a single vector: might need reshaping
        """
        try:
            # Get the shape to determine output format
            output_shape = model_output.shape
            logger.debug(f"Model output shape: {output_shape}")

            # Case 1: Output is (64, num_classes) - flattened board
            if len(output_shape) == 2 and output_shape[0] == 64:
                return self._convert_flattened_output(model_output)

            # Case 2: Output is (8, 8, num_classes) - board format
            elif len(output_shape) == 3 and output_shape[0] == 8 and output_shape[1] == 8:
                return self._convert_board_format_output(model_output)

            # Case 3: Try to reshape if it's close to 64 * num_classes
            else:
                total_elements = np.prod(output_shape)
                if total_elements % 64 == 0:
                    num_classes = total_elements // 64
                    reshaped = model_output.reshape(64, num_classes)
                    return self._convert_flattened_output(reshaped)
                else:
                    logger.error(f"Unexpected model output shape: {output_shape}")
                    return self._get_empty_board(), 0.0

        except Exception as e:
            logger.error(f"Error converting model output: {e}")
            return self._get_empty_board(), 0.0

    def _convert_flattened_output(self, output: np.ndarray) -> Tuple[List[List[str]], float]:
        """Convert flattened (64, num_classes) output to board matrix"""
        board_matrix = [[''] * 8 for _ in range(8)]
        confidences = []

        # Configure confidence thresholds
        piece_threshold = self.config.get('piece_confidence_threshold', 0.3)
        empty_threshold = self.config.get('empty_confidence_threshold', 0.5)

        for i in range(64):
            # Get the most likely class for this square
            class_probs = output[i]
            predicted_class_idx = np.argmax(class_probs)
            confidence = float(class_probs[predicted_class_idx])

            # Convert flat index to board coordinates
            rank = i // 8
            file = i % 8

            # Get piece symbol
            if predicted_class_idx < len(self.piece_classes):
                piece_name = self.piece_classes[predicted_class_idx]
                piece_symbol = self._piece_name_to_symbol(piece_name)

                if piece_symbol != '':  # It's a piece
                    if confidence > piece_threshold:
                        board_matrix[rank][file] = piece_symbol
                        confidences.append(confidence)
                else:  # It's empty
                    if piece_name.lower() == 'empty' and confidence > empty_threshold:
                        confidences.append(confidence)

        # Calculate average confidence
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return board_matrix, avg_confidence

    def _convert_board_format_output(self, output: np.ndarray) -> Tuple[List[List[str]], float]:
        """Convert (8, 8, num_classes) output to board matrix"""
        board_matrix = [[''] * 8 for _ in range(8)]
        confidences = []

        # Configure confidence thresholds
        piece_threshold = self.config.get('piece_confidence_threshold', 0.3)
        empty_threshold = self.config.get('empty_confidence_threshold', 0.5)

        for rank in range(8):
            for file in range(8):
                class_probs = output[rank, file]
                predicted_class_idx = np.argmax(class_probs)
                confidence = float(class_probs[predicted_class_idx])

                # Get piece symbol
                if predicted_class_idx < len(self.piece_classes):
                    piece_name = self.piece_classes[predicted_class_idx]
                    piece_symbol = self._piece_name_to_symbol(piece_name)

                    if piece_symbol != '':  # It's a piece
                        if confidence > piece_threshold:
                            board_matrix[rank][file] = piece_symbol
                            confidences.append(confidence)
                    else:  # It's empty
                        if piece_name.lower() == 'empty' and confidence > empty_threshold:
                            confidences.append(confidence)

        # Calculate average confidence
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return board_matrix, avg_confidence

    def _piece_name_to_symbol(self, piece_name: str) -> str:
        """Convert piece class name to FEN symbol"""
        name_to_symbol = {
            # Empty square
            'empty': '', 'none': '', 'blank': '', '': '',

            # Standard naming with underscores
            'white_king': 'K', 'white_queen': 'Q', 'white_rook': 'R',
            'white_bishop': 'B', 'white_knight': 'N', 'white_pawn': 'P',
            'black_king': 'k', 'black_queen': 'q', 'black_rook': 'r',
            'black_bishop': 'b', 'black_knight': 'n', 'black_pawn': 'p',

            # Hyphen versions
            'white-king': 'K', 'white-queen': 'Q', 'white-rook': 'R',
            'white-bishop': 'B', 'white-knight': 'N', 'white-pawn': 'P',
            'black-king': 'k', 'black-queen': 'q', 'black-rook': 'r',
            'black-bishop': 'b', 'black-knight': 'n', 'black-pawn': 'p',

            # Direct symbols
            'K': 'K', 'Q': 'Q', 'R': 'R', 'B': 'B', 'N': 'N', 'P': 'P',
            'k': 'k', 'q': 'q', 'r': 'r', 'b': 'b', 'n': 'n', 'p': 'p',
        }

        symbol = name_to_symbol.get(piece_name.lower(), '')

        if symbol == '' and piece_name.lower() != 'empty':
            logger.warning(f"Unknown piece class: '{piece_name}' - check class name format")

        return symbol

    def _get_empty_board(self) -> List[List[str]]:
        """Return an empty 8x8 board"""
        return [[''] * 8 for _ in range(8)]