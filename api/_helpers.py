# api/_helpers.py - Helper functions and services for Chess Board to FEN API

import sys
import hashlib
import logging
import time
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image
import chess
import tensorflow as tf

# Add the parent directory (project root) to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "api" else current_dir
sys.path.insert(0, str(project_root))

# Import configuration
from api.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PredictionResult:
    """Structured result from chess board prediction"""
    success: bool
    fen: Optional[str] = None
    board_matrix: Optional[List[List[str]]] = None
    confidence: Optional[float] = None
    board_detected: Optional[bool] = None
    error_message: Optional[str] = None
    processing_steps: Optional[Dict[str, Any]] = None


# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

class ImageProcessor:
    """Utility class for image processing operations"""

    # Get configuration from settings
    SUPPORTED_FORMATS = set(settings.supported_image_formats)
    MAX_IMAGE_SIZE = settings.max_image_size_bytes

    @staticmethod
    def calculate_image_hash(image_bytes: bytes) -> str:
        """Calculate SHA256 hash of image bytes"""
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def validate_image_format(filename: str) -> bool:
        """Validate if image format is supported"""
        if not filename:
            return False

        extension = filename.lower().split('.')[-1]
        return extension in ImageProcessor.SUPPORTED_FORMATS

    @staticmethod
    def validate_image_size(image_bytes: bytes) -> bool:
        """Validate if image size is within limits"""
        return len(image_bytes) <= ImageProcessor.MAX_IMAGE_SIZE

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
        """Load image from bytes and convert to RGB numpy array"""
        try:
            # Use PIL to load the image
            pil_image = Image.open(BytesIO(image_bytes))

            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Convert to numpy array
            image_array = np.array(pil_image)

            return image_array

        except Exception as e:
            logger.error(f"Failed to load image from bytes: {e}")
            return None

    @staticmethod
    def preprocess_for_model(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess image for the end-to-end model"""
        try:
            # Resize to target size
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0

            return normalized

        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise e


def resize_image_for_model(image_bytes: bytes) -> bytes:
    """
    Resize image to model's expected input size and return as bytes
    This is what gets stored in the database
    """
    try:
        # Load image
        pil_image = Image.open(BytesIO(image_bytes))

        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Resize to model input size from settings
        resized_image = pil_image.resize(settings.model_input_size, Image.Resampling.LANCZOS)

        # Convert back to bytes
        output_buffer = BytesIO()
        resized_image.save(output_buffer, format='JPEG', quality=settings.image_storage_quality)
        output_buffer.seek(0)

        return output_buffer.read()

    except Exception as e:
        logger.error(f"Failed to resize image for model: {e}")
        raise e


# ============================================================================
# FEN VALIDATION AND CONVERSION
# ============================================================================

class FENValidator:
    """Utility class for FEN notation validation and conversion"""

    # Standard piece symbols
    PIECE_SYMBOLS = {'p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K'}

    @staticmethod
    def validate_fen(fen_string: str) -> bool:
        """Validate FEN notation using python-chess library"""
        try:
            board = chess.Board(fen_string)
            return True
        except ValueError:
            return False

    @staticmethod
    def fen_to_board_matrix(fen: str) -> List[List[str]]:
        """Convert FEN notation to 8x8 board matrix"""
        try:
            board = chess.Board(fen)
            matrix = []

            for rank in range(8):  # 8 ranks (rows)
                row = []
                for file in range(8):  # 8 files (columns)
                    square = chess.square(file, 7 - rank)  # chess uses 0-based indexing
                    piece = board.piece_at(square)

                    if piece is None:
                        row.append('')  # Empty square
                    else:
                        row.append(str(piece))  # Piece symbol

                matrix.append(row)

            return matrix

        except Exception as e:
            logger.error(f"Error converting FEN to matrix: {e}")
            return [[''] * 8 for _ in range(8)]  # Empty board

    @staticmethod
    def board_matrix_to_fen(matrix: List[List[str]],
                            active_color: str = 'w',
                            castling: str = 'KQkq',
                            en_passant: str = '-',
                            halfmove: int = 0,
                            fullmove: int = 1) -> str:
        """Convert 8x8 board matrix to FEN notation"""
        try:
            # Build the board position part
            fen_parts = []

            for row in matrix:
                fen_row = ""
                empty_count = 0

                for square in row:
                    if square == '' or square is None:
                        empty_count += 1
                    else:
                        if empty_count > 0:
                            fen_row += str(empty_count)
                            empty_count = 0
                        fen_row += square

                if empty_count > 0:
                    fen_row += str(empty_count)

                fen_parts.append(fen_row)

            board_position = '/'.join(fen_parts)

            # Combine all FEN components
            fen = f"{board_position} {active_color} {castling} {en_passant} {halfmove} {fullmove}"

            # Validate the generated FEN
            if FENValidator.validate_fen(fen):
                return fen
            else:
                logger.error("Generated invalid FEN")
                return None

        except Exception as e:
            logger.error(f"Error converting matrix to FEN: {e}")
            return None

    @staticmethod
    def compare_positions(fen1: str, fen2: str) -> float:
        """Compare two chess positions and return similarity score (0-1)"""
        try:
            matrix1 = FENValidator.fen_to_board_matrix(fen1)
            matrix2 = FENValidator.fen_to_board_matrix(fen2)

            matches = 0
            total_squares = 64

            for i in range(8):
                for j in range(8):
                    if matrix1[i][j] == matrix2[i][j]:
                        matches += 1

            return matches / total_squares

        except Exception as e:
            logger.error(f"Error comparing positions: {e}")
            return 0.0


# ============================================================================
# CHESS PIPELINE SERVICE (SIMPLIFIED FOR END-TO-END MODEL)
# ============================================================================

class ChessPipelineService:
    """
    Simplified service for end-to-end chess board to FEN conversion
    Uses a single CNN model that processes the entire chess board image
    """

    def __init__(self, model_path: str):
        """Initialize the chess pipeline with model path"""
        self.model_path = Path(model_path)
        self.model = None
        self.model_loaded = False
        self.piece_classes = list(settings.piece_classes)

        self._load_model()

    def _load_model(self):
        """Load the trained end-to-end model"""
        try:
            # Check if model file exists
            if not self.model_path.exists():
                logger.error(f"❌ Model file not found: {self.model_path}")
                return

            # Load the model
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info(f"✅ End-to-end chess model loaded from {self.model_path}")
            logger.info(f"   Model input shape: {self.model.input_shape}")
            logger.info(f"   Model output shape: {self.model.output_shape}")

            self.model_loaded = True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model_loaded = False

    def predict_from_image(self, image_bytes: bytes) -> PredictionResult:
        """
        Main prediction method: convert image bytes to FEN notation

        For end-to-end model:
        1. Load and preprocess image
        2. Run inference with the model
        3. Convert model output to board matrix
        4. Generate FEN notation
        """

        if not self.model_loaded:
            return PredictionResult(
                success=False,
                error_message="Model not loaded properly"
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
                confidence=float(confidence),  # Ensure Python float
                board_detected=True,
                processing_steps=processing_steps
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return PredictionResult(
                success=False,
                error_message=f"Pipeline error: {str(e)}",
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

        This function needs to be adapted based on your model's output format:
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

            # Case 3: Output is a single prediction vector
            elif len(output_shape) == 1:
                return self._convert_single_vector_output(model_output)

            # Case 4: Try to reshape if it's close to 64 * num_classes
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

                # IMPORTANT: Now that 'empty' is a valid class that returns '',
                # we need to distinguish between:
                # 1. High confidence empty prediction (model says it's empty)
                # 2. Low confidence piece prediction (uncertain, treat as empty)

                if piece_symbol != '':  # It's a piece
                    if confidence > 0.3:  # Increased threshold for pieces
                        board_matrix[rank][file] = piece_symbol
                        confidences.append(confidence)
                else:  # It's empty (piece_symbol == '')
                    # Empty squares don't need to be added to board_matrix (already '')
                    # But we might want to track high-confidence empty predictions
                    if piece_name.lower() == 'empty' and confidence > 0.5:
                        confidences.append(confidence)  # Track confident empty predictions

        # Calculate average confidence and convert to Python float
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return board_matrix, avg_confidence

    def _convert_board_format_output(self, output: np.ndarray) -> Tuple[List[List[str]], float]:
        """Convert (8, 8, num_classes) output to board matrix"""
        board_matrix = [[''] * 8 for _ in range(8)]
        confidences = []

        for rank in range(8):
            for file in range(8):
                class_probs = output[rank, file]
                predicted_class_idx = np.argmax(class_probs)
                confidence = float(class_probs[predicted_class_idx])

                # Get piece symbol
                if predicted_class_idx < len(self.piece_classes):
                    piece_name = self.piece_classes[predicted_class_idx]
                    piece_symbol = self._piece_name_to_symbol(piece_name)

                    # Same logic as flattened version
                    if piece_symbol != '':  # It's a piece
                        if confidence > 0.3:  # Increased threshold for pieces
                            board_matrix[rank][file] = piece_symbol
                            confidences.append(confidence)
                    else:  # It's empty
                        if piece_name.lower() == 'empty' and confidence > 0.5:
                            confidences.append(confidence)  # Track confident empty predictions

        # Calculate average confidence and convert to Python float
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return board_matrix, avg_confidence

    def _convert_single_vector_output(self, output: np.ndarray) -> Tuple[List[List[str]], float]:
        """Convert single vector output (if model outputs a single prediction)"""
        # This is a fallback - you might need to adapt this based on your specific model
        logger.warning("Single vector output detected - using fallback conversion")
        return self._get_empty_board(), 0.5

    def _piece_name_to_symbol(self, piece_name: str) -> str:
        """Convert piece class name to FEN symbol"""
        # Handle different possible naming conventions
        name_to_symbol = {
            # Empty square - MUST be handled first and return empty string
            'empty': '', 'none': '', 'blank': '', '': '',

            # Standard naming
            'white_king': 'K', 'white_queen': 'Q', 'white_rook': 'R',
            'white_bishop': 'B', 'white_knight': 'N', 'white_pawn': 'P',
            'black_king': 'k', 'black_queen': 'q', 'black_rook': 'r',
            'black_bishop': 'b', 'black_knight': 'n', 'black_pawn': 'p',

            # Alternative naming (direct piece symbols)
            'K': 'K', 'Q': 'Q', 'R': 'R', 'B': 'B', 'N': 'N', 'P': 'P',
            'k': 'k', 'q': 'q', 'r': 'r', 'b': 'b', 'n': 'n', 'p': 'p',
        }

        symbol = name_to_symbol.get(piece_name.lower(), '')

        # Debug logging to help identify any issues
        if piece_name.lower() == 'empty':
            logger.debug(f"Empty square detected: {piece_name} -> '{symbol}'")
        elif symbol == '':
            logger.warning(f"Unknown piece class: {piece_name}")

        return symbol

    def _get_empty_board(self) -> List[List[str]]:
        """Return an empty 8x8 board"""
        return [[''] * 8 for _ in range(8)]


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_uploaded_image(image_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Validate uploaded image file"""

    # Check file format
    if not ImageProcessor.validate_image_format(filename):
        supported_formats = ", ".join(settings.supported_image_formats).upper()
        return {
            "valid": False,
            "error": f"Unsupported file format. Supported formats: {supported_formats}"
        }

    # Check file size
    if not ImageProcessor.validate_image_size(image_bytes):
        return {
            "valid": False,
            "error": f"File too large. Maximum size: {settings.max_image_size_mb:.1f}MB"
        }

    # Try to load the image
    image = ImageProcessor.load_image_from_bytes(image_bytes)
    if image is None:
        return {
            "valid": False,
            "error": "Invalid or corrupted image file"
        }

    # Check image dimensions
    h, w = image.shape[:2]
    if h < settings.min_image_dimension or w < settings.min_image_dimension:
        return {
            "valid": False,
            "error": f"Image too small. Minimum size: {settings.min_image_dimension}x{settings.min_image_dimension} pixels"
        }

    return {
        "valid": True,
        "width": w,
        "height": h,
        "size_bytes": len(image_bytes)
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def chess_square_to_coords(square: str) -> Tuple[int, int]:
    """Convert chess square notation (e.g., 'e4') to matrix coordinates"""
    if len(square) != 2:
        raise ValueError("Invalid square notation")

    file_char, rank_char = square[0].lower(), square[1]

    file_idx = ord(file_char) - ord('a')  # a=0, b=1, ..., h=7
    rank_idx = 8 - int(rank_char)  # 8=0, 7=1, ..., 1=7

    return rank_idx, file_idx


def coords_to_chess_square(rank_idx: int, file_idx: int) -> str:
    """Convert matrix coordinates to chess square notation"""
    if not (0 <= rank_idx < 8 and 0 <= file_idx < 8):
        raise ValueError("Invalid coordinates")

    file_char = chr(ord('a') + file_idx)
    rank_char = str(8 - rank_idx)

    return file_char + rank_char


def get_default_chess_position() -> List[List[str]]:
    """Get the standard starting chess position as a board matrix"""
    return FENValidator.fen_to_board_matrix("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")