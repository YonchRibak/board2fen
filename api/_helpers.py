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

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration
from config import settings

# Import your existing chess detection components
try:
    from board_detector.chess_board_detector import ChessBoardDetector
    # from piece_classifier.inference import load_piece_classifier  # You'll need to create this
except ImportError as e:
    logging.warning(f"Could not import chess components: {e}")

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


@dataclass
class BoardDetectionResult:
    """Result from board detection step"""
    success: bool
    corners: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    method: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class PieceClassificationResult:
    """Result from piece classification step"""
    success: bool
    pieces: Optional[Dict[str, str]] = None  # square -> piece mapping
    confidence_scores: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None


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
    def resize_image(image: np.ndarray, max_size: int = 2048) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]

        if max(h, w) <= max_size:
            return image

        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized

    @staticmethod
    def preprocess_for_detection(image: np.ndarray) -> np.ndarray:
        """Preprocess image for board detection"""
        # Resize if too large
        image = ImageProcessor.resize_image(image, max_size=1600)

        # Basic noise reduction
        image = cv2.bilateralFilter(image, 9, 75, 75)

        return image


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
# CHESS PIPELINE SERVICE
# ============================================================================

class ChessPipelineService:
    """Main service that orchestrates the chess board to FEN conversion pipeline"""

    def __init__(self, model_path: str):
        """Initialize the chess pipeline with model path"""
        self.model_path = Path(model_path)
        self.board_detector = None
        self.piece_classifier = None
        self.model_loaded = False

        self._load_models()

    def _load_models(self):
        """Load the trained models"""
        try:
            # Initialize board detector
            self.board_detector = ChessBoardDetector(debug=False)
            logger.info("✅ Chess board detector loaded")

            # Load piece classifier model
            if self.model_path.exists():
                self.piece_classifier = tf.keras.models.load_model(str(self.model_path))
                logger.info(f"✅ Piece classifier loaded from {self.model_path}")
            else:
                logger.error(f"❌ Model file not found: {self.model_path}")
                return

            self.model_loaded = True

        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            self.model_loaded = False

    def predict_from_image(self, image_bytes: bytes) -> PredictionResult:
        """
        Main prediction method: convert image bytes to FEN notation

        Pipeline:
        1. Load and preprocess image
        2. Detect chess board
        3. Extract and classify pieces
        4. Generate FEN notation
        """

        if not self.model_loaded:
            return PredictionResult(
                success=False,
                error_message="Models not loaded properly"
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

            image = ImageProcessor.preprocess_for_detection(image)
            processing_steps['image_loading'] = time.time() - step_start

            # Step 2: Detect chess board
            step_start = time.time()
            board_result = self._detect_board(image)
            processing_steps['board_detection'] = time.time() - step_start

            if not board_result.success:
                return PredictionResult(
                    success=False,
                    board_detected=False,
                    error_message=board_result.error_message,
                    processing_steps=processing_steps
                )

            # Step 3: Extract and classify pieces
            step_start = time.time()
            piece_result = self._classify_pieces(image, board_result.corners)
            processing_steps['piece_classification'] = time.time() - step_start

            if not piece_result.success:
                return PredictionResult(
                    success=False,
                    board_detected=True,
                    error_message=piece_result.error_message,
                    processing_steps=processing_steps
                )

            # Step 4: Generate FEN notation
            step_start = time.time()
            board_matrix = self._pieces_to_matrix(piece_result.pieces)
            fen = FENValidator.board_matrix_to_fen(board_matrix)
            processing_steps['fen_generation'] = time.time() - step_start

            if fen is None:
                return PredictionResult(
                    success=False,
                    board_detected=True,
                    error_message="Failed to generate valid FEN",
                    processing_steps=processing_steps
                )

            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                board_result.confidence,
                piece_result.confidence_scores
            )

            processing_steps['total_time'] = time.time() - start_time

            return PredictionResult(
                success=True,
                fen=fen,
                board_matrix=board_matrix,
                confidence=confidence,
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

    def _detect_board(self, image: np.ndarray) -> BoardDetectionResult:
        """Detect chess board in the image"""
        try:
            detection = self.board_detector.detect_board(image)

            if detection is None:
                return BoardDetectionResult(
                    success=False,
                    error_message="No chess board detected in image"
                )

            return BoardDetectionResult(
                success=True,
                corners=detection.corners,
                confidence=detection.confidence,
                method=detection.method
            )

        except Exception as e:
            return BoardDetectionResult(
                success=False,
                error_message=f"Board detection error: {str(e)}"
            )

    def _classify_pieces(self, image: np.ndarray, corners: np.ndarray) -> PieceClassificationResult:
        """Extract and classify pieces from the detected board"""
        try:
            # Step 1: Perform perspective correction
            corrected_board = self._correct_perspective(image, corners)

            # Step 2: Divide into 64 squares
            squares = self._divide_into_squares(corrected_board)

            # Step 3: Detect which squares have pieces
            occupied_squares = self._detect_occupied_squares(squares)

            # Step 4: Classify pieces in occupied squares
            pieces = {}
            confidence_scores = {}

            for square_name in occupied_squares:
                square_image = squares[square_name]
                piece_prediction = self._classify_single_piece(square_image)

                if piece_prediction:
                    pieces[square_name] = piece_prediction['piece']
                    confidence_scores[square_name] = piece_prediction['confidence']

            return PieceClassificationResult(
                success=True,
                pieces=pieces,
                confidence_scores=confidence_scores
            )

        except Exception as e:
            return PieceClassificationResult(
                success=False,
                error_message=f"Piece classification error: {str(e)}"
            )

    def _correct_perspective(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Apply perspective correction to get top-down view of the board"""
        # Target size for corrected board
        board_size = 640

        # Define destination corners (perfect square)
        dst_corners = np.array([
            [0, 0],
            [board_size, 0],
            [board_size, board_size],
            [0, board_size]
        ], dtype=np.float32)

        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)

        # Apply transformation
        corrected = cv2.warpPerspective(image, transform_matrix, (board_size, board_size))

        return corrected

    def _divide_into_squares(self, board_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Divide the corrected board into 64 individual squares"""
        squares = {}
        h, w = board_image.shape[:2]
        square_h, square_w = h // 8, w // 8

        for rank in range(8):
            for file in range(8):
                # Calculate square coordinates
                y1 = rank * square_h
                y2 = (rank + 1) * square_h
                x1 = file * square_w
                x2 = (file + 1) * square_w

                # Extract square
                square_image = board_image[y1:y2, x1:x2]

                # Convert coordinates to chess notation
                file_char = chr(ord('a') + file)
                rank_char = str(8 - rank)
                square_name = file_char + rank_char

                squares[square_name] = square_image

        return squares

    def _detect_occupied_squares(self, squares: Dict[str, np.ndarray]) -> List[str]:
        """Detect which squares contain pieces (simple approach)"""
        occupied = []

        for square_name, square_image in squares.items():
            # Simple occupancy detection based on color variance
            gray = cv2.cvtColor(square_image, cv2.COLOR_RGB2GRAY)

            # Calculate variance - pieces usually create higher variance
            variance = np.var(gray)

            # Threshold for piece detection (you may need to tune this)
            if variance > 200:  # Adjust based on your data
                occupied.append(square_name)

        return occupied

    def _classify_single_piece(self, square_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Classify a single piece using the trained model"""
        try:
            # Preprocess image for the model
            processed_image = self._preprocess_piece_image(square_image)

            # Make prediction
            prediction = self.piece_classifier.predict(np.expand_dims(processed_image, axis=0), verbose=0)

            # Get piece classes from settings
            piece_classes = settings.piece_classes

            # Get prediction results
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])

            # Convert class name to FEN piece symbol
            piece_name = piece_classes[predicted_class_idx]
            piece_symbol = self._piece_name_to_symbol(piece_name)

            return {
                'piece': piece_symbol,
                'confidence': confidence,
                'class_name': piece_name
            }

        except Exception as e:
            logger.error(f"Error classifying piece: {e}")
            return None

    def _preprocess_piece_image(self, square_image: np.ndarray) -> np.ndarray:
        """Preprocess piece image for the classifier model"""
        # Resize to model input size from settings
        resized = cv2.resize(square_image, settings.model_input_size)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        return normalized

    def _piece_name_to_symbol(self, piece_name: str) -> str:
        """Convert piece class name to FEN symbol"""
        name_to_symbol = {
            'white_king': 'K', 'white_queen': 'Q', 'white_rook': 'R',
            'white_bishop': 'B', 'white_knight': 'N', 'white_pawn': 'P',
            'black_king': 'k', 'black_queen': 'q', 'black_rook': 'r',
            'black_bishop': 'b', 'black_knight': 'n', 'black_pawn': 'p'
        }
        return name_to_symbol.get(piece_name, '?')

    def _pieces_to_matrix(self, pieces: Dict[str, str]) -> List[List[str]]:
        """Convert piece dictionary to 8x8 board matrix"""
        matrix = [[''] * 8 for _ in range(8)]

        for square, piece in pieces.items():
            if len(square) == 2:
                file_char, rank_char = square[0], square[1]
                file_idx = ord(file_char.lower()) - ord('a')  # a=0, b=1, ..., h=7
                rank_idx = 8 - int(rank_char)  # 8=0, 7=1, ..., 1=7

                if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
                    matrix[rank_idx][file_idx] = piece

        return matrix

    def _calculate_overall_confidence(self,
                                      board_confidence: float,
                                      piece_confidences: Dict[str, float]) -> float:
        """Calculate overall prediction confidence"""
        if not piece_confidences:
            return board_confidence * 0.5  # Low confidence if no pieces detected

        avg_piece_confidence = np.mean(list(piece_confidences.values()))

        # Weighted average: 30% board detection, 70% piece classification
        overall_confidence = 0.3 * board_confidence + 0.7 * avg_piece_confidence

        return min(overall_confidence, 1.0)


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