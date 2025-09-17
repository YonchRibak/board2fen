#!/usr/bin/env python3
"""
DEBUG SCRIPT: ResNeXt Chess Model Analysis
===========================================

This script provides comprehensive debugging for the ResNeXt chess prediction model,
focusing on identifying why predictions are failing.

Usage:
    python debug_resnext_model.py                    # Run synthetic tests only
    python debug_resnext_model.py path/to/image.jpg  # Test with real image
    python debug_resnext_model.py --full path/to/image.jpg  # Run both synthetic and real image tests

Key investigations:
1. Model loading and architecture analysis
2. Input preprocessing validation
3. Model prediction testing with various inputs
4. Output format analysis and conversion
5. Configuration compatibility check
6. Real image testing (optional)
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from PIL import Image
import requests
import tempfile
import logging
import traceback
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from api.config import settings
    from api._helpers import ImageProcessor, ModelDownloader

    CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import config: {e}")
    CONFIG_AVAILABLE = False


class ResNeXtModelDebugger:
    """Comprehensive debugger for ResNeXt chess model"""

    def __init__(self, model_url=None):
        self.model_url = model_url or "https://storage.googleapis.com/chess_board_cllassification_model/one_epoch_resnext_thorough.keras"
        self.model = None
        self.piece_classes = [
            "empty",
            "black-bishop", "black-king", "black-knight", "black-pawn",
            "black-queen", "black-rook", "white-bishop", "white-king",
            "white-knight", "white-pawn", "white-queen", "white-rook",
        ]

        # FIXED: ResNeXt model expects 224x224, not 256x256
        self.expected_input_size = (224, 224)
        self.expected_output_shape = (64, 13)  # 64 squares, 13 classes

    def download_and_load_model(self):
        """Download and load the model, with detailed error reporting"""
        print("=" * 70)
        print("🔄 DOWNLOADING AND LOADING MODEL")
        print("=" * 70)

        try:
            # Create temporary directory for model
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = Path(temp_dir) / "debug_model.keras"

                print(f"📥 Downloading model from: {self.model_url}")
                response = requests.get(self.model_url, timeout=300, stream=True)
                response.raise_for_status()

                # Download with progress
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0

                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                if downloaded_size % (1024 * 1024 * 5) == 0:  # Every 5MB
                                    print(f"   Progress: {progress:.1f}% ({downloaded_size // (1024 * 1024)}MB)")

                file_size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"✅ Download complete: {file_size_mb:.1f}MB")

                # Load the model
                print("🧠 Loading model...")
                self.model = tf.keras.models.load_model(str(model_path))
                print("✅ Model loaded successfully!")

                return True

        except Exception as e:
            print(f"❌ Failed to download/load model: {e}")
            traceback.print_exc()
            return False

    def analyze_model_architecture(self):
        """Analyze the loaded model's architecture"""
        print("\n" + "=" * 70)
        print("🏗️ MODEL ARCHITECTURE ANALYSIS")
        print("=" * 70)

        if not self.model:
            print("❌ No model loaded")
            return

        try:
            print(f"📊 Model Summary:")
            print(f"   Input Shape: {self.model.input_shape}")
            print(f"   Output Shape: {self.model.output_shape}")
            print(f"   Total Parameters: {self.model.count_params():,}")

            # Expected configuration - FIXED for ResNeXt
            expected_input = (None, self.expected_input_size[0], self.expected_input_size[1], 3)
            expected_output = (None, self.expected_output_shape[0] * self.expected_output_shape[1])

            print(f"\n🎯 Expected vs Actual:")
            print(f"   Expected Input: {expected_input}")
            print(f"   Actual Input:   {self.model.input_shape}")
            print(f"   Input Match: {'✅' if self.model.input_shape == expected_input else '❌'}")

            print(f"   Expected Output: {expected_output}")
            print(f"   Actual Output:   {self.model.output_shape}")
            print(f"   Output Match: {'✅' if self.model.output_shape == expected_output else '❌'}")

            # Detailed layer analysis
            print(f"\n🔍 Layer Analysis:")
            print(f"   Total Layers: {len(self.model.layers)}")

            # Show first few and last few layers
            print(f"   First 3 layers:")
            for i, layer in enumerate(self.model.layers[:3]):
                try:
                    if hasattr(layer, 'output_shape'):
                        shape = layer.output_shape
                    elif hasattr(layer, 'output'):
                        shape = layer.output.shape if hasattr(layer.output, 'shape') else 'Unknown'
                    else:
                        shape = 'N/A'
                    print(f"     {i + 1}. {layer.name} ({layer.__class__.__name__}) - {shape}")
                except Exception as e:
                    print(f"     {i + 1}. {layer.name} ({layer.__class__.__name__}) - Error: {e}")

            print(f"   Last 3 layers:")
            for i, layer in enumerate(self.model.layers[-3:]):
                try:
                    idx = len(self.model.layers) - 3 + i + 1
                    if hasattr(layer, 'output_shape'):
                        shape = layer.output_shape
                    elif hasattr(layer, 'output'):
                        shape = layer.output.shape if hasattr(layer.output, 'shape') else 'Unknown'
                    else:
                        shape = 'N/A'
                    print(f"     {idx}. {layer.name} ({layer.__class__.__name__}) - {shape}")
                except Exception as e:
                    idx = len(self.model.layers) - 3 + i + 1
                    print(f"     {idx}. {layer.name} ({layer.__class__.__name__}) - Error: {e}")

            # Check for common ResNeXt patterns
            resnext_layers = [layer for layer in self.model.layers if
                              'res' in layer.name.lower() or 'next' in layer.name.lower()]
            print(f"   ResNeXt-specific layers found: {len(resnext_layers)}")

        except Exception as e:
            print(f"❌ Error analyzing architecture: {e}")
            traceback.print_exc()

    def test_model_with_synthetic_inputs(self):
        """Test model with various synthetic inputs to understand behavior"""
        print("\n" + "=" * 70)
        print("🧪 SYNTHETIC INPUT TESTING")
        print("=" * 70)

        if not self.model:
            print("❌ No model loaded")
            return

        test_cases = [
            ("Random noise", np.random.random((1, 224, 224, 3)).astype(np.float32)),
            ("Zeros", np.zeros((1, 224, 224, 3), dtype=np.float32)),
            ("Ones", np.ones((1, 224, 224, 3), dtype=np.float32)),
            ("0.5 gray", np.full((1, 224, 224, 3), 0.5, dtype=np.float32)),
            ("Normalized random", np.random.normal(0.5, 0.1, (1, 224, 224, 3)).astype(np.float32)),
        ]

        for test_name, test_input in test_cases:
            try:
                print(f"\n🔬 Testing with {test_name}:")
                print(f"   Input shape: {test_input.shape}")
                print(f"   Input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
                print(f"   Input mean: {test_input.mean():.4f}")

                # Run prediction
                prediction = self.model.predict(test_input, verbose=0)

                print(f"   ✅ Prediction successful!")
                print(f"   Output shape: {prediction.shape}")
                print(f"   Output range: [{prediction.min():.4f}, {prediction.max():.4f}]")
                print(f"   Output mean: {prediction.mean():.4f}")
                print(f"   Output std: {prediction.std():.4f}")

                # Analyze the prediction structure
                self._analyze_prediction_output(prediction, test_name)

            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
                traceback.print_exc()

    def _analyze_prediction_output(self, prediction, test_name):
        """Analyze the structure of model predictions"""
        try:
            # Reshape prediction to expected format
            if prediction.shape[1] == 832:  # 64 * 13
                reshaped = prediction.reshape(1, 64, 13)
                print(f"   📊 Reshaped to (1, 64, 13) for analysis")

                # Analyze first few squares
                print(f"   🎯 First 5 squares analysis:")
                for square_idx in range(5):
                    square_probs = reshaped[0, square_idx]
                    predicted_class = np.argmax(square_probs)
                    confidence = square_probs[predicted_class]

                    square_name = f"{chr(ord('a') + (square_idx % 8))}{8 - (square_idx // 8)}"
                    piece_name = self.piece_classes[predicted_class] if predicted_class < len(
                        self.piece_classes) else "unknown"

                    print(f"     {square_name}: {piece_name} (conf: {confidence:.4f})")

                # Count confident predictions
                confident_pieces = 0
                confident_empty = 0

                for square_idx in range(64):
                    square_probs = reshaped[0, square_idx]
                    predicted_class = np.argmax(square_probs)
                    confidence = square_probs[predicted_class]

                    piece_name = self.piece_classes[predicted_class] if predicted_class < len(
                        self.piece_classes) else "unknown"

                    if piece_name.lower() == 'empty':
                        if confidence > 0.5:
                            confident_empty += 1
                    else:
                        if confidence > 0.3:
                            confident_pieces += 1

                print(f"   📈 Confidence analysis:")
                print(f"     Confident pieces (>0.3): {confident_pieces}")
                print(f"     Confident empty (>0.5): {confident_empty}")
                print(f"     Total confident: {confident_pieces + confident_empty}/64")

        except Exception as e:
            print(f"   ❌ Error analyzing prediction: {e}")

    def test_preprocessing_pipeline(self):
        """Test the image preprocessing pipeline"""
        print("\n" + "=" * 70)
        print("🖼️ PREPROCESSING PIPELINE TEST")
        print("=" * 70)

        try:
            # Create a test image
            test_image = self._create_test_chess_image()
            print(f"📸 Created test image: {test_image.shape}")

            # Test preprocessing steps
            if CONFIG_AVAILABLE:
                processed = ImageProcessor.preprocess_for_model(test_image, target_size=(224, 224))
                print(f"✅ Preprocessing successful")
                print(f"   Processed shape: {processed.shape}")
                print(f"   Processed range: [{processed.min():.4f}, {processed.max():.4f}]")
                print(f"   Processed mean: {processed.mean():.4f}")

                # Test with model if available
                if self.model:
                    batch_input = np.expand_dims(processed, axis=0)
                    prediction = self.model.predict(batch_input, verbose=0)
                    print(f"✅ Model prediction with preprocessed image successful")
                    self._analyze_prediction_output(prediction, "test_chess_image")
            else:
                # Manual preprocessing
                print("⚠️ Using manual preprocessing (config not available)")
                resized = cv2.resize(test_image, (224, 224), interpolation=cv2.INTER_LINEAR)
                normalized = resized.astype(np.float32) / 255.0

                print(f"   Manual processed shape: {normalized.shape}")
                print(f"   Manual processed range: [{normalized.min():.4f}, {normalized.max():.4f}]")

                if self.model:
                    batch_input = np.expand_dims(normalized, axis=0)
                    prediction = self.model.predict(batch_input, verbose=0)
                    print(f"✅ Model prediction with manual preprocessing successful")
                    self._analyze_prediction_output(prediction, "manual_preprocessed")

        except Exception as e:
            print(f"❌ Preprocessing test failed: {e}")
            traceback.print_exc()

    def _create_test_chess_image(self):
        """Create a synthetic chess board image for testing"""
        # Create a simple 8x8 checkerboard pattern
        board_size = 512
        square_size = board_size // 8

        image = np.zeros((board_size, board_size, 3), dtype=np.uint8)

        # Create alternating pattern
        for row in range(8):
            for col in range(8):
                y1 = row * square_size
                y2 = (row + 1) * square_size
                x1 = col * square_size
                x2 = (col + 1) * square_size

                # Alternate between light and dark squares
                if (row + col) % 2 == 0:
                    color = [240, 217, 181]  # Light square
                else:
                    color = [181, 136, 99]  # Dark square

                image[y1:y2, x1:x2] = color

        return image

    def test_output_conversion(self):
        """Test the conversion from model output to board matrix and FEN"""
        print("\n" + "=" * 70)
        print("🔄 OUTPUT CONVERSION TESTING")
        print("=" * 70)

        if not self.model:
            print("❌ No model loaded")
            return

        try:
            # Create a test input with correct size for ResNeXt
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            prediction = self.model.predict(test_input, verbose=0)

            print(f"🔍 Testing output conversion logic:")
            print(f"   Raw prediction shape: {prediction.shape}")

            # Test the conversion logic from your helper file
            board_matrix, confidence = self._convert_flattened_output(prediction[0])

            print(f"   ✅ Conversion successful")
            print(f"   Board matrix created: {type(board_matrix)}")
            print(f"   Average confidence: {confidence:.4f}")

            # Count pieces on board
            piece_count = sum(1 for row in board_matrix for cell in row if cell != '')
            print(f"   Pieces on board: {piece_count}/64")

            # Try to create FEN
            from api._helpers import FENValidator
            fen = FENValidator.board_matrix_to_fen(board_matrix)

            if fen:
                print(f"   ✅ FEN generation successful: {fen[:50]}...")

                # Validate FEN
                is_valid = FENValidator.validate_fen(fen)
                print(f"   FEN valid: {'✅' if is_valid else '❌'}")
            else:
                print(f"   ❌ FEN generation failed")

        except Exception as e:
            print(f"❌ Output conversion test failed: {e}")
            traceback.print_exc()

    def _convert_flattened_output(self, output):
        """Convert flattened model output to board matrix (copied from _helpers.py)"""
        board_matrix = [[''] * 8 for _ in range(8)]
        confidences = []

        for i in range(64):
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

                # Use lower thresholds for debugging
                if piece_symbol != '':  # It's a piece
                    if confidence > 0.1:  # Lower threshold for debugging
                        board_matrix[rank][file] = piece_symbol
                        confidences.append(confidence)
                else:  # It's empty
                    if piece_name.lower() == 'empty' and confidence > 0.1:
                        confidences.append(confidence)

        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        return board_matrix, avg_confidence

    def _piece_name_to_symbol(self, piece_name):
        """Convert piece class name to FEN symbol"""
        name_to_symbol = {
            'empty': '', 'none': '', 'blank': '', '': '',
            'white-king': 'K', 'white-queen': 'Q', 'white-rook': 'R',
            'white-bishop': 'B', 'white-knight': 'N', 'white-pawn': 'P',
            'black-king': 'k', 'black-queen': 'q', 'black-rook': 'r',
            'black-bishop': 'b', 'black-knight': 'n', 'black-pawn': 'p',
        }
        return name_to_symbol.get(piece_name.lower(), '')

    def test_real_image(self, image_path: str):
        """Test the model with a real chess board image"""
        print("\n" + "=" * 70)
        print("🖼️ REAL IMAGE TESTING")
        print("=" * 70)

        if not self.model:
            print("❌ No model loaded")
            return

        # Validate image path
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"❌ Image file not found: {image_path}")
            return

        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"❌ Unsupported image format: {image_path.suffix}")
            return

        try:
            print(f"📸 Loading image: {image_path}")

            # Load image
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image = np.array(pil_image)

            print(f"   Original image shape: {image.shape}")
            print(f"   Original image range: [{image.min()}, {image.max()}]")

            # Preprocess image
            resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            normalized = resized.astype(np.float32) / 255.0
            batch_input = np.expand_dims(normalized, axis=0)

            print(f"   Preprocessed shape: {batch_input.shape}")
            print(f"   Preprocessed range: [{normalized.min():.4f}, {normalized.max():.4f}]")
            print(f"   Preprocessed mean: {normalized.mean():.4f}")

            # Run prediction
            print(f"\n🔮 Running prediction...")
            prediction = self.model.predict(batch_input, verbose=0)

            print(f"✅ Prediction successful!")
            print(f"   Output shape: {prediction.shape}")
            print(f"   Output range: [{prediction.min():.4f}, {prediction.max():.4f}]")

            # Convert to board matrix and FEN
            board_matrix, confidence = self._convert_flattened_output(prediction[0])
            pieces_count = sum(1 for row in board_matrix for cell in row if cell != '')

            print(f"\n♟️ Chess Board Analysis:")
            print(f"   Pieces detected: {pieces_count}/32")
            print(f"   Average confidence: {confidence:.4f}")

            # Display board
            print(f"\n🏁 Predicted Board Position:")
            self._display_board_matrix(board_matrix)

            # Generate FEN
            fen = self._board_matrix_to_fen(board_matrix)
            if fen:
                print(f"\n📝 Generated FEN:")
                print(f"   {fen}")

                # Validate FEN
                try:
                    import chess
                    board = chess.Board(fen)
                    print(f"   ✅ FEN is valid")
                    print(f"   Turn: {'White' if board.turn else 'Black'}")
                    print(f"   Castling rights: {board.castling_rights}")
                except:
                    print(f"   ❌ Generated FEN is invalid")
            else:
                print(f"   ❌ Failed to generate FEN")

            # Detailed square analysis
            print(f"\n🔍 Top 10 Most Confident Predictions:")
            self._show_top_predictions(prediction[0])

            return True

        except Exception as e:
            print(f"❌ Real image test failed: {e}")
            traceback.print_exc()
            return False

    def _display_board_matrix(self, board_matrix):
        """Display the board matrix in a chess-like format"""
        files = "  a b c d e f g h"
        print(files)
        for rank in range(8):
            rank_num = 8 - rank
            row_display = f"{rank_num} "
            for file in range(8):
                piece = board_matrix[rank][file]
                if piece == '':
                    piece = '.'
                row_display += f"{piece} "
            row_display += f"{rank_num}"
            print(row_display)
        print(files)

    def _show_top_predictions(self, output):
        """Show the top confident predictions across all squares"""
        predictions = []

        for i in range(64):
            class_probs = output[i]
            predicted_class_idx = np.argmax(class_probs)
            confidence = float(class_probs[predicted_class_idx])

            # Convert to chess square notation
            rank = i // 8
            file = i % 8
            square = f"{chr(ord('a') + file)}{8 - rank}"

            piece_name = self.piece_classes[predicted_class_idx] if predicted_class_idx < len(
                self.piece_classes) else "unknown"
            piece_symbol = self._piece_name_to_symbol(piece_name)

            predictions.append({
                'square': square,
                'piece_name': piece_name,
                'piece_symbol': piece_symbol if piece_symbol else 'empty',
                'confidence': confidence
            })

        # Sort by confidence and show top 10
        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        for i, pred in enumerate(predictions[:10]):
            symbol_display = pred['piece_symbol'] if pred['piece_symbol'] != 'empty' else '.'
            print(
                f"   {i + 1:2d}. {pred['square']}: {symbol_display:>6} ({pred['piece_name']:>12}) - {pred['confidence']:.4f}")

    def _board_matrix_to_fen(self, matrix):
        """Convert board matrix to FEN notation"""
        try:
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
            fen = f"{board_position} w KQkq - 0 1"
            return fen
        except Exception as e:
            print(f"Error converting to FEN: {e}")
            return None

    def run_comprehensive_debug(self, skip_synthetic=False):
        """Run all debug tests with option to skip synthetic tests"""
        print("🚀 STARTING COMPREHENSIVE RESNEXT MODEL DEBUG")
        print("=" * 70)

        # Step 1: Download and load model
        if not self.download_and_load_model():
            print("❌ Cannot continue without model")
            return

        # Step 2: Analyze architecture
        self.analyze_model_architecture()

        if not skip_synthetic:
            # Step 3: Test with synthetic inputs
            self.test_model_with_synthetic_inputs()

            # Step 4: Test preprocessing
            self.test_preprocessing_pipeline()

            # Step 5: Test output conversion
            self.test_output_conversion()

        print("\n" + "=" * 70)
        print("✅ COMPREHENSIVE DEBUG COMPLETE")
        print("=" * 70)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Debug ResNeXt Chess Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug_resnext_model.py                           # Run synthetic tests only
  python debug_resnxt_model.py image.jpg                  # Test with real image only  
  python debug_resnext_model.py --full image.jpg          # Run both synthetic and real image tests
  python debug_resnext_model.py --synthetic               # Run synthetic tests only (explicit)
        """
    )

    parser.add_argument('image', nargs='?', help='Path to chess board image to test')
    parser.add_argument('--full', action='store_true', help='Run both synthetic and real image tests')
    parser.add_argument('--synthetic', action='store_true', help='Run synthetic tests only')
    parser.add_argument('--model-url', help='Custom model URL to test')

    return parser.parse_args()


def main():
    """Main debug execution with command line argument support"""
    args = parse_arguments()

    print("ResNeXt Chess Model Debugger")
    print("=" * 70)

    # Initialize debugger
    debugger = ResNeXtModelDebugger(model_url=args.model_url)

    # Determine what tests to run
    if args.image and not args.full:
        # Real image test only
        print(f"🖼️ Testing with real image: {args.image}")
        success = debugger.download_and_load_model()
        if success:
            debugger.analyze_model_architecture()
            debugger.test_real_image(args.image)

    elif args.image and args.full:
        # Both synthetic and real image tests
        print(f"🔬 Running full debug suite + real image test: {args.image}")
        debugger.run_comprehensive_debug()
        debugger.test_real_image(args.image)

    elif args.synthetic or not args.image:
        # Synthetic tests only (default if no image provided)
        print("🧪 Running synthetic tests only")
        debugger.run_comprehensive_debug()

    # Additional manual testing suggestions
    print("\n🔧 MANUAL TESTING SUGGESTIONS:")
    if not args.image:
        print("1. Test with a real chess board image using: python debug_resnext_model.py path/to/image.jpg")
    print("2. Check the model architecture matches training expectations")
    print("3. Verify input preprocessing matches training pipeline")
    print("4. Check if confidence thresholds need adjustment")
    print("5. Validate piece class names match model training labels")

    if args.image:
        print("6. Try different chess board images with various angles and lighting")
        print("7. Compare results with the multi-model pipeline service")
        print("8. Test with both clear and challenging board positions")


if __name__ == "__main__":
    main()