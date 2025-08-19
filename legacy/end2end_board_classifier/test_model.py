# test_chess_model.py
# Test trained chess CNN model on images and visualize predictions

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import chess
from pathlib import Path
import argparse
from typing import Optional, Tuple
import tkinter as tk
from tkinter import filedialog
import matplotlib.patheffects as path_effects

class ChessModelTester:
    """Test trained chess position prediction model"""

    def __init__(self, model_path: str, dataset_dir: str = None):
        print("🔧 Loading chess position prediction model...")

        # Load trained model
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")

        # Load metadata if available
        self.piece_mapping = self._load_piece_mapping(dataset_dir)

        # Define piece symbols for visualization
        self.piece_symbols = {
            0: '.',  # Empty
            1: 'P', 2: 'R', 3: 'N', 4: 'B', 5: 'Q', 6: 'K',  # White pieces
            7: 'p', 8: 'r', 9: 'n', 10: 'b', 11: 'q', 12: 'k'  # Black pieces
        }

        self.piece_names = {
            0: 'empty',
            1: 'white_pawn', 2: 'white_rook', 3: 'white_knight', 4: 'white_bishop', 5: 'white_queen', 6: 'white_king',
            7: 'black_pawn', 8: 'black_rook', 9: 'black_knight', 10: 'black_bishop', 11: 'black_queen', 12: 'black_king'
        }

        print(f"🎯 Model expects input shape: {self.model.input_shape}")
        print(f"📊 Model output shape: {self.model.output_shape}")

    def _load_piece_mapping(self, dataset_dir: str) -> dict:
        """Load piece mapping from dataset metadata"""
        if dataset_dir and Path(dataset_dir).exists():
            metadata_path = Path(dataset_dir) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata.get('piece_mapping', {})
        return {}

    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess image for model prediction"""

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image = cv2.resize(image, target_size)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def predict_position(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict chess position from image

        Returns:
            predicted_grid: 8x8 array with predicted piece classes
            confidence_scores: 8x8 array with prediction confidences
            raw_predictions: 8x8x13 array with raw logits
        """

        print(f"🔍 Analyzing image: {Path(image_path).name}")

        # Preprocess image
        processed_image = self.preprocess_image(image_path)

        # Make prediction
        raw_predictions = self.model.predict(processed_image, verbose=0)[0]  # Remove batch dimension

        # Get predicted classes and confidence scores
        predicted_grid = np.argmax(raw_predictions, axis=-1)
        confidence_scores = np.max(raw_predictions, axis=-1)

        return predicted_grid, confidence_scores, raw_predictions

    def grid_to_fen(self, predicted_grid: np.ndarray) -> str:
        """Convert predicted grid to FEN notation"""

        try:
            # Create empty chess board
            board = chess.Board(fen=None)
            board.clear_board()

            # Place pieces based on predictions
            for row in range(8):
                for col in range(8):
                    piece_class = predicted_grid[row, col]

                    if piece_class > 0:  # Not empty
                        # Convert grid coordinates to chess square
                        rank = 7 - row  # Flip rank (grid row 0 = rank 8)
                        file = col  # File stays same
                        square = chess.square(file, rank)

                        # Get piece symbol
                        piece_symbol = self.piece_symbols.get(piece_class, None)
                        if piece_symbol and piece_symbol != '.':
                            piece = chess.Piece.from_symbol(piece_symbol)
                            board.set_piece_at(square, piece)

            return board.fen()

        except Exception as e:
            print(f"Warning: Could not generate FEN - {e}")
            return "Invalid position"

    def visualize_prediction(self, image_path: str, predicted_grid: np.ndarray,
                             confidence_scores: np.ndarray, save_path: str = None):
        """Visualize image and predicted chess position"""

        # Load original image for display
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Original image
        axes[0].imshow(original_image)
        axes[0].set_title(f"Original Image\n{Path(image_path).name}", fontsize=12)
        axes[0].axis('off')

        # 2. Predicted position as chess board
        self._draw_chess_board(axes[1], predicted_grid, "Predicted Position")

        # 3. Confidence heatmap
        im = axes[2].imshow(confidence_scores, cmap='RdYlGn', vmin=0, vmax=1)
        axes[2].set_title("Prediction Confidence", fontsize=12)
        axes[2].set_xticks(range(8))
        axes[2].set_yticks(range(8))
        axes[2].set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        axes[2].set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])

        # Add confidence values as text
        for row in range(8):
            for col in range(8):
                conf = confidence_scores[row, col]
                color = 'white' if conf < 0.5 else 'black'
                axes[2].text(col, row, f'{conf:.2f}', ha='center', va='center',
                             color=color, fontsize=8, fontweight='bold')

        # Add colorbar for confidence
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")

        plt.show()

    def _draw_chess_board(self, ax, grid: np.ndarray, title: str):
        """Draw chess board with pieces - FIXED VERSION"""

        # Create board background
        board_colors = np.zeros((8, 8, 3))
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    board_colors[row, col] = [0.95, 0.95, 0.85]  # Light square
                else:
                    board_colors[row, col] = [0.4, 0.2, 0.1]  # Dark square

        ax.imshow(board_colors)
        ax.set_title(title, fontsize=12)

        # Add pieces
        for row in range(8):
            for col in range(8):
                piece_class = grid[row, col]
                if piece_class > 0:  # Not empty
                    symbol = self.piece_symbols.get(piece_class, '?')

                    # Choose text color based on piece color
                    if 1 <= piece_class <= 6:  # White pieces
                        text_color = 'white'
                        outline_color = 'black'
                    else:  # Black pieces
                        text_color = 'black'
                        outline_color = 'white'

                    # Add piece symbol with outline for better visibility
                    # FIXED: Use proper import
                    ax.text(col, row, symbol, ha='center', va='center',
                            fontsize=20, fontweight='bold', color=text_color,
                            path_effects=[path_effects.withStroke(linewidth=2, foreground=outline_color)])

        # Set proper ticks and labels
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])

        # Add grid lines
        for x in range(9):
            ax.axhline(x - 0.5, color='black', linewidth=1)
            ax.axvline(x - 0.5, color='black', linewidth=1)

    def analyze_prediction_quality(self, predicted_grid: np.ndarray, confidence_scores: np.ndarray):
        """Analyze prediction quality and provide insights"""

        print(f"\n📊 PREDICTION ANALYSIS")
        print("=" * 40)

        # Basic statistics
        total_squares = 64
        occupied_squares = np.sum(predicted_grid > 0)
        empty_squares = total_squares - occupied_squares

        print(f"🎯 Position Summary:")
        print(f"   Empty squares: {empty_squares}/64 ({empty_squares / 64 * 100:.1f}%)")
        print(f"   Occupied squares: {occupied_squares}/64 ({occupied_squares / 64 * 100:.1f}%)")

        # Count pieces by type
        piece_counts = {}
        for piece_class in range(1, 13):
            count = np.sum(predicted_grid == piece_class)
            if count > 0:
                piece_name = self.piece_names.get(piece_class, f"class_{piece_class}")
                piece_counts[piece_name] = count

        if piece_counts:
            print(f"\n♟️  Detected Pieces:")
            for piece, count in sorted(piece_counts.items()):
                print(f"   {piece}: {count}")

        # Confidence analysis
        avg_confidence = np.mean(confidence_scores)
        min_confidence = np.min(confidence_scores)
        max_confidence = np.max(confidence_scores)

        print(f"\n🎯 Confidence Analysis:")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Min confidence: {min_confidence:.3f}")
        print(f"   Max confidence: {max_confidence:.3f}")

        # Find low confidence squares
        low_confidence_threshold = 0.5
        low_conf_squares = np.where(confidence_scores < low_confidence_threshold)

        if len(low_conf_squares[0]) > 0:
            print(f"\n⚠️  Low confidence squares (< {low_confidence_threshold}):")
            for row, col in zip(low_conf_squares[0], low_conf_squares[1]):
                file = chr(ord('a') + col)
                rank = 8 - row
                conf = confidence_scores[row, col]
                predicted_piece = self.piece_names.get(predicted_grid[row, col], "unknown")
                print(f"   {file}{rank}: {predicted_piece} (conf: {conf:.3f})")

        # Position validation
        kings = np.sum(predicted_grid == 6) + np.sum(predicted_grid == 12)  # White + Black kings

        print(f"\n✅ Position Validation:")
        if kings == 2:
            print(f"   Kings: ✅ Found 2 kings")
        elif kings == 0:
            print(f"   Kings: ❌ No kings detected")
        elif kings == 1:
            print(f"   Kings: ⚠️  Only 1 king detected")
        else:
            print(f"   Kings: ❌ Too many kings ({kings})")

        # Check for reasonable piece counts
        total_pieces = occupied_squares
        if 16 <= total_pieces <= 32:
            print(f"   Piece count: ✅ Reasonable ({total_pieces} pieces)")
        elif total_pieces < 16:
            print(f"   Piece count: ⚠️  Few pieces ({total_pieces})")
        else:
            print(f"   Piece count: ⚠️  Many pieces ({total_pieces})")

    def test_single_image(self, image_path: str, save_visualization: bool = True):
        """Test model on a single image with full analysis"""

        print(f"\n🎮 TESTING CHESS POSITION PREDICTION")
        print("=" * 50)
        print(f"📸 Image: {image_path}")

        try:
            # Make prediction
            predicted_grid, confidence_scores, raw_predictions = self.predict_position(image_path)

            # Generate FEN
            predicted_fen = self.grid_to_fen(predicted_grid)
            print(f"🎯 Predicted FEN: {predicted_fen}")

            # Analyze prediction quality
            self.analyze_prediction_quality(predicted_grid, confidence_scores)

            # Visualize results
            save_path = None
            if save_visualization:
                output_dir = Path("outputs/predictions")
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"prediction_{Path(image_path).stem}.png"

            self.visualize_prediction(image_path, predicted_grid, confidence_scores, save_path)

            print(f"\n✅ Analysis complete!")

            return predicted_grid, confidence_scores, predicted_fen

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


def select_image_file() -> Optional[str]:
    """Open file dialog to select image"""

    # Hide main tkinter window
    root = tk.Tk()
    root.withdraw()

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select Chess Board Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )

    return file_path if file_path else None


def test_model_interactive(model_path: str, dataset_dir: str = None):
    """Interactive model testing"""

    # Initialize tester
    tester = ChessModelTester(model_path, dataset_dir)

    while True:
        print(f"\n🎮 CHESS POSITION PREDICTION TESTER")
        print("=" * 45)
        print("1. Select image file")
        print("2. Enter image path")
        print("3. Test sample images from dataset")
        print("4. Exit")

        choice = input("\nChoose option (1-4): ").strip()

        if choice == '1':
            image_path = select_image_file()
            if image_path:
                tester.test_single_image(image_path)
            else:
                print("No file selected.")

        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if Path(image_path).exists():
                tester.test_single_image(image_path)
            else:
                print(f"❌ File not found: {image_path}")

        elif choice == '3':
            if dataset_dir and Path(dataset_dir).exists():
                test_dataset_samples(tester, dataset_dir)
            else:
                print("❌ Dataset directory not available")

        elif choice == '4':
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please select 1-4.")


def test_dataset_samples(tester: ChessModelTester, dataset_dir: str, num_samples: int = 3):
    """Test model on sample images from the dataset"""

    dataset_path = Path(dataset_dir)

    # Load image paths
    image_paths_file = dataset_path / "image_paths.txt"
    if not image_paths_file.exists():
        print(f"❌ Image paths file not found: {image_paths_file}")
        return

    with open(image_paths_file, 'r') as f:
        all_image_paths = [line.strip() for line in f.readlines()]

    # Test random samples
    import random
    sample_paths = random.sample(all_image_paths, min(num_samples, len(all_image_paths)))

    print(f"\n🧪 Testing {len(sample_paths)} random dataset samples:")

    for i, image_path in enumerate(sample_paths):
        print(f"\n--- Sample {i + 1}/{len(sample_paths)} ---")
        if Path(image_path).exists():
            tester.test_single_image(image_path, save_visualization=True)
        else:
            print(f"❌ Sample image not found: {image_path}")

        if i < len(sample_paths) - 1:
            input("\nPress Enter to continue to next sample...")


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description='Test Chess Position Prediction Model')
    parser.add_argument('--model', type=str, default='outputs/chess_cnn_quick/final_chess_cnn.keras',
                        help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='C:/datasets/ChessRender360_GridLabels',
                        help='Dataset directory (for metadata and sample testing)')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image to test')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("Make sure you've trained a model first!")
        return

    # Initialize tester
    try:
        tester = ChessModelTester(args.model, args.dataset)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Test single image if provided
    if args.image:
        if Path(args.image).exists():
            tester.test_single_image(args.image)
        else:
            print(f"❌ Image not found: {args.image}")
        return

    # Run interactive mode
    if args.interactive or not args.image:
        test_model_interactive(args.model, args.dataset)


if __name__ == "__main__":
    main()