# run_preprocessing.py
# Simple script to preprocess ChessRender360 data into grid labels

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import chess
from tqdm import tqdm


class ChessPositionEncoder:
    """Convert chess positions from FEN notation to multi-level grid labels"""

    def __init__(self):
        # Complete piece mapping (13 classes: empty + 12 pieces)
        self.piece_to_class = {
            None: 0,  # Empty square
            'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6,  # White pieces
            'p': 7, 'r': 8, 'n': 9, 'b': 10, 'q': 11, 'k': 12  # Black pieces
        }

    def parse_fen_to_board_matrix(self, fen_string: str) -> Optional[np.ndarray]:
        """Convert FEN string to 8x8 board matrix with piece symbols"""
        try:
            board = chess.Board(fen_string)
            board_matrix = np.full((8, 8), None, dtype=object)

            for square in chess.SQUARES:
                piece = board.piece_at(square)
                # Convert chess coordinates to matrix coordinates
                row = 7 - chess.square_rank(square)  # Flip rank
                col = chess.square_file(square)  # File stays same

                if piece:
                    board_matrix[row, col] = piece.symbol()
                else:
                    board_matrix[row, col] = None

            return board_matrix

        except Exception as e:
            print(f"Error parsing FEN '{fen_string}': {e}")
            return None

    def encode_position_multilevel(self, board_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate multi-level labels from board matrix"""
        labels = {}

        # Complete classification (13 classes)
        complete_grid = np.zeros((8, 8), dtype=np.int8)

        # Occupancy detection (2 classes)
        occupancy_grid = np.zeros((8, 8), dtype=np.int8)

        # Color classification (3 classes)
        color_grid = np.zeros((8, 8), dtype=np.int8)

        # Piece type classification (7 classes, color-blind)
        piece_type_grid = np.zeros((8, 8), dtype=np.int8)
        piece_type_classes = {None: 0, 'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6}

        for row in range(8):
            for col in range(8):
                piece_symbol = board_matrix[row, col]

                # Complete classification
                complete_grid[row, col] = self.piece_to_class[piece_symbol]

                # Occupancy classification
                occupancy_grid[row, col] = 0 if piece_symbol is None else 1

                # Color classification
                if piece_symbol is None:
                    color_grid[row, col] = 0  # Empty
                elif piece_symbol.isupper():
                    color_grid[row, col] = 1  # White
                else:
                    color_grid[row, col] = 2  # Black

                # Piece type classification (remove color)
                if piece_symbol is None:
                    piece_type_grid[row, col] = 0  # Empty
                else:
                    piece_type = piece_symbol.upper()  # Normalize to uppercase
                    piece_type_grid[row, col] = piece_type_classes[piece_type]

        return {
            'complete': complete_grid,
            'occupancy': occupancy_grid,
            'color': color_grid,
            'piece_type': piece_type_grid
        }


def load_fens_from_csv(csv_path: str) -> List[str]:
    """Load FEN strings from the unusual CSV format"""
    fen_strings = []

    with open(csv_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line and '/' in line:  # Basic FEN validation
            fen_strings.append(line)

    return fen_strings


def find_rgb_images(rgb_dir: Path) -> List[Path]:
    """Find all RGB image files in order"""
    rgb_files = []

    # Look for pattern: rgb_0.jpeg, rgb_1.jpeg, etc.
    for i in range(10000):  # ChessRender360 has up to 10k images
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = rgb_dir / f"rgb_{i}{ext}"
            if img_path.exists():
                rgb_files.append(img_path)
                break
        else:
            # If we can't find rgb_{i}.ext for 100 consecutive numbers, stop
            if i > 100 and len(rgb_files) == 0:
                break
            if i > max(100, len(rgb_files) + 100):
                break

    # Sort to ensure correct order
    rgb_files.sort(key=lambda x: int(x.stem.split('_')[1]))
    return rgb_files


def preprocess_chessrender360(dataset_path: str, output_dir: str, max_samples: int = None):
    """Main preprocessing function"""

    print("♟️  CHESS GRID LABEL PREPROCESSING")
    print("=" * 50)

    dataset_path = Path(dataset_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load FEN strings
    fens_csv = dataset_path / "FENs.csv"
    print(f"📄 Loading FENs from: {fens_csv}")
    fen_strings = load_fens_from_csv(str(fens_csv))
    print(f"✅ Loaded {len(fen_strings)} FEN strings")

    # Find RGB images
    rgb_dir = dataset_path / "rgb"
    print(f"📁 Looking for images in: {rgb_dir}")
    rgb_files = find_rgb_images(rgb_dir)
    print(f"✅ Found {len(rgb_files)} RGB images")

    # Match FENs to images
    min_samples = min(len(fen_strings), len(rgb_files))
    if max_samples:
        min_samples = min(min_samples, max_samples)

    print(f"📊 Processing {min_samples} FEN-image pairs")

    # Initialize encoder
    encoder = ChessPositionEncoder()

    # Storage for processed data
    valid_image_paths = []
    all_labels = {
        'complete': [],
        'occupancy': [],
        'color': [],
        'piece_type': []
    }

    # Processing statistics
    stats = {
        'total_processed': 0,
        'successful': 0,
        'failed_fen_parsing': 0,
        'failed_image_not_found': 0
    }

    # Process each sample
    print(f"🔄 Processing samples...")
    for idx in tqdm(range(min_samples)):
        stats['total_processed'] += 1

        # Get FEN and image path
        fen_string = fen_strings[idx]
        image_path = rgb_files[idx]

        # Verify image exists
        if not image_path.exists():
            stats['failed_image_not_found'] += 1
            continue

        # Parse FEN to board matrix
        board_matrix = encoder.parse_fen_to_board_matrix(fen_string)
        if board_matrix is None:
            stats['failed_fen_parsing'] += 1
            continue

        # Generate multi-level labels
        labels = encoder.encode_position_multilevel(board_matrix)

        # Store successful result
        valid_image_paths.append(str(image_path))
        for label_type in all_labels:
            all_labels[label_type].append(labels[label_type])

        stats['successful'] += 1

    # Convert to numpy arrays
    for label_type in all_labels:
        all_labels[label_type] = np.array(all_labels[label_type])

    # Save processed data
    print(f"\n💾 Saving processed data to {output_path}...")

    # Save image paths
    with open(output_path / "image_paths.txt", 'w') as f:
        for img_path in valid_image_paths:
            f.write(f"{img_path}\n")

    # Save label arrays
    for label_type, labels in all_labels.items():
        np.save(output_path / f"labels_{label_type}.npy", labels)
        print(f"  💾 {label_type}: {labels.shape} -> labels_{label_type}.npy")

    # Save metadata
    metadata = {
        'dataset_source': str(dataset_path),
        'num_samples': stats['successful'],
        'label_types': list(all_labels.keys()),
        'piece_mapping': encoder.piece_to_class,
        'grid_shape': (8, 8),
        'processing_stats': stats,
        'label_shapes': {k: v.shape for k, v in all_labels.items()}
    }

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\n✅ Preprocessing complete!")
    print(f"📊 Results:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed - FEN parsing: {stats['failed_fen_parsing']}")
    print(f"  Failed - Image not found: {stats['failed_image_not_found']}")
    print(f"  Success rate: {stats['successful'] / stats['total_processed'] * 100:.1f}%")

    print(f"\n📁 Generated files:")
    print(f"  📄 image_paths.txt - {len(valid_image_paths)} image paths")
    print(f"  📊 metadata.json - Dataset metadata")
    for label_type in all_labels:
        shape = all_labels[label_type].shape
        print(f"  🔢 labels_{label_type}.npy - {shape}")

    return stats


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "C:/datasets/ChessRender360"  # Update this path
    OUTPUT_DIR = "C:/datasets/ChessRender360_GridLabels"
    MAX_SAMPLES = None  # Set to None for all samples, or a number for testing

    # Check if dataset exists
    if not Path(DATASET_PATH).exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print("Please update the DATASET_PATH variable")
        exit(1)

    # Run preprocessing
    try:
        stats = preprocess_chessrender360(DATASET_PATH, OUTPUT_DIR, MAX_SAMPLES)

        if stats['successful'] > 0:
            print(f"\n🎉 SUCCESS! Generated grid labels for {stats['successful']} samples")
            print(f"📂 Output directory: {OUTPUT_DIR}")
            print(f"\n📋 Next steps:")
            print(f"  1. Check the generated files in {OUTPUT_DIR}")
            print(f"  2. If everything looks good, run again with MAX_SAMPLES=None for full dataset")
            print(f"  3. Use the generated dataset for training your end-to-end CNN")
        else:
            print(f"\n❌ No samples processed successfully. Check error messages above.")

    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback

        traceback.print_exc()