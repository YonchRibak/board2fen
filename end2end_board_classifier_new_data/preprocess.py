# board2fen/end2end_board_classifier/preprocess.py
"""
Preprocess ChessReD annotations for GCS-based training.

This script:
1. Checks if preprocessed data already exists on GCS
2. If exists, skips preprocessing (data will be downloaded during training)
3. If not exists, downloads annotations from GCS and creates preprocessed files locally
4. Provides comprehensive error handling and logging

GCS Structure:
  https://storage.googleapis.com/chess_red_dataset/
    ├── annotations.json
    ├── chessred/images/0-99/ (image folders)
    └── chessred_preprocessed/ (preprocessed data - may or may not exist)

Outputs (local only - cannot write to public GCS bucket):
  ./gcs_cache/
    ├── class_order.json
    ├── index_train.jsonl
    ├── index_val.jsonl
    └── index_test.jsonl
"""
from __future__ import annotations
import json
import logging
import requests
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# GCS Configuration
GCS_BUCKET_BASE = "https://storage.googleapis.com/chess_red_dataset"
GCS_ANNOTATIONS_URL = f"{GCS_BUCKET_BASE}/annotations.json"
GCS_PREPROCESSED_BASE = f"{GCS_BUCKET_BASE}/chessred_preprocessed"
GCS_IMAGES_BASE = f"{GCS_BUCKET_BASE}/chessred/images"

# Local directories
LOCAL_CACHE_DIR = Path("./gcs_cache")
LOCAL_ANNOTATIONS_PATH = LOCAL_CACHE_DIR / "annotations.json"
LOCAL_CLASS_ORDER_PATH = LOCAL_CACHE_DIR / "class_order.json"
LOCAL_PREPROCESSED_BASE = LOCAL_CACHE_DIR
LOCAL_IMAGE_CACHE = LOCAL_CACHE_DIR / "images"

# Constants based on ChessReD categories
CLASS_MAPPING = {
    'w-pawn': 0, 'w-knight': 1, 'w-bishop': 2, 'w-rook': 3, 'w-queen': 4, 'w-king': 5,
    'b-pawn': 6, 'b-knight': 7, 'b-bishop': 8, 'b-rook': 9, 'b-queen': 10, 'b-king': 11,
    'empty': 12
}
CLASS_NAMES = [
    'w-pawn', 'w-knight', 'w-bishop', 'w-rook', 'w-queen', 'w-king',
    'b-pawn', 'b-knight', 'b-bishop', 'b-rook', 'b-queen', 'b-king',
    'empty'
]
NUM_CLASSES = len(CLASS_NAMES)
EMPTY_CLASS_ID = 12

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def chess_position_to_index(position: str) -> Optional[int]:
    """
    Convert chess notation (e.g., 'a8', 'e4') to board array index (0-63).

    Args:
        position: Chess notation string like 'a8', 'e4'

    Returns:
        Array index 0-63, or None if invalid position

    Board layout:
        a8=0,  b8=1,  c8=2,  d8=3,  e8=4,  f8=5,  g8=6,  h8=7
        a7=8,  b7=9,  c7=10, d7=11, e7=12, f7=13, g7=14, h7=15
        ...
        a1=56, b1=57, c1=58, d1=59, e1=60, f1=61, g1=62, h1=63
    """
    if not isinstance(position, str) or len(position) != 2:
        return None

    file_char = position[0].lower()  # a-h
    rank_char = position[1]  # 1-8

    if file_char not in 'abcdefgh' or rank_char not in '12345678':
        return None

    file_idx = ord(file_char) - ord('a')  # 0-7
    rank_idx = int(rank_char) - 1  # 0-7

    # Convert to array index: a8=0, b8=1, ..., h8=7, a7=8, ..., h1=63
    return (7 - rank_idx) * 8 + file_idx


def _download_file_with_retry(url: str, local_path: Path, max_retries: int = 3) -> bool:
    """Download a file from a URL to a local path with retry logic."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} to {local_path} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)

            logger.info(f"Successfully downloaded {url}")
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return False

    return False


def _create_label_from_pieces(pieces: List[Dict]) -> List[int]:
    """
    Convert piece annotations to a dense 64-element board array.

    Args:
        pieces: List of piece dictionaries from ChessReD annotations

    Returns:
        List of 64 class indices (0-12, where 12 = empty)
    """
    # Initialize all squares as empty
    board = [EMPTY_CLASS_ID] * 64

    for piece in pieces:
        try:
            position = piece.get('chessboard_position')
            category_id = piece.get('category_id')

            if position is None or category_id is None:
                logger.warning(f"Piece missing required fields: {piece}")
                continue

            # Convert chess position to array index
            idx = chess_position_to_index(position)
            if idx is None:
                logger.warning(f"Invalid chess position '{position}' in piece: {piece}")
                continue

            # Validate category_id
            if not (0 <= category_id <= 12):
                logger.warning(f"Invalid category_id {category_id} in piece: {piece}")
                continue

            # Place piece on board
            board[idx] = category_id

        except Exception as e:
            logger.warning(f"Error processing piece {piece}: {e}")
            continue

    return board


def check_preprocessed_data_exists() -> bool:
    """Check if preprocessed data exists on GCS."""
    required_files = [
        "class_order.json",
        "index_train.jsonl",
        "index_val.jsonl",
        "index_test.jsonl"
    ]

    print("[GCS] Checking if preprocessed data exists...")
    logger.info("Checking if preprocessed data exists on GCS")

    for file_name in required_files:
        url = f"{GCS_PREPROCESSED_BASE}/{file_name}"
        try:
            response = requests.head(url, timeout=10)
            if response.status_code != 200:
                print(f"[GCS] Missing file: {file_name}")
                logger.info(f"Preprocessed file not found: {file_name}")
                return False
        except Exception as e:
            print(f"[GCS] Error checking {file_name}: {e}")
            logger.warning(f"Error checking preprocessed file {file_name}: {e}")
            return False

    print("[GCS] All preprocessed files found!")
    logger.info("All preprocessed files found on GCS")
    return True


def download_and_preprocess() -> bool:
    """Main function to download annotations and preprocess them into JSONL format."""
    try:
        # Create cache directory
        LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Download annotations.json
        print("[DOWNLOAD] Downloading ChessReD annotations...")
        if not _download_file_with_retry(GCS_ANNOTATIONS_URL, LOCAL_ANNOTATIONS_PATH):
            print("[ERROR] Failed to download annotations.json")
            return False

        # 2. Load and parse annotations
        print("[PARSE] Loading annotations...")
        try:
            with open(LOCAL_ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
                chessred_data = json.load(f)

            logger.info(f"Loaded ChessReD data with keys: {list(chessred_data.keys())}")

        except Exception as e:
            logger.error(f"Failed to parse annotations.json: {e}")
            return False

        # 3. Extract data components
        try:
            images_data = chessred_data['images']
            annotations_data = chessred_data['annotations']
            categories_data = chessred_data['categories']

            # Extract pieces from annotations
            pieces_data = annotations_data['pieces']

            print(f"[INFO] Found {len(images_data)} images")
            print(f"[INFO] Found {len(pieces_data)} piece annotations")
            print(f"[INFO] Found {len(categories_data)} categories")

            logger.info(
                f"Data summary: {len(images_data)} images, {len(pieces_data)} pieces, {len(categories_data)} categories")

        except KeyError as e:
            logger.error(f"Missing expected key in ChessReD data: {e}")
            return False

        # 4. Create class order mapping from categories
        print("[CATEGORIES] Processing categories...")

        # Verify we have the expected 13 categories (0-11 pieces + 12 empty)
        if len(categories_data) != 13:
            logger.warning(f"Expected 13 categories, found {len(categories_data)}")

        # Save class order
        with open(LOCAL_CLASS_ORDER_PATH, 'w', encoding='utf-8') as f:
            json.dump(CLASS_NAMES, f, indent=2)

        logger.info("Saved class_order.json")

        # 5. Create image_id to image mapping
        print("[MAPPING] Creating image mappings...")
        image_id_to_info = {img['id']: img for img in images_data}

        # Group pieces by image_id
        pieces_by_image = {}
        for piece in pieces_data:
            image_id = piece['image_id']
            if image_id not in pieces_by_image:
                pieces_by_image[image_id] = []
            pieces_by_image[image_id].append(piece)

        logger.info(f"Grouped pieces into {len(pieces_by_image)} unique images")

        # 6. Create samples for each image
        print("[SAMPLES] Creating training samples...")
        samples = []

        for image_id, image_info in image_id_to_info.items():
            try:
                # Get pieces for this image (empty list if no pieces)
                image_pieces = pieces_by_image.get(image_id, [])

                # Convert pieces to board array
                board_labels = _create_label_from_pieces(image_pieces)

                # Create sparse representation for compatibility
                labels_sparse = [[i, board_labels[i]] for i in range(64) if board_labels[i] != EMPTY_CLASS_ID]

                # Convert local file path to GCS URL path structure
                file_path = image_info.get('file_name', '')
                if not file_path:
                    logger.warning(f"Image {image_id} missing file_name, skipping")
                    continue

                sample = {
                    'image_id': image_id,
                    'file_path': file_path,  # Keep original for GCS URL construction
                    'labels_dense': board_labels,
                    'labels_sparse': labels_sparse,
                    'piece_count': len(image_pieces),
                    'board_dims': (8, 8),
                    'num_classes': NUM_CLASSES
                }

                samples.append(sample)

            except Exception as e:
                logger.warning(f"Error processing image {image_id}: {e}")
                continue

        print(f"[SUCCESS] Created {len(samples)} samples from {len(images_data)} images")
        logger.info(f"Created {len(samples)} samples")

        # 7. Split into train/val/test
        print("[SPLIT] Splitting into train/val/test...")

        total_samples = len(samples)
        if total_samples == 0:
            logger.error("No valid samples created")
            return False

        train_split_idx = int(total_samples * 0.8)
        val_split_idx = int(total_samples * 0.9)

        train_samples = samples[:train_split_idx]
        val_samples = samples[train_split_idx:val_split_idx]
        test_samples = samples[val_split_idx:]

        print(f"[SPLIT] Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        logger.info(f"Split - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

        # 8. Save JSONL files
        datasets = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }

        for split_name, split_samples in datasets.items():
            output_path = LOCAL_PREPROCESSED_BASE / f"index_{split_name}.jsonl"

            print(f"[SAVE] Writing {split_name} data to {output_path}...")

            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for sample in split_samples:
                        f.write(json.dumps(sample) + '\n')

                logger.info(f"Saved {len(split_samples)} samples to {output_path}")

            except Exception as e:
                logger.error(f"Failed to save {split_name} data: {e}")
                return False

        # 9. Validation - check some samples
        print("[VALIDATE] Performing validation checks...")

        # Check that we have reasonable piece distributions
        total_pieces = sum(len(pieces_by_image.get(img_id, [])) for img_id in image_id_to_info.keys())
        avg_pieces_per_image = total_pieces / len(samples) if samples else 0

        print(f"[STATS] Total pieces: {total_pieces}")
        print(f"[STATS] Average pieces per image: {avg_pieces_per_image:.1f}")

        # Check for completely empty boards (suspicious)
        empty_boards = sum(1 for sample in samples if sample['piece_count'] == 0)
        if empty_boards > len(samples) * 0.1:  # More than 10% empty
            logger.warning(
                f"High number of empty boards: {empty_boards}/{len(samples)} ({100 * empty_boards / len(samples):.1f}%)")

        print(f"[STATS] Empty boards: {empty_boards}/{len(samples)} ({100 * empty_boards / len(samples):.1f}%)")

        print("[SUCCESS] Preprocessing completed successfully!")
        logger.info("Preprocessing completed successfully")
        return True

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        print(f"[ERROR] Preprocessing failed: {e}")
        return False


def main() -> bool:
    """Main execution function for the preprocessing script."""
    print("=" * 70)
    print("ChessReD Data Preprocessing")
    print("=" * 70)

    try:
        # Step 1: Check if preprocessed data already exists on GCS
        print(f"\n[CHECK] Checking for existing preprocessed data on GCS...")

        if check_preprocessed_data_exists():
            print(f"\n[SKIP] Preprocessed data already exists on GCS!")
            print(f"[INFO] Training will download this data automatically.")
            print(f"[INFO] If you want to recreate the data, delete it from GCS first.")
            logger.info("Preprocessed data already exists on GCS, skipping preprocessing")
            return True

        # Step 2: Download and preprocess
        print(f"\n[START] No preprocessed data found. Starting preprocessing...")

        if not download_and_preprocess():
            print(f"\n[FATAL] Preprocessing failed!")
            logger.error("Preprocessing failed")
            return False

        # Step 3: Success message
        print(f"\n" + "=" * 70)
        print(f"[SUCCESS] Preprocessing completed successfully!")
        print(f"=" * 70)
        print(f"[INFO] Local cache: {LOCAL_CACHE_DIR}")
        print(f"[INFO] Files created:")

        created_files = [
            "class_order.json",
            "index_train.jsonl",
            "index_val.jsonl",
            "index_test.jsonl"
        ]

        for filename in created_files:
            filepath = LOCAL_CACHE_DIR / filename
            if filepath.exists():
                size_kb = filepath.stat().st_size / 1024
                print(f"  • {filename}: {size_kb:.1f} KB")

        print(f"\n[NEXT STEPS]")
        print(f"1. Run train.py to start training with this data")
        print(f"2. Optional: Upload preprocessed data to GCS to share with others:")
        print(f"   - Upload contents of {LOCAL_CACHE_DIR}/ to:")
        print(f"   - {GCS_PREPROCESSED_BASE}/")
        print(f"   - Then future runs will skip preprocessing automatically")

        logger.info("Preprocessing completed successfully")
        return True

    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Preprocessing interrupted by user")
        logger.info("Preprocessing interrupted by user")
        return False

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"\n[FATAL] {error_msg}")
        logger.error(error_msg)

        import traceback
        traceback_str = traceback.format_exc()
        logger.error(f"Traceback: {traceback_str}")

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)