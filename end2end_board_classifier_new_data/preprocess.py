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
from typing import Dict, Any, List, Tuple

# GCS Configuration
GCS_BUCKET_BASE = "https://storage.googleapis.com/chess_red_dataset"
GCS_ANNOTATIONS_URL = f"{GCS_BUCKET_BASE}/annotations.json"
GCS_PREPROCESSED_BASE = f"{GCS_BUCKET_BASE}/chessred_preprocessed"
GCS_IMAGES_BASE = f"{GCS_BUCKET_BASE}/chessred"

# Local output directory
LOCAL_CACHE_DIR = Path(__file__).resolve().parent / "gcs_cache"

# Chess board mapping
FILE_TO_COL = {c: i for i, c in enumerate("abcdefgh")}


def setup_logging() -> logging.Logger:
    """Setup logging for preprocessing."""
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger('chess_preprocessing')
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    log_file = log_dir / 'preprocessing.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def download_with_retry(url: str, max_retries: int = 3, timeout: int = 30) -> requests.Response:
    """Download with retry logic and comprehensive error handling."""
    logger = logging.getLogger('chess_preprocessing')

    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
            print(f"[DOWNLOAD] Fetching {url} (attempt {attempt + 1}/{max_retries})")

            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            logger.info(f"Successfully downloaded {url}")
            print(f"[SUCCESS] Downloaded {url}")
            return response

        except requests.exceptions.Timeout as e:
            error_msg = f"Timeout downloading {url}: {e}"
            logger.warning(error_msg)
            print(f"[WARNING] {error_msg}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error downloading {url}: {e}"
            logger.warning(error_msg)
            print(f"[WARNING] {error_msg}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error downloading {url}: {e.response.status_code}"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}")
            raise

        except Exception as e:
            error_msg = f"Unexpected error downloading {url}: {e}"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}")
            raise


def check_preprocessed_exists() -> bool:
    """Check if preprocessed data exists on GCS."""
    logger = logging.getLogger('chess_preprocessing')

    required_files = [
        "class_order.json",
        "index_train.jsonl",
        "index_val.jsonl",
        "index_test.jsonl"
    ]

    print("[CHECK] Checking if preprocessed data exists on GCS...")
    logger.info("Checking if preprocessed data exists on GCS")

    missing_files = []

    for file_name in required_files:
        url = f"{GCS_PREPROCESSED_BASE}/{file_name}"
        try:
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                print(f"[CHECK] Found: {file_name}")
                logger.info(f"Found preprocessed file: {file_name}")
            else:
                print(f"[CHECK] Missing: {file_name} (status: {response.status_code})")
                logger.info(f"Missing preprocessed file: {file_name}")
                missing_files.append(file_name)

        except Exception as e:
            error_msg = f"Error checking {file_name}: {e}"
            print(f"[WARNING] {error_msg}")
            logger.warning(error_msg)
            missing_files.append(file_name)

    if not missing_files:
        print("[SUCCESS] All preprocessed files found on GCS!")
        logger.info("All preprocessed files found on GCS")
        return True
    else:
        print(f"[INFO] Missing files: {missing_files}")
        logger.info(f"Missing preprocessed files: {missing_files}")
        return False


def algebraic_to_index(square: str) -> int:
    """Convert algebraic notation (e.g., 'a1') to board index (0-63)."""
    file_c = square[0].lower()
    rank_c = square[1]
    col = FILE_TO_COL[file_c]
    row = int(rank_c) - 1
    return row * 8 + col


def build_class_order(categories: List[Dict[str, Any]]) -> List[str]:
    """Build class order from categories (sorted by id)."""
    categories_sorted = sorted(categories, key=lambda c: c["id"])
    return [c["name"].strip().lower() for c in categories_sorted]


def build_category_maps(categories: List[Dict[str, Any]], class_order: List[str]) -> Tuple[Dict[int, int], int]:
    """Map dataset category_id -> model class index; return mapping and empty category id."""
    logger = logging.getLogger('chess_preprocessing')

    name_to_idx = {n: i for i, n in enumerate(class_order)}
    catid_to_idx: Dict[int, int] = {}
    empty_id = None

    for c in categories:
        nm = c["name"].strip().lower()
        if nm not in name_to_idx:
            error_msg = f"Category '{nm}' not present in class_order"
            logger.error(error_msg)
            raise ValueError(error_msg)

        catid_to_idx[c["id"]] = name_to_idx[nm]
        if nm == "empty":
            empty_id = c["id"]

    if empty_id is None:
        error_msg = "No 'empty' category found in annotations"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Built category mapping with {len(catid_to_idx)} categories")
    return catid_to_idx, empty_id


def collect_pieces_by_image(anno: Dict[str, Any], class_order: List[str]) -> Dict[int, List[Tuple[int, int]]]:
    """Collect piece positions for each image."""
    logger = logging.getLogger('chess_preprocessing')

    try:
        catid_to_idx, _empty_id = build_category_maps(anno["categories"], class_order)
        per_image: Dict[int, List[Tuple[int, int]]] = {}

        total_pieces = len(anno["annotations"]["pieces"])
        processed = 0

        for p in anno["annotations"]["pieces"]:
            img_id = int(p["image_id"])
            sq = p["chessboard_position"]
            ds_cat = int(p["category_id"])

            sq_idx = algebraic_to_index(sq)
            cls_idx = catid_to_idx[ds_cat]

            # Skip empty pieces (they're implicit)
            if class_order[cls_idx] == "empty":
                continue

            per_image.setdefault(img_id, []).append((sq_idx, cls_idx))
            processed += 1

            if processed % 10000 == 0:
                print(f"[PROGRESS] Processed {processed}/{total_pieces} piece annotations")

        logger.info(f"Collected pieces for {len(per_image)} images from {total_pieces} annotations")
        return per_image

    except Exception as e:
        error_msg = f"Error collecting pieces: {e}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}")
        raise


def images_index_by_id(anno: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Create image lookup by ID."""
    return {int(im["id"]): im for im in anno["images"]}


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> bool:
    """Write rows to JSONL file with error handling."""
    logger = logging.getLogger('chess_preprocessing')

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info(f"Successfully wrote {len(rows)} rows to {path}")
        return True

    except Exception as e:
        error_msg = f"Error writing JSONL to {path}: {e}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}")
        return False


def resolve_split_ids(anno: Dict[str, Any], split_name: str) -> List[int]:
    """Get image IDs for a data split."""
    try:
        return [int(x) for x in anno["splits"][split_name]["image_ids"]]
    except Exception as e:
        logger = logging.getLogger('chess_preprocessing')
        error_msg = f"Error resolving {split_name} split: {e}"
        logger.error(error_msg)
        raise


def build_gcs_rows(image_ids: List[int], img_by_id: Dict[int, Dict[str, Any]],
                   per_image_sparse: Dict[int, List[Tuple[int, int]]]) -> List[Dict[str, Any]]:
    """Build training rows with GCS URLs."""
    logger = logging.getLogger('chess_preprocessing')

    rows = []
    missing_count = 0

    for iid in image_ids:
        if iid not in img_by_id:
            missing_count += 1
            continue

        meta = img_by_id[iid]
        # Convert path to GCS URL
        rel_path = meta["path"].replace("\\", "/")  # Normalize path separators
        gcs_url = f"{GCS_IMAGES_BASE}/{rel_path}"

        # Get sparse piece labels (empty squares are implicit)
        sparse = per_image_sparse.get(iid, [])

        rows.append({
            "image_id": iid,
            "file_path": gcs_url,  # Now points to GCS
            "labels_sparse": sparse,
        })

    if missing_count > 0:
        logger.warning(f"Skipped {missing_count} images with missing metadata")
        print(f"[WARNING] Skipped {missing_count} images with missing metadata")

    logger.info(f"Built {len(rows)} rows for data split")
    return rows


def download_and_preprocess() -> bool:
    """Download annotations from GCS and create preprocessed data."""
    logger = logging.getLogger('chess_preprocessing')

    try:
        print("[DOWNLOAD] Downloading annotations from GCS...")
        logger.info("Starting download and preprocessing")

        # Download annotations.json
        response = download_with_retry(GCS_ANNOTATIONS_URL)
        anno = response.json()

        print(f"[SUCCESS] Loaded annotations with {len(anno['images'])} images")
        logger.info(f"Loaded annotations with {len(anno['images'])} images")

        # Build class order and piece mappings
        print("[PROCESS] Building class order and mappings...")
        class_order = build_class_order(anno["categories"])
        per_image_sparse = collect_pieces_by_image(anno, class_order)

        # Create output directory
        LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Save class order
        class_order_path = LOCAL_CACHE_DIR / "class_order.json"
        with open(class_order_path, 'w', encoding='utf-8') as f:
            json.dump(class_order, f, indent=2)

        print(f"[SUCCESS] Saved class order with {len(class_order)} classes")
        logger.info(f"Saved class order with {len(class_order)} classes: {class_order}")

        # Build image index
        img_by_id = images_index_by_id(anno)

        # Process splits
        print("[PROCESS] Processing data splits...")

        splits_data = {}
        for split_name in ["train", "val", "test"]:
            try:
                image_ids = resolve_split_ids(anno, split_name)
                rows = build_gcs_rows(image_ids, img_by_id, per_image_sparse)
                splits_data[split_name] = rows

                print(f"[SUCCESS] {split_name}: {len(rows)} samples")
                logger.info(f"Processed {split_name} split: {len(rows)} samples")

            except Exception as e:
                error_msg = f"Error processing {split_name} split: {e}"
                logger.error(error_msg)
                print(f"[ERROR] {error_msg}")
                return False

        # Write JSONL files
        print("[SAVE] Writing preprocessed data files...")
        for split_name, rows in splits_data.items():
            jsonl_path = LOCAL_CACHE_DIR / f"index_{split_name}.jsonl"
            if not write_jsonl(jsonl_path, rows):
                return False
            print(f"[SUCCESS] Saved {split_name} data: {jsonl_path}")

        # Summary
        total_samples = sum(len(rows) for rows in splits_data.values())
        print(f"\n[COMPLETE] Preprocessing finished successfully!")
        print(f"[SUMMARY] Total samples: {total_samples}")
        print(f"[SUMMARY] Train: {len(splits_data['train'])}")
        print(f"[SUMMARY] Val: {len(splits_data['val'])}")
        print(f"[SUMMARY] Test: {len(splits_data['test'])}")
        print(f"[SUMMARY] Output directory: {LOCAL_CACHE_DIR}")

        logger.info(f"Preprocessing complete - {total_samples} total samples processed")
        return True

    except Exception as e:
        error_msg = f"Fatal error during preprocessing: {e}"
        logger.error(error_msg)
        print(f"[FATAL] {error_msg}")
        return False


def main():
    """Main preprocessing function."""
    print("=" * 70)
    print("Chess Dataset Preprocessing - GCS Version")
    print("=" * 70)

    # Setup logging
    logger = setup_logging()
    logger.info("Starting chess dataset preprocessing")

    try:
        # Step 1: Check if preprocessed data already exists
        if check_preprocessed_exists():
            print("\n[INFO] Preprocessed data already exists on GCS!")
            print("[INFO] Skipping preprocessing - data will be downloaded during training.")
            print("[INFO] If you need to recreate the preprocessed data, please contact the dataset maintainer.")
            logger.info("Preprocessed data exists on GCS - skipping preprocessing")
            return True

        # Step 2: Data doesn't exist, so we need to create it
        print("\n[INFO] Preprocessed data not found on GCS.")
        print("[INFO] Downloading annotations and creating preprocessed data locally...")
        logger.info("Preprocessed data not found - starting local preprocessing")

        # Step 3: Download and preprocess
        if not download_and_preprocess():
            print("\n[FATAL] Preprocessing failed!")
            logger.error("Preprocessing failed")
            return False

        # Step 4: Success message
        print(f"\n[SUCCESS] Preprocessing completed successfully!")
        print(f"[INFO] Data saved locally to: {LOCAL_CACHE_DIR}")
        print(f"[INFO] This data will be used for training.")
        print(f"\n[NOTE] To share this preprocessed data:")
        print(f"       1. Upload the contents of {LOCAL_CACHE_DIR} to:")
        print(f"          {GCS_PREPROCESSED_BASE}/")
        print(f"       2. Then future runs will skip preprocessing automatically.")

        logger.info("Preprocessing completed successfully")
        return True

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Preprocessing interrupted by user")
        logger.info("Preprocessing interrupted by user")
        return False

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"\n[FATAL] {error_msg}")
        logger.error(error_msg)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)