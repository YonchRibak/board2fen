# board2fen/end2end_board_classifier/train.py
import os
import sys
import time
import json
import gc
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_disable_all_hlo_passes'
os.environ['TF_DISABLE_XLA'] = '1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from _helpers import (
    get_mode_config,
    get_light_model_config,
    seed_everything,
    build_resnext101_32x8d_head,
    build_light_chess_classifier,
    make_dirs,
    make_callbacks,
    pretty_cfg,
    setup_logging,
    check_preprocessed_data_exists,
    check_gcs_upload_requirements,
    prompt_upload_models,
    BoardLevelEvalCallback,
    _read_jsonl,
    load_class_order_from_gcs,
    find_empty_index,
    _augmentor,
    GCS_IMAGES_BASE,
    GCS_PREPROCESSED_BASE,
    download_preprocessed_data
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "end2end_board_classifier_new_data" / "outputs" / "best_model"
DEFAULT_ANALYTICS_DIR = PROJECT_ROOT / "end2end_board_classifier_new_data" / "analytics"

# Global logger will be initialized in main()
logger = None

# Set matplotlib to non-interactive backend for headless environments
mplstyle.use('default')
plt.ioff()  # Turn off interactive mode


# Add missing function as a patch (to avoid needing to update _helpers.py)
def download_file_with_retry(url: str, max_retries: int = 3, timeout: int = 30):
    """Quiet version of download function to replace the one in _helpers.py"""
    import requests
    import time

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
        except requests.exceptions.ConnectionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
        except requests.exceptions.HTTPError:
            raise
        except Exception:
            raise


class ProgressBar:
    """Simple thread-safe progress bar for download tracking."""

    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, increment: int = 1):
        with self.lock:
            self.current += increment
            self._display()

    def _display(self):
        if self.total == 0:
            return

        percent = (self.current / self.total) * 100
        filled = int(50 * self.current // self.total)
        bar = '█' * filled + '-' * (50 - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f" ETA: {eta:.0f}s" if eta < 3600 else f" ETA: {eta / 3600:.1f}h"
        else:
            eta_str = ""

        print(f'\r{self.description}: |{bar}| {self.current}/{self.total} ({percent:.1f}%){eta_str}',
              end='', flush=True)

        if self.current >= self.total:
            print()  # New line when complete


def extract_image_paths_from_jsonl(jsonl_files: List[Path]) -> Set[str]:
    """Extract all unique image paths from JSONL files."""
    image_paths = set()

    for jsonl_file in jsonl_files:
        if not jsonl_file.exists():
            print(f"[WARNING] JSONL file not found: {jsonl_file}")
            continue

        try:
            for row in _read_jsonl(jsonl_file):
                file_path = row.get("file_path", "")
                if file_path:
                    image_paths.add(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to read {jsonl_file}: {e}")

    return image_paths


def convert_path_to_gcs_url(file_path: str) -> str:
    """Convert local file path to GCS URL with proper folder structure."""
    if file_path.startswith("http"):
        return file_path

    # Extract filename from path
    filename = Path(file_path).name

    # ChessReD images are named like G011_IMG155.jpg where G011 means game 11
    # The images are stored in folders 0-99 corresponding to the game number
    if filename.startswith('G') and '_' in filename:
        try:
            # Extract game number from filename (e.g., G011 -> 11)
            game_part = filename.split('_')[0]  # Get "G011" part
            game_number = int(game_part[1:])  # Remove 'G' and convert to int

            # Construct proper GCS URL with folder structure
            gcs_url = f"{GCS_IMAGES_BASE}/{game_number}/{filename}"
            return gcs_url

        except (ValueError, IndexError) as e:
            if logger:
                logger.warning(f"Could not extract game number from filename {filename}: {e}")

    # Fallback: try original path parsing
    path_parts = file_path.replace("\\", "/").split("/")
    try:
        images_idx = path_parts.index("images")
        rel_path_parts = path_parts[images_idx + 1:]  # Skip "images", take folder/file
        gcs_url = f"{GCS_IMAGES_BASE}/{'/'.join(rel_path_parts)}"
    except (ValueError, IndexError):
        # Last resort: assume it's a direct filename and try folder 0
        gcs_url = f"{GCS_IMAGES_BASE}/0/{filename}"
        if logger:
            logger.warning(f"Using fallback URL for {filename}: {gcs_url}")

    return gcs_url


def download_single_image(args: Tuple[str, Path]) -> Tuple[bool, str, str]:
    """Download a single image. Returns (success, gcs_url, error_message)."""
    gcs_url, local_path = args

    try:
        # Create parent directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists and is not empty
        if local_path.exists() and local_path.stat().st_size > 0:
            return True, gcs_url, ""

        # Download the image
        response = download_file_with_retry(gcs_url, max_retries=2, timeout=10)

        # Save to local path
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Validate the downloaded file
        if local_path.stat().st_size == 0:
            local_path.unlink()  # Remove empty file
            return False, gcs_url, "Downloaded file was empty"

        return True, gcs_url, ""

    except Exception as e:
        # Clean up any partial download
        if local_path.exists():
            try:
                local_path.unlink()
            except:
                pass
        return False, gcs_url, str(e)


def pre_download_images(image_paths: Set[str], cache_dir: Path, max_workers: int = 10) -> Tuple[int, int, List[str]]:
    """
    Pre-download all images with progress bar.

    Returns:
        (successful_downloads, failed_downloads, failed_urls)
    """
    print(f"\n[DOWNLOAD] Pre-downloading {len(image_paths)} images...")
    if logger:
        logger.info(f"Starting pre-download of {len(image_paths)} images")

    # Create download tasks
    download_tasks = []
    for file_path in image_paths:
        gcs_url = convert_path_to_gcs_url(file_path)

        # Create local path structure
        # Extract folder/filename from GCS URL
        url_parts = gcs_url.replace(f"{GCS_IMAGES_BASE}/", "").split("/")
        if len(url_parts) >= 2:
            folder, filename = url_parts[-2], url_parts[-1]
            local_path = cache_dir / "images" / folder / filename
        else:
            # Fallback
            filename = url_parts[-1] if url_parts else "unknown.jpg"
            local_path = cache_dir / "images" / filename

        download_tasks.append((gcs_url, local_path))

    # Progress tracking
    progress = ProgressBar(len(download_tasks), "Downloading images")
    successful_downloads = 0
    failed_downloads = 0
    failed_urls = []

    # Download with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(download_single_image, task): task[0]
                         for task in download_tasks}

        # Process completed downloads
        for future in as_completed(future_to_url):
            success, gcs_url, error_msg = future.result()

            if success:
                successful_downloads += 1
            else:
                failed_downloads += 1
                failed_urls.append(gcs_url)
                # Only print failures (not 404s which are common and expected)
                if error_msg and "404" not in error_msg and "Not Found" not in error_msg:
                    print(f"\n[FAILED] {gcs_url}: {error_msg}")

            progress.update(1)

    # Summary
    print(f"\n[DOWNLOAD SUMMARY]")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Successfully downloaded: {successful_downloads}")
    print(f"  Failed downloads: {failed_downloads}")

    if failed_downloads > 0:
        success_rate = (successful_downloads / len(image_paths)) * 100
        print(f"  Success rate: {success_rate:.1f}%")

        # Show some failed URLs (not all to avoid spam)
        print(f"  Sample failed URLs:")
        for url in failed_urls[:5]:
            print(f"    - {url}")
        if len(failed_urls) > 5:
            print(f"    - ... and {len(failed_urls) - 5} more")

    if logger:
        logger.info(f"Download complete: {successful_downloads} successful, {failed_downloads} failed")

    return successful_downloads, failed_downloads, failed_urls


def load_image_from_cache(local_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    """Load image from local cache with fallback to placeholder."""
    try:
        if local_path.exists() and local_path.stat().st_size > 0:
            # Load from local file
            with open(local_path, 'rb') as f:
                image_bytes = f.read()

            # Decode image
            image = tf.io.decode_jpeg(image_bytes, channels=3)
            image = tf.image.resize(image, target_size, method="bilinear")
            image = tf.cast(image, tf.float32) / 255.0

            # Basic validation
            if tf.reduce_mean(image) > 0.001:
                return image.numpy()

        # Fallback for missing/invalid images
        fallback = np.zeros((*target_size, 3), dtype=np.float32)
        fallback[::16, ::16] = 0.5  # Checkerboard pattern
        return fallback

    except Exception as e:
        if logger:
            logger.warning(f"Error loading cached image {local_path}: {e}")
        # Return fallback
        fallback = np.zeros((*target_size, 3), dtype=np.float32)
        fallback[::16, ::16] = 0.5
        return fallback


def get_datasets_with_cache(cfg: Dict, cache_dir: Path) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Load datasets using pre-downloaded cached images."""
    try:
        # Load preprocessed data indices
        idx_train = cache_dir / "index_train.jsonl"
        idx_val = cache_dir / "index_val.jsonl"

        if not idx_train.exists() or not idx_val.exists():
            raise FileNotFoundError(f"Missing preprocessed files in cache: {cache_dir}")

        # Load class order
        class_order = load_class_order_from_gcs(cache_dir)
        num_classes_file = len(class_order)
        empty_idx = find_empty_index(class_order)

        if cfg.get("num_classes") != num_classes_file:
            print(f"[WARNING] Overriding cfg['num_classes'] from {cfg.get('num_classes')} -> {num_classes_file}")
            cfg["num_classes"] = num_classes_file

        # Read JSONL files
        train_rows = list(_read_jsonl(idx_train))
        val_rows = list(_read_jsonl(idx_val))

        print(f"[SUCCESS] Loaded {len(train_rows)} training and {len(val_rows)} validation samples")

        bs = cfg["batch_size"]
        size = cfg["input_shape"]
        num_squares = cfg["num_squares"]
        num_classes = cfg["num_classes"]
        shuffle_buffer = cfg["dataset"].get("shuffle_buffer", 512)

        # Create generators that use cached images
        def train_generator():
            for row in train_rows:
                file_path = row["file_path"]
                gcs_url = convert_path_to_gcs_url(file_path)

                # Convert to local cache path
                url_parts = gcs_url.replace(f"{GCS_IMAGES_BASE}/", "").split("/")
                if len(url_parts) >= 2:
                    folder, filename = url_parts[-2], url_parts[-1]
                    local_path = cache_dir / "images" / folder / filename
                else:
                    filename = url_parts[-1] if url_parts else "unknown.jpg"
                    local_path = cache_dir / "images" / filename

                yield (str(local_path), row["labels_sparse"])

        def val_generator():
            for row in val_rows:
                file_path = row["file_path"]
                gcs_url = convert_path_to_gcs_url(file_path)

                # Convert to local cache path
                url_parts = gcs_url.replace(f"{GCS_IMAGES_BASE}/", "").split("/")
                if len(url_parts) >= 2:
                    folder, filename = url_parts[-2], url_parts[-1]
                    local_path = cache_dir / "images" / folder / filename
                else:
                    filename = url_parts[-1] if url_parts else "unknown.jpg"
                    local_path = cache_dir / "images" / filename

                yield (str(local_path), row["labels_sparse"])

        # Define output signature
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),  # Local file path
            tf.TensorSpec(shape=(None, 2), dtype=tf.int32),  # sparse labels
        )

        # Create datasets
        train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=output_signature)
        val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=output_signature)

        # Parse function using cached images
        def parse_sample(local_path_str, labels_sparse):
            # Load image from cache
            image_loader_fn = lambda path_str: tf.py_function(
                lambda p: load_image_from_cache(Path(p.numpy().decode('utf-8')), size[:2]),
                [path_str], tf.float32
            )
            image = image_loader_fn(local_path_str)
            image.set_shape(cfg["input_shape"])

            # Convert sparse to dense
            def sparse_to_dense_py(sparse_pairs):
                sparse_pairs = sparse_pairs.numpy()
                dense = np.zeros((num_squares, num_classes), dtype=np.float32)
                dense[:, empty_idx] = 1.0  # Initialize as empty

                for sq_idx, cls_idx in sparse_pairs:
                    if 0 <= sq_idx < num_squares and 0 <= cls_idx < num_classes:
                        dense[sq_idx, :] = 0.0
                        dense[sq_idx, cls_idx] = 1.0
                return dense

            dense_labels = tf.py_function(sparse_to_dense_py, [labels_sparse], tf.float32)
            dense_labels.set_shape((num_squares, num_classes))

            return image, dense_labels

        # Apply parsing
        train_ds = train_ds.map(parse_sample, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(parse_sample, num_parallel_calls=tf.data.AUTOTUNE)

        # Apply augmentations
        if cfg["dataset"]["augment"]:
            train_ds = train_ds.map(_augmentor, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle, batch, prefetch
        if cfg["dataset"]["shuffle"]:
            train_ds = train_ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

        train_ds = train_ds.batch(bs, drop_remainder=False)
        val_ds = val_ds.batch(bs, drop_remainder=False)

        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        steps_per_epoch = max(1, len(train_rows) // bs)
        val_steps = max(1, len(val_rows) // bs)

        print(f"[SUCCESS] Created datasets - Steps/epoch: {steps_per_epoch}, Val steps: {val_steps}")

        return train_ds, val_ds, steps_per_epoch, val_steps

    except Exception as e:
        error_msg = f"Failed to create datasets: {e}"
        print(f"[ERROR] {error_msg}")
        if logger:
            logger.error(error_msg)
        raise


def generate_training_graphs(history: dict, output_dir: Path, model_info: dict):
    """Generate and save training visualization graphs."""
    try:
        print("[GRAPHS] Generating training visualizations...")
        if logger:
            logger.info("Generating training graphs")

        # Setup matplotlib for better plots
        plt.style.use('default')
        fig_size = (15, 10)

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=fig_size)
        fig.suptitle(
            f'Training Results - {model_info.get("model_type", "Unknown")} ({model_info.get("mode", "Unknown")} mode)',
            fontsize=16, fontweight='bold')

        epochs = range(1, len(history['loss']) + 1)

        # 1. Loss curves
        axes[0, 0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Accuracy curves
        if 'per_square_acc' in history:
            axes[0, 1].plot(epochs, history['per_square_acc'], 'b-', label='Training Accuracy', linewidth=2)
            axes[0, 1].plot(epochs, history['val_per_square_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            axes[0, 1].set_title('Per-Square Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Learning rate (if available)
        if 'lr' in history:
            axes[0, 2].plot(epochs, history['lr'], 'g-', linewidth=2)
            axes[0, 2].set_title('Learning Rate Schedule')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Overfitting analysis (loss gap)
        loss_gap = np.array(history['val_loss']) - np.array(history['loss'])
        axes[1, 0].plot(epochs, loss_gap, 'purple', linewidth=2)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Overfitting Analysis (Val Loss - Train Loss)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Gap')
        axes[1, 0].grid(True, alpha=0.3)

        # Color code overfitting regions
        axes[1, 0].fill_between(epochs, loss_gap, 0, where=(loss_gap > 0),
                                color='red', alpha=0.3, label='Overfitting')
        axes[1, 0].fill_between(epochs, loss_gap, 0, where=(loss_gap <= 0),
                                color='green', alpha=0.3, label='Underfitting')
        axes[1, 0].legend()

        # 5. Board-level metrics (if available)
        if 'val_perfect_pct' in history:
            axes[1, 1].plot(epochs, history['val_perfect_pct'], 'gold', label='Perfect Boards %', linewidth=2)
            if 'val_le1_pct' in history:
                axes[1, 1].plot(epochs, history['val_le1_pct'], 'orange', label='≤1 Error %', linewidth=2)
            axes[1, 1].set_title('Board-Level Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Percentage')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Training summary box
        axes[1, 2].axis('off')
        summary_text = f"""
Training Summary:
• Model: {model_info.get('model_type', 'Unknown')}
• Mode: {model_info.get('mode', 'Unknown')}
• Parameters: {model_info.get('total_params', 'Unknown'):,}
• Input Shape: {model_info.get('input_shape', 'Unknown')}
• Batch Size: {model_info.get('batch_size', 'Unknown')}
• Epochs: {model_info.get('epochs', len(epochs))}
• Training Time: {model_info.get('training_time_minutes', 'Unknown'):.1f} min

Final Metrics:
• Train Loss: {history['loss'][-1]:.4f}
• Val Loss: {history['val_loss'][-1]:.4f}
• Val Accuracy: {history['val_per_square_acc'][-1]:.3f}"""

        if 'val_perfect_pct' in history:
            summary_text += f"\n• Perfect Boards: {history['val_perfect_pct'][-1]:.1f}%"

        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        # Adjust layout and save
        plt.tight_layout()

        # Save the comprehensive graph
        graph_filename = f"training_graphs_{model_info.get('model_type', 'model')}_{model_info.get('mode', 'mode')}_{model_info.get('training_date', 'unknown')}.png"
        graph_path = output_dir / graph_filename
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[SUCCESS] Training graphs saved as: {graph_filename}")
        if logger:
            logger.info(f"Training graphs saved as: {graph_filename}")

        # Generate individual plots for specific analysis

        # Loss comparison plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.title(f'Loss Curves - {model_info.get("model_type", "Model")}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        loss_filename = f"loss_curves_{model_info.get('model_type', 'model')}_{model_info.get('training_date', 'unknown')}.png"
        loss_path = output_dir / loss_filename
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[SUCCESS] Loss curves saved as: {loss_filename}")

        return True

    except Exception as e:
        error_msg = f"Failed to generate training graphs: {e}"
        print(f"[ERROR] {error_msg}")
        if logger:
            logger.error(error_msg)
        return False


def configure_gpu_memory():
    """Configure GPU memory growth to prevent OOM errors and disable XLA."""
    try:
        # Disable XLA to avoid compilation issues
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=""'

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Disable XLA JIT compilation
                tf.config.optimizer.set_jit(False)

                message = f"[GPU] Configured memory growth for {len(gpus)} GPU(s), XLA disabled"
                print(message)
                if logger:
                    logger.info(message)

            except RuntimeError as e:
                error_msg = f"GPU memory configuration failed: {e}"
                print(f"[WARNING] {error_msg}")
                if logger:
                    logger.warning(error_msg)
        else:
            message = "No GPUs detected, using CPU"
            print(f"[INFO] {message}")
            if logger:
                logger.info(message)

    except Exception as e:
        error_msg = f"Error during GPU configuration: {e}"
        print(f"[ERROR] {error_msg}")
        if logger:
            logger.error(error_msg)

def prompt_model():
    """Prompt user to select model architecture."""
    print("\nSelect model architecture:")
    print("  [l] light      -> lightweight CNN (~2M params, faster training)")
    print("  [r] resnext    -> ResNeXt-101 32x8d (~88M params, better accuracy)\n")

    while True:
        ans = input("Enter l / r (or press Enter for resnext): ").strip().lower()
        mapping = {"l": "light", "r": "resnext", "": "resnext"}

        if ans in mapping:
            selected = mapping[ans]
            if logger:
                logger.info(f"User selected model architecture: {selected}")
            return selected
        else:
            print("Invalid input. Please enter 'l' for light or 'r' for resnext.")


def prompt_mode():
    """Prompt user to select training mode."""
    print("\nSelect training mode:")
    print("  [q] quick      -> sanity check / tiny run (fast)")
    print("  [t] thorough   -> solid baseline (default)")
    print("  [p] production -> full training (slow)\n")

    while True:
        ans = input("Enter q / t / p (or press Enter for thorough): ").strip().lower()
        mapping = {"q": "quick", "t": "thorough", "p": "production", "": "thorough"}

        if ans in mapping:
            selected = mapping[ans]
            if logger:
                logger.info(f"User selected training mode: {selected}")
            return selected
        else:
            print("Invalid input. Please enter 'q', 't', 'p', or press Enter for default.")


def check_data_availability():
    """Check if training data is available."""
    print("\n[DATA] Checking data availability...")
    if logger:
        logger.info("Checking data availability")

    try:
        # Check if preprocessed data exists on GCS
        if check_preprocessed_data_exists():
            print("[SUCCESS] Preprocessed data found on GCS - ready for training!")
            if logger:
                logger.info("Preprocessed data found on GCS")
            return True
        else:
            print("[ERROR] Preprocessed data not found on GCS!")
            print("Please run preprocess.py first to create the preprocessed data.")
            if logger:
                logger.error("Preprocessed data not found on GCS")
            return False

    except Exception as e:
        error_msg = f"Error checking data availability: {e}"
        print(f"[ERROR] {error_msg}")
        if logger:
            logger.error(error_msg)
        return False


def main():
    global logger

    print("=" * 70)
    print("Enhanced Chess Recognition Training with Pre-download")
    print("=" * 70)

    # Setup logging first
    try:
        log_dir = Path(__file__).resolve().parent / "logs"
        logger = setup_logging(log_dir)
        # Also set the global logger in _helpers
        import _helpers
        _helpers.logger = logger

        logger.info("=" * 50)
        logger.info("Starting enhanced chess recognition training")
        logger.info("=" * 50)

    except Exception as e:
        print(f"[ERROR] Failed to setup logging: {e}")
        return False

    try:
        # Configure GPU memory early
        configure_gpu_memory()

        # Set TensorFlow logging level to reduce noise
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        # Check data availability before proceeding
        if not check_data_availability():
            print("\n[FATAL] Cannot proceed without data. Exiting.")
            logger.error("Training aborted due to missing data")
            return False

        # Get user preferences
        model_type = prompt_model()
        mode = prompt_mode()

        # Get appropriate configuration based on model type
        if model_type == "light":
            cfg = get_light_model_config(mode)
            model_description = f"Light Chess Classifier ({model_type})"
        else:
            cfg = get_mode_config(mode)
            model_description = f"ResNeXt-101 32x8d ({model_type})"

        print(f"\n[MODEL] Selected: {model_description}")
        logger.info(f"Selected model: {model_description}")

        # Setup directories
        cache_dir = Path(__file__).resolve().parent / "gcs_cache"

        if model_type == "light":
            output_dir = Path(
                cfg.get("output_dir", PROJECT_ROOT / "end2end_board_classifier_new_data" / "outputs" / "light_model"))
            analytics_dir = Path(
                cfg.get("analytics_dir", PROJECT_ROOT / "end2end_board_classifier_new_data" / "analytics" / "light"))
        else:
            output_dir = Path(cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
            analytics_dir = Path(cfg.get("analytics_dir", DEFAULT_ANALYTICS_DIR))

        # Update config with final paths
        cfg["output_dir"] = str(output_dir)
        cfg["analytics_dir"] = str(analytics_dir)

        # Prep environment & folders
        seed_everything(cfg["seed"])
        make_dirs([cache_dir, output_dir, analytics_dir])

        print(f"\n[CONFIG] Configuration:")
        print(pretty_cfg(cfg))
        logger.info(f"Training configuration: {json.dumps(cfg, indent=2)}")

        # Download preprocessed data if not already cached
        print(f"\n[CACHE] Setting up local data cache...")

        # Check if we need to download preprocessed data
        required_files = ["class_order.json", "index_train.jsonl", "index_val.jsonl", "index_test.jsonl"]
        missing_files = [f for f in required_files if not (cache_dir / f).exists()]

        if missing_files:
            print(f"[CACHE] Missing preprocessed files: {missing_files}")
            if not download_preprocessed_data(cache_dir):
                raise RuntimeError("Failed to download preprocessed data")
            print(f"[SUCCESS] Preprocessed data cached locally")
        else:
            print(f"[SUCCESS] Preprocessed data already cached")

        # Extract image paths from all JSONL files
        jsonl_files = [cache_dir / "index_train.jsonl", cache_dir / "index_val.jsonl", cache_dir / "index_test.jsonl"]
        image_paths = extract_image_paths_from_jsonl(jsonl_files)

        print(f"[INFO] Found {len(image_paths)} unique images to download")

        # Pre-download all images
        successful_downloads, failed_downloads, failed_urls = pre_download_images(
            image_paths, cache_dir, max_workers=10
        )

        if successful_downloads == 0:
            print("[FATAL] No images were downloaded successfully!")
            logger.error("No images downloaded - cannot proceed with training")
            return False

        if failed_downloads > len(image_paths) * 0.5:  # More than 50% failed
            print(f"[WARNING] High failure rate: {failed_downloads}/{len(image_paths)} failed")
            print("This may affect training quality. Continue? [y/N]")
            if input().strip().lower() not in ['y', 'yes']:
                print("[ABORTED] Training cancelled by user")
                return False

        # Load datasets using cached images
        print("\n[DATA] Creating datasets from cached images...")
        logger.info("Creating datasets from cached images")

        try:
            train_ds, val_ds, steps_per_epoch, val_steps = get_datasets_with_cache(cfg, cache_dir)
            logger.info(
                f"Successfully created datasets - {steps_per_epoch} training steps, {val_steps} validation steps")
        except Exception as e:
            error_msg = f"Failed to create datasets: {e}"
            print(f"[ERROR] {error_msg}")
            logger.error(error_msg)
            raise

        # Force garbage collection after dataset loading
        gc.collect()

        # Build model
        print(f"\n[MODEL] Building {model_type} model...")
        logger.info(f"Building {model_type} model")

        try:
            if model_type == "light":
                model = build_light_chess_classifier(
                    input_shape=tuple(cfg["input_shape"]),
                    num_squares=cfg["num_squares"],
                    num_classes=cfg["num_classes"],
                )
            else:
                model = build_resnext101_32x8d_head(
                    input_shape=tuple(cfg["input_shape"]),
                    cardinality=cfg["cardinality"],
                    base_width=cfg["base_width"],
                    num_squares=cfg["num_squares"],
                    num_classes=cfg["num_classes"],
                )

            logger.info(f"Successfully built model with {model.count_params():,} parameters")

        except Exception as e:
            error_msg = f"Failed to build model: {e}"
            print(f"[ERROR] {error_msg}")
            logger.error(error_msg)
            raise

        # Compile model
        print("\n[COMPILE] Compiling model...")
        logger.info("Compiling model")

        try:
            # Enable mixed precision if available and using large model
            # if model_type == "resnext":
            #     try:
            #         policy = tf.keras.mixed_precision.Policy('mixed_float16')
            #         tf.keras.mixed_precision.set_global_policy(policy)
            #         message = "Mixed precision enabled"
            #         print(f"[INFO] {message}")
            #         logger.info(message)
            #     except Exception as mp_error:
            #         message = f"Mixed precision not available: {mp_error}, using float32"
            #         print(f"[WARNING] {message}")
            #         logger.warning(message)

            # Force float32 for stability
            tf.keras.mixed_precision.set_global_policy('float32')
            print("[INFO] Using float32 precision for stability")

            loss = tf.keras.losses.CategoricalCrossentropy(axis=-1)
            optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["lr_start"])

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=[
                    tf.keras.metrics.CategoricalAccuracy(name="per_square_acc"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top1_per_square"),
                ],
            )

            print(f"\n[INFO] Model Summary:")
            print(f"Total parameters: {model.count_params():,}")
            print(f"Model size: ~{model.count_params() * 4 / 1024 / 1024:.1f} MB")

            logger.info(f"Model compiled - {model.count_params():,} parameters")

        except Exception as e:
            error_msg = f"Failed to compile model: {e}"
            print(f"[ERROR] {error_msg}")
            logger.error(error_msg)
            raise

        # Create callbacks
        print("\n[CALLBACKS] Preparing callbacks...")
        logger.info("Preparing training callbacks")

        try:
            callbacks = make_callbacks(cfg=cfg, output_dir=output_dir, analytics_dir=analytics_dir, val_ds=val_ds)
            logger.info("Successfully created all training callbacks")
        except Exception as e:
            error_msg = f"Failed to create callbacks: {e}"
            print(f"[ERROR] {error_msg}")
            logger.error(error_msg)
            raise

        # Test first batch
        print("\n[TEST] Testing first batch...")
        logger.info("Testing first batch")

        try:
            test_count = 0
            for x_batch, y_batch in train_ds.take(1):
                print(f"Input shape: {x_batch.shape}")
                print(f"Label shape: {y_batch.shape}")
                logger.info(f"First batch - Input: {x_batch.shape}, Labels: {y_batch.shape}")

                _ = model(x_batch, training=False)
                print("[SUCCESS] First batch test successful")
                logger.info("First batch test successful")
                test_count += 1
                break

            if test_count == 0:
                raise RuntimeError("No data in training dataset")

        except Exception as e:
            error_msg = f"First batch test failed: {e}"
            print(f"[ERROR] {error_msg}")
            logger.error(error_msg)
            raise

        # Start training
        print(f"\n[TRAIN] Starting training ({model_type} model, {mode} mode)...")
        logger.info(f"Starting training - {model_type} model, {mode} mode, {cfg['epochs']} epochs")

        t0 = time.time()

        try:
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=cfg["epochs"],
                steps_per_epoch=steps_per_epoch,
                validation_steps=val_steps,
                callbacks=callbacks,
                verbose=cfg["keras_verbose"],
            )

        except tf.errors.ResourceExhaustedError as e:
            error_msg = f"Out of memory error: {e}"
            print(f"\n[ERROR] {error_msg}")
            logger.error(error_msg)

            if model_type == "resnext":
                suggestion = "Try using the light model or reducing batch size"
            else:
                suggestion = "Try reducing batch size or image resolution"
            print(f"[SUGGESTION] {suggestion}")
            logger.error(f"Suggestion: {suggestion}")
            return False

        except Exception as e:
            error_msg = f"Training error: {e}"
            print(f"\n[ERROR] {error_msg}")
            logger.error(error_msg)
            raise

        t1 = time.time()
        training_time = (t1 - t0) / 60
        print(f"\n[DONE] Training finished in {training_time:.1f} min")
        logger.info(f"Training completed in {training_time:.1f} minutes")

        # Save final model
        current_date = datetime.now().strftime("%Y%m%d")
        model_type_name = "large" if model_type == "resnext" else model_type
        final_filename = f"final_{model_type_name}_{mode}_{current_date}.keras"
        final_path = output_dir / final_filename

        print(f"[SAVE] Saving final model...")
        logger.info(f"Saving final model to: {final_path}")

        try:
            model.save(final_path)
            print(f"[SUCCESS] Final model saved as: {final_filename}")
            logger.info(f"Final model saved successfully as: {final_filename}")
        except Exception as e:
            error_msg = f"Could not save final model: {e}"
            print(f"[WARNING] {error_msg}")
            logger.warning(error_msg)

        # Save training history
        hist_filename = f"history_{model_type_name}_{mode}_{current_date}.json"
        hist_path = analytics_dir / hist_filename

        model_info = {
            "model_type": model_type_name,
            "mode": mode,
            "total_params": int(model.count_params()),
            "input_shape": cfg["input_shape"],
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "training_time_minutes": training_time,
            "training_date": current_date,
            "final_model_filename": final_filename,
            "download_stats": {
                "total_images": len(image_paths),
                "successful_downloads": successful_downloads,
                "failed_downloads": failed_downloads,
                "success_rate": (successful_downloads / len(image_paths) * 100) if image_paths else 0
            }
        }

        try:
            history_dict = history.history.copy()
            history_dict["model_info"] = model_info

            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(history_dict, f, indent=2)

            print(f"[SUCCESS] Training history saved as: {hist_filename}")
            logger.info(f"Training history saved as: {hist_filename}")

        except Exception as e:
            error_msg = f"Could not save training history: {e}"
            print(f"[WARNING] {error_msg}")
            logger.warning(error_msg)

        # Generate training graphs
        print(f"\n[GRAPHS] Generating training visualizations...")
        try:
            generate_training_graphs(history.history, analytics_dir, model_info)
        except Exception as e:
            error_msg = f"Failed to generate graphs: {e}"
            print(f"[WARNING] {error_msg}")
            logger.warning(error_msg)

        # Training completion summary
        print(f"\n{'=' * 70}")
        print(f"[SUCCESS] Enhanced Training Completed!")
        print(f"{'=' * 70}")
        print(f"Model: {model_type_name} | Mode: {mode} | Time: {training_time:.1f} min")
        print(f"Parameters: {model.count_params():,}")
        print(f"")
        print(f"Files saved:")
        print(f"  • Model: {output_dir}/best_model.keras (best checkpoint)")
        print(f"  • Model: {final_filename} (final model)")
        print(f"  • History: {hist_filename}")
        print(f"  • Graphs: training_graphs_{model_type_name}_{mode}_{current_date}.png")
        print(f"  • Graphs: loss_curves_{model_type_name}_{current_date}.png")
        print(f"")
        print(f"Download Summary:")
        print(f"  • Images processed: {len(image_paths)}")
        print(f"  • Successful downloads: {successful_downloads}")
        print(f"  • Failed downloads: {failed_downloads}")
        print(f"  • Success rate: {(successful_downloads / len(image_paths) * 100):.1f}%")

        logger.info("=" * 50)
        logger.info("Enhanced training completed successfully")
        logger.info(f"Model: {model_type_name}, Mode: {mode}, Parameters: {model.count_params():,}")
        logger.info(f"Training time: {training_time:.1f} minutes")
        logger.info(f"Download success rate: {(successful_downloads / len(image_paths) * 100):.1f}%")
        logger.info("=" * 50)

        return True

    except Exception as e:
        error_msg = f"Fatal error: {e}"
        print(f"\n[FATAL] {error_msg}")
        if logger:
            logger.error(error_msg)

        import traceback
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")
        if logger:
            logger.error(f"Traceback: {traceback_str}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user")
        if logger:
            logger.info("Training interrupted by user")
        sys.exit(0)

    except Exception as e:
        error_msg = f"Fatal error: {e}"
        print(f"\n[FATAL] {error_msg}")
        sys.exit(1)