# board2fen/end2end_board_classifier/_helpers.py

import math
import json
import logging
import requests
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from urllib.parse import urljoin
import platform
import os
# Hard-disable XLA everywhere
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_XLA"] = "0"
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np


# =========================
# Configuration & Logging
# =========================

def setup_logging(log_dir: Path, log_level: int = logging.INFO) -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger('chess_training')
    logger.setLevel(log_level)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler
    log_file = log_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Global logger (will be initialized in main)
logger = None

# GCS Configuration
GCS_BUCKET_BASE = "https://storage.googleapis.com/chess_red_dataset"
GCS_ANNOTATIONS_URL = f"{GCS_BUCKET_BASE}/annotations.json"
GCS_PREPROCESSED_BASE = f"{GCS_BUCKET_BASE}/chessred_preprocessed"
GCS_IMAGES_BASE = f"{GCS_BUCKET_BASE}/chessred/images"


def get_platform_config():
    """Detect platform and return appropriate settings."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    is_mac_m1_m2 = system == "darwin" and ("arm" in machine or "aarch64" in machine)

    return {
        "is_mac_m1_m2": is_mac_m1_m2,
        "system": system,
        "machine": machine
    }

def get_mode_config(mode: str) -> Dict[str, Any]:
    """
    Returns a dict of hyperparams for 'quick' | 'thorough' | 'production'.
    Tweak to your taste. All modes share the same model; they differ in epochs, batch size, etc.
    """


    base = dict(
        mode=mode,
        seed=42,
        num_squares=64,
        num_classes=13,
        input_shape=(512, 512, 3),  # resize your images to this
        cardinality=32,
        base_width=8,
        keras_verbose=1,
        # training defaults (can be overridden per-mode)
        epochs=200,
        batch_size=4,  # REDUCED from 8 to help with memory
        lr_start=1e-3,
        lr_after=1e-4,  # applied after epoch >= lr_drop_epoch
        lr_drop_epoch=100,
        early_stop_patience=15,
        # dirs
        output_dir=str(Path(__file__).resolve().parents[0] / "outputs" / "best_model"),
        analytics_dir=str(Path(__file__).resolve().parents[0] / "analytics"),
        # dataset config - adapt to your loader
        dataset={
            "train_records": None,  # e.g. path to TFRecords or a JSON index - YOU set this
            "val_records": None,
            "train_size": None,  # number of training samples (for steps/epoch) - YOU set this
            "val_size": None,  # number of validation samples - YOU set this
            "augment": True,
            "prefetch": True,
            "shuffle": True,
            "shuffle_buffer": 512,  # REDUCED from 2048 to help with memory
        }
    )

    if mode == "quick":
        base.update(
            epochs=8,
            batch_size=2,  # Even smaller for quick mode
            lr_drop_epoch=4,
            early_stop_patience=5,
        )
    elif mode == "production":
        base.update(
            epochs=300,
            batch_size=6,  # Slightly larger for production
            lr_drop_epoch=120,
            early_stop_patience=25,
        )
    elif mode == "thorough":
        base.update(
            batch_size=2,
            input_shape=(224, 224, 3),
            shuffle_buffer=128,
        )
    return base


def get_light_model_config(mode: str) -> Dict[str, Any]:
    """
    Configuration for the lightweight model from the notebook.
    Uses smaller input size and different batch sizes optimized for the light architecture.
    """
    base = dict(
        mode=mode,
        seed=42,
        num_squares=64,
        num_classes=13,
        input_shape=(256, 256, 3),  # Smaller input size for light model
        keras_verbose=1,
        # training defaults
        epochs=50,  # Fewer epochs since light model trains faster
        batch_size=8,  # Can use larger batch size due to smaller model
        lr_start=1e-3,
        lr_after=1e-4,
        lr_drop_epoch=30,
        early_stop_patience=10,
        # dirs
        output_dir=str(Path(__file__).resolve().parents[0] / "outputs" / "light_model"),
        analytics_dir=str(Path(__file__).resolve().parents[0] / "analytics"),
        # dataset config
        dataset={
            "train_records": None,
            "val_records": None,
            "train_size": None,
            "val_size": None,
            "augment": True,
            "prefetch": True,
            "shuffle": True,
            "shuffle_buffer": 1024,  # Can afford larger buffer with light model
        }
    )

    if mode == "quick":
        base.update(
            epochs=20,  # More epochs for quick testing
            batch_size=4,
            lr_drop_epoch=10,
            early_stop_patience=5,
        )
    elif mode == "production":
        base.update(
            epochs=100,  # More epochs for production
            batch_size=12,  # Larger batch size
            lr_drop_epoch=60,
            early_stop_patience=15,
        )
    # 'thorough' uses defaults
    return base


def pretty_cfg(cfg: Dict[str, Any]) -> str:
    _cpy = dict(cfg)
    _cpy["dataset"] = {**cfg["dataset"]}  # shallow copy
    return json.dumps(_cpy, indent=2)


def seed_everything(seed: int = 42):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def make_dirs(paths: List[Path]):
    for p in paths:
        try:
            p.mkdir(parents=True, exist_ok=True)
            if logger:
                logger.info(f"Created directory: {p}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to create directory {p}: {e}")
            raise


# =========================
# GCS Download Functions
# =========================

def download_file_with_retry(url: str, max_retries: int = 3, timeout: int = 30) -> requests.Response:
    """Download file from URL with retry logic and comprehensive error handling."""
    for attempt in range(max_retries):
        try:
            if logger:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")

            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            if logger:
                logger.info(f"Successfully downloaded {url}")
            return response

        except requests.exceptions.Timeout as e:
            if logger:
                logger.warning(f"Timeout downloading {url}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

        except requests.exceptions.ConnectionError as e:
            if logger:
                logger.warning(f"Connection error downloading {url}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

        except requests.exceptions.HTTPError as e:
            if logger:
                logger.error(f"HTTP error downloading {url}: {e}")
            raise  # Don't retry HTTP errors

        except Exception as e:
            if logger:
                logger.error(f"Unexpected error downloading {url}: {e}")
            raise


def download_file_quietly(url: str, max_retries: int = 3, timeout: int = 30) -> requests.Response:
    """Download file from URL quietly without logging progress."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            return response

        except requests.exceptions.Timeout as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

        except requests.exceptions.ConnectionError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

        except requests.exceptions.HTTPError as e:
            raise  # Don't retry HTTP errors

        except Exception as e:
            raise


def check_preprocessed_data_exists() -> bool:
    """Check if preprocessed data exists on GCS."""
    required_files = [
        "class_order.json",
        "index_train.jsonl",
        "index_val.jsonl",
        "index_test.jsonl"
    ]

    print("[GCS] Checking if preprocessed data exists...")
    if logger:
        logger.info("Checking if preprocessed data exists on GCS")

    for file_name in required_files:
        url = f"{GCS_PREPROCESSED_BASE}/{file_name}"
        try:
            response = requests.head(url, timeout=10)
            if response.status_code != 200:
                print(f"[GCS] Missing file: {file_name}")
                if logger:
                    logger.info(f"Preprocessed file not found: {file_name}")
                return False
        except Exception as e:
            print(f"[GCS] Error checking {file_name}: {e}")
            if logger:
                logger.warning(f"Error checking preprocessed file {file_name}: {e}")
            return False

    print("[GCS] All preprocessed files found!")
    if logger:
        logger.info("All preprocessed files found on GCS")
    return True


def download_preprocessed_data(local_dir: Path) -> bool:
    """Download preprocessed data from GCS."""
    try:
        local_dir.mkdir(parents=True, exist_ok=True)

        files_to_download = [
            "class_order.json",
            "index_train.jsonl",
            "index_val.jsonl",
            "index_test.jsonl"
        ]

        print("[GCS] Downloading preprocessed data...")
        if logger:
            logger.info("Downloading preprocessed data from GCS")

        for file_name in files_to_download:
            url = f"{GCS_PREPROCESSED_BASE}/{file_name}"
            local_path = local_dir / file_name

            try:
                response = download_file_with_retry(url)
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"[GCS] Downloaded: {file_name}")
                if logger:
                    logger.info(f"Downloaded preprocessed file: {file_name}")

            except Exception as e:
                print(f"[ERROR] Failed to download {file_name}: {e}")
                if logger:
                    logger.error(f"Failed to download {file_name}: {e}")
                return False

        print("[GCS] Preprocessed data download complete!")
        if logger:
            logger.info("Preprocessed data download complete")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to download preprocessed data: {e}")
        if logger:
            logger.error(f"Failed to download preprocessed data: {e}")
        return False


# =========================
# Model Architectures
# =========================

def conv_bn_act(x, filters, kernel_size, strides=1, groups=1, use_bias=False, name=None):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same",
                      use_bias=use_bias, groups=groups, name=None if name is None else name + "_conv")(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_bn")(x)
    x = layers.Activation("relu", name=None if name is None else name + "_relu")(x)
    return x


def resnext_bottleneck(x, out_channels, stride, cardinality=32, base_width=8, name=None):
    """
    ResNeXt bottleneck block (expansion=4).
    """
    shortcut = x
    in_channels = x.shape[-1]
    expansion = 4
    bottleneck_channels = out_channels // expansion

    # group width scaling from ResNeXt paper
    D = int(math.floor(bottleneck_channels * (base_width / 64.0)))
    group_width = max(D * cardinality, 1)

    # 1x1 reduce
    x = conv_bn_act(x, group_width, 1, strides=1, groups=1, name=None if name is None else name + "_conv1")
    # 3x3 grouped conv
    x = conv_bn_act(x, group_width, 3, strides=stride, groups=cardinality,
                    name=None if name is None else name + "_conv2")
    # 1x1 expand
    x = layers.Conv2D(out_channels, 1, padding="same", use_bias=False, name=None if name is None else name + "_conv3")(
        x)
    x = layers.BatchNormalization(name=None if name is None else name + "_bn3")(x)

    # projection if channel or stride mismatch
    if (in_channels != out_channels) or (stride != 1):
        shortcut = layers.Conv2D(out_channels, 1, strides=stride, padding="same", use_bias=False,
                                 name=None if name is None else name + "_proj_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=None if name is None else name + "_proj_bn")(shortcut)

    x = layers.Add(name=None if name is None else name + "_add")([x, shortcut])
    x = layers.Activation("relu", name=None if name is None else name + "_out")(x)
    return x


def resnext_stage(x, out_channels, blocks, stride1, cardinality=32, base_width=8, name=None):
    x = resnext_bottleneck(x, out_channels, stride1, cardinality, base_width,
                           name=None if name is None else name + "_block1")
    for i in range(2, blocks + 1):
        x = resnext_bottleneck(x, out_channels, 1, cardinality, base_width,
                               name=None if name is None else f"{name}_block{i}")
    return x


def build_resnext101_32x8d_head(input_shape=(512, 512, 3), cardinality=32, base_width=8,
                                num_squares=64, num_classes=13) -> tf.keras.Model:
    """
    End-to-end classifier: predicts (64, 13) softmax (per square).
    """
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="stem_pool")(x)

    # ResNet-101 layout: [3, 4, 23, 3]
    x = resnext_stage(x, out_channels=256, blocks=3, stride1=1, cardinality=cardinality, base_width=base_width,
                      name="conv2")
    x = resnext_stage(x, out_channels=512, blocks=4, stride1=2, cardinality=cardinality, base_width=base_width,
                      name="conv3")
    x = resnext_stage(x, out_channels=1024, blocks=23, stride1=2, cardinality=cardinality, base_width=base_width,
                      name="conv4")
    x = resnext_stage(x, out_channels=2048, blocks=3, stride1=2, cardinality=cardinality, base_width=base_width,
                      name="conv5")

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(num_squares * num_classes, name="square_logits")(x)
    x = layers.Reshape((num_squares, num_classes), name="reshape_to_squares")(x)
    outputs = layers.Activation("softmax", name="per_square_softmax")(x)

    return models.Model(inputs, outputs, name="ResNeXt101_32x8d_ChessSquares")


def build_light_chess_classifier(input_shape=(256, 256, 3), num_squares=64, num_classes=13) -> tf.keras.Model:
    """
    Lightweight CNN more suitable for fast training and resource-constrained environments.
    Only ~2M parameters vs 88M for ResNeXt-101.
    Based on the simplified architecture from the Colab notebook.
    """
    inputs = layers.Input(shape=input_shape)

    # Stem with aggressive downsampling
    x = layers.Conv2D(32, 7, strides=2, padding="same", name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)

    # Simple conv blocks
    x = layers.Conv2D(64, 3, padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)

    x = layers.Conv2D(128, 3, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="conv2_bn")(x)
    x = layers.Activation("relu", name="conv2_relu")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)

    x = layers.Conv2D(256, 3, padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="conv3_bn")(x)
    x = layers.Activation("relu", name="conv3_relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Chess board head
    x = layers.Dense(512, activation="relu", name="head_dense1")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    x = layers.Dense(num_squares * num_classes, name="square_logits")(x)
    x = layers.Reshape((num_squares, num_classes), name="reshape_to_squares")(x)
    outputs = layers.Activation("softmax", name="per_square_softmax")(x)

    model = models.Model(inputs, outputs, name="LightChessClassifier")

    if logger:
        logger.info(f"Light model parameters: {model.count_params():,}")
    else:
        print(f"[INFO] Light model parameters: {model.count_params():,}")

    return model


# =========================
# LR Schedule & Callbacks
# =========================

def _step_lr_factory(lr_start: float, lr_after: float, drop_epoch: int):
    def step_lr(epoch, lr):
        return lr_after if epoch >= drop_epoch else lr_start

    return step_lr


class BoardLevelEvalCallback(callbacks.Callback):
    """
    After each epoch, compute board-level metrics on a given validation dataset:
      - perfect boards (0 wrong squares)
      - boards with <=1 wrong square
      - mean wrong squares / board
      - per-square error rate
    Assumes y_true & y_pred are shaped (B, 64, 13).
    """

    def __init__(self, val_ds, name_prefix="val"):
        super().__init__()
        self.val_ds = val_ds
        self.prefix = name_prefix

    @staticmethod
    def _stats(y_true, y_pred):
        y_true_idx = np.argmax(y_true, axis=-1)  # (B,64)
        y_pred_idx = np.argmax(y_pred, axis=-1)  # (B,64)
        wrong_per_board = np.sum(y_true_idx != y_pred_idx, axis=1)  # (B,)
        perfect = np.mean(wrong_per_board == 0) * 100.0
        le1 = np.mean(wrong_per_board <= 1) * 100.0
        mean_wrong = np.mean(wrong_per_board)
        per_square_err = np.mean(y_true_idx != y_pred_idx) * 100.0
        return dict(perfect=perfect, le1=le1, mean_wrong=mean_wrong, per_square_err=per_square_err)

    def on_epoch_end(self, epoch, logs=None):
        try:
            # Use a smaller subset for evaluation to save memory
            ys_true, ys_pred = [], []
            count = 0
            for xb, yb in self.val_ds.take(10):  # Only take first 10 batches for evaluation
                pred = self.model.predict(xb, verbose=0)
                ys_true.append(yb.numpy() if hasattr(yb, "numpy") else yb)
                ys_pred.append(pred)
                count += 1
                if count >= 10:  # Limit to prevent memory issues
                    break

            if ys_true:  # Only proceed if we have data
                y_true = np.concatenate(ys_true, axis=0)
                y_pred = np.concatenate(ys_pred, axis=0)
                s = self._stats(y_true, y_pred)
                msg = (f"[EVAL] {self.prefix} - perfect: {s['perfect']:.2f}% | <=1 mistake: {s['le1']:.2f}% | "
                       f"mean wrong squares: {s['mean_wrong']:.2f} | per-square err: {s['per_square_err']:.2f}%")
                print("\n" + msg)
                if logger:
                    logger.info(f"Board evaluation - {msg}")

                # put into logs so CSVLogger captures it
                if logs is not None:
                    logs[f"{self.prefix}_perfect_pct"] = s["perfect"]
                    logs[f"{self.prefix}_le1_pct"] = s["le1"]
                    logs[f"{self.prefix}_mean_wrong"] = s["mean_wrong"]
                    logs[f"{self.prefix}_per_square_err_pct"] = s["per_square_err"]

        except Exception as e:
            error_msg = f"Error in board level evaluation: {e}"
            print(f"[ERROR] {error_msg}")
            if logger:
                logger.error(error_msg)


def make_callbacks(cfg: Dict[str, Any], output_dir: Path, analytics_dir: Path, val_ds: tf.data.Dataset):
    try:
        ckpt_path = output_dir / "best_model.keras"
        csv_path = analytics_dir / f"train_log_{cfg['mode']}.csv"
        tb_dir = analytics_dir / f"tb_{cfg['mode']}"

        lr_cb = callbacks.LearningRateScheduler(
            _step_lr_factory(cfg["lr_start"], cfg["lr_after"], cfg["lr_drop_epoch"]), verbose=0
        )
        ckpt_cb = callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_per_square_acc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
        es_cb = callbacks.EarlyStopping(
            monitor="val_per_square_acc",
            mode="max",
            patience=cfg["early_stop_patience"],
            restore_best_weights=True,
            verbose=1,
        )
        csv_cb = callbacks.CSVLogger(str(csv_path), append=False)
        tb_cb = callbacks.TensorBoard(log_dir=str(tb_dir))
        board_eval = BoardLevelEvalCallback(val_ds, name_prefix="val")

        if logger:
            logger.info("Successfully created all training callbacks")

        return [lr_cb, ckpt_cb, es_cb, csv_cb, tb_cb, board_eval]

    except Exception as e:
        error_msg = f"Failed to create callbacks: {e}"
        print(f"[ERROR] {error_msg}")
        if logger:
            logger.error(error_msg)
        raise


# =========================
# Data loading with GCS Support
# =========================

def load_class_order_from_gcs(local_dir: Path) -> List[str]:
    """Load class_order.json from local cache (downloaded from GCS)."""
    class_order_path = local_dir / "class_order.json"
    try:
        if not class_order_path.exists():
            raise FileNotFoundError(f"{class_order_path} not found")

        with open(class_order_path, 'r', encoding='utf-8') as f:
            class_order = json.load(f)

        if logger:
            logger.info(f"Loaded class order with {len(class_order)} classes: {class_order}")

        return class_order

    except Exception as e:
        error_msg = f"Failed to load class order: {e}"
        print(f"[ERROR] {error_msg}")
        if logger:
            logger.error(error_msg)
        raise


def find_empty_index(class_order: List[str]) -> int:
    """Return index of 'empty' (case-insensitive) in class_order."""
    for i, name in enumerate(class_order):
        if name.strip().lower() == "empty":
            return i
    raise ValueError("'empty' class not found in class_order.json")


def _augmentor(image, label):
    # Simplified augmentation to reduce memory usage
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.05)  # Reduced intensity
    return image, label


def _read_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    except Exception as e:
        error_msg = f"Error reading JSONL file {path}: {e}"
        print(f"[ERROR] {error_msg}")
        if logger:
            logger.error(error_msg)
        raise


def load_image_from_gcs_url(url: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Load image from GCS URL with error handling and retries."""
    for attempt in range(3):
        try:
            response = download_file_quietly(url, max_retries=2, timeout=15)
            image_bytes = response.content

            # Decode image
            image = tf.io.decode_jpeg(image_bytes, channels=3)
            image = tf.image.resize(image, target_size, method="bilinear")
            image = tf.cast(image, tf.float32) / 255.0

            # Validate image
            if tf.reduce_mean(image) > 0.001:  # Basic validation
                return image.numpy()
            else:
                raise ValueError("Image appears to be invalid (all zeros)")

        except Exception as e:
            if logger:
                logger.warning(f"Failed to load image {url} (attempt {attempt + 1}): {e}")
            if attempt == 2:  # Last attempt
                # Return fallback pattern
                fallback = np.zeros((*target_size, 3), dtype=np.float32)
                fallback[::16, ::16] = 0.5  # Checkerboard pattern
                if logger:
                    logger.warning(f"Using fallback image for {url}")
                return fallback
            time.sleep(0.5)

    # Should not reach here, but just in case
    return np.zeros((*target_size, 3), dtype=np.float32)


def get_datasets(cfg: Dict[str, Any]) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Load datasets from GCS-based preprocessed data."""
    try:
        # Use local cache directory
        cache_dir = Path(__file__).resolve().parents[0] / "gcs_cache"

        # Check if we have preprocessed data on GCS, download if available
        if check_preprocessed_data_exists():
            if not download_preprocessed_data(cache_dir):
                raise RuntimeError("Failed to download preprocessed data from GCS")
        else:
            raise FileNotFoundError("Preprocessed data not found on GCS. Please run preprocessing first.")

        # Now load from local cache
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
            if logger:
                logger.warning(f"Overriding num_classes from {cfg.get('num_classes')} to {num_classes_file}")
            cfg["num_classes"] = num_classes_file

        # Read JSONL files
        train_rows = list(_read_jsonl(idx_train))
        val_rows = list(_read_jsonl(idx_val))

        print(f"[SUCCESS] Loaded {len(train_rows)} training and {len(val_rows)} validation samples")
        if logger:
            logger.info(f"Loaded {len(train_rows)} training and {len(val_rows)} validation samples")

        bs = cfg["batch_size"]
        size = cfg["input_shape"]
        num_squares = cfg["num_squares"]
        num_classes = cfg["num_classes"]
        shuffle_buffer = cfg["dataset"].get("shuffle_buffer", 512)

        # Create generators
        def train_generator():
            for row in train_rows:
                # Convert local file path to GCS URL
                file_path = row["file_path"]
                if file_path.startswith("C:") or file_path.startswith("/"):
                    # Convert local path to GCS URL
                    # Extract just the folder number and filename (skip "images" part)
                    path_parts = file_path.replace("\\", "/").split("/")
                    # Find the "images" part and take everything after it
                    try:
                        images_idx = path_parts.index("images")
                        rel_path_parts = path_parts[images_idx + 1:]  # Skip "images", take folder/file
                        gcs_url = f"{GCS_IMAGES_BASE}/{'/'.join(rel_path_parts)}"
                    except ValueError:
                        # Fallback: take last 2 parts (folder/file)
                        rel_path_parts = path_parts[-2:]
                        gcs_url = f"{GCS_IMAGES_BASE}/{'/'.join(rel_path_parts)}"

                    if logger:
                        logger.debug(f"Converted path: {file_path} -> {gcs_url}")
                else:
                    # Already a URL
                    gcs_url = file_path
                yield (gcs_url, row["labels_sparse"])

        def val_generator():
            for row in val_rows:
                file_path = row["file_path"]
                if file_path.startswith("C:") or file_path.startswith("/"):
                    # Convert local path to GCS URL (same logic as train)
                    path_parts = file_path.replace("\\", "/").split("/")
                    try:
                        images_idx = path_parts.index("images")
                        rel_path_parts = path_parts[images_idx + 1:]  # Skip "images", take folder/file
                        gcs_url = f"{GCS_IMAGES_BASE}/{'/'.join(rel_path_parts)}"
                    except ValueError:
                        # Fallback: take last 2 parts (folder/file)
                        rel_path_parts = path_parts[-2:]
                        gcs_url = f"{GCS_IMAGES_BASE}/{'/'.join(rel_path_parts)}"
                else:
                    gcs_url = file_path
                yield (gcs_url, row["labels_sparse"])

        # Define output signature
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),  # GCS URL
            tf.TensorSpec(shape=(None, 2), dtype=tf.int32),  # sparse labels
        )

        # Create datasets
        train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=output_signature)
        val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=output_signature)

        # Parse function with GCS image loading
        def parse_sample(gcs_url, labels_sparse):
            # Load image from GCS
            image_loader_fn = lambda url: tf.py_function(
                lambda u: load_image_from_gcs_url(u.numpy().decode('utf-8'), size[:2]),
                [url], tf.float32
            )
            image = image_loader_fn(gcs_url)
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
        train_ds = train_ds.map(parse_sample, num_parallel_calls=2)
        val_ds = val_ds.map(parse_sample, num_parallel_calls=2)

        # Apply augmentations
        if cfg["dataset"]["augment"]:
            train_ds = train_ds.map(_augmentor, num_parallel_calls=2)

        # Shuffle, batch, prefetch
        if cfg["dataset"]["shuffle"]:
            train_ds = train_ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

        train_ds = train_ds.batch(bs, drop_remainder=False)
        val_ds = val_ds.batch(bs, drop_remainder=False)

        train_ds = train_ds.prefetch(2)
        val_ds = val_ds.prefetch(2)

        steps_per_epoch = max(1, len(train_rows) // bs)
        val_steps = max(1, len(val_rows) // bs)

        print(f"[SUCCESS] Created datasets - Steps/epoch: {steps_per_epoch}, Val steps: {val_steps}")
        if logger:
            logger.info(f"Created datasets - Steps/epoch: {steps_per_epoch}, Val steps: {val_steps}")

        return train_ds, val_ds, steps_per_epoch, val_steps

    except Exception as e:
        error_msg = f"Failed to create datasets: {e}"
        print(f"[ERROR] {error_msg}")
        if logger:
            logger.error(error_msg)
        raise


# =========================
# GCS Model Upload Functions
# =========================

def upload_to_gcs_bucket(
        local_file_path: Path,
        bucket_name: str,
        destination_blob_name: str,
        credentials_path: Optional[str] = None
) -> bool:
    """
    Upload a file to Google Cloud Storage bucket.

    Args:
        local_file_path: Path to local file
        bucket_name: Name of the GCS bucket (without gs://)
        destination_blob_name: Path/name in the bucket
        credentials_path: Optional path to service account JSON (if not using default auth)

    Returns:
        True if successful, False otherwise
    """
    try:
        from google.cloud import storage
    except ImportError:
        print("[ERROR] google-cloud-storage not installed. Run: pip install google-cloud-storage")
        if logger:
            logger.error("google-cloud-storage library not installed")
        return False

    try:
        # Initialize client
        if credentials_path and Path(credentials_path).exists():
            client = storage.Client.from_service_account_json(credentials_path)
            if logger:
                logger.info(f"Using service account credentials: {credentials_path}")
        else:
            client = storage.Client()  # Uses default credentials
            if logger:
                logger.info("Using default Google Cloud credentials")

        # Get bucket
        bucket = client.bucket(bucket_name)

        # Create blob and upload
        blob = bucket.blob(destination_blob_name)

        print(f"[UPLOAD] Uploading {local_file_path.name} to gs://{bucket_name}/{destination_blob_name}")
        if logger:
            logger.info(f"Starting upload of {local_file_path.name} to gs://{bucket_name}/{destination_blob_name}")

        # Upload with progress
        file_size = local_file_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"[UPLOAD] File size: {file_size:.1f} MB")

        blob.upload_from_filename(str(local_file_path))

        print(f"[SUCCESS] Uploaded to gs://{bucket_name}/{destination_blob_name}")
        if logger:
            logger.info(f"Successfully uploaded {local_file_path.name} to gs://{bucket_name}/{destination_blob_name}")

        return True

    except Exception as e:
        error_msg = f"Failed to upload {local_file_path.name}: {e}"
        print(f"[ERROR] {error_msg}")
        if logger:
            logger.error(error_msg)

        # Provide helpful error messages
        if "403" in str(e):
            print("[HINT] Permission denied. Check if you have write access to the bucket.")
            print("[HINT] Run: gcloud auth application-default login")
        elif "404" in str(e):
            print(f"[HINT] Bucket '{bucket_name}' not found. Check the bucket name.")
        elif "credentials" in str(e).lower():
            print("[HINT] Authentication issue. Run: gcloud auth application-default login")

        return False


def prompt_upload_models(output_dir: Path, analytics_dir: Path, final_filename: str, hist_filename: str) -> bool:
    """
    Prompt user if they want to upload trained models to GCS.
    """
    print(f"\n{'=' * 50}")
    print(f"[UPLOAD] Upload trained models to Google Cloud Storage?")
    print(f"{'=' * 50}")
    print(f"Files ready for upload:")
    print(f"  • {final_filename}")
    print(f"  • best_model.keras")
    print(f"  • {hist_filename}")
    print(f"")

    # Check file sizes
    total_size = 0
    files_info = []

    files_to_check = [
        (output_dir / final_filename, final_filename),
        (output_dir / "best_model.keras", "best_model.keras"),
        (analytics_dir / hist_filename, hist_filename),
    ]

    for file_path, file_name in files_to_check:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            files_info.append((file_path, file_name, size_mb))
            print(f"  • {file_name}: {size_mb:.1f} MB")
        else:
            print(f"  • {file_name}: NOT FOUND")

    print(f"  Total size: {total_size:.1f} MB")
    print(f"")

    # Prompt user
    while True:
        answer = input("Upload to GCS? [y/N]: ").strip().lower()
        if answer in ['n', 'no', '']:
            print("[SKIP] Skipping upload")
            if logger:
                logger.info("User chose to skip model upload")
            return False
        elif answer in ['y', 'yes']:
            break
        else:
            print("Please enter 'y' for yes or 'n' for no (or press Enter for no)")

    # Get bucket info
    print(f"\n[CONFIG] Upload configuration:")
    bucket_name = input("Enter destination bucket name (without gs://): ").strip()
    if not bucket_name:
        print("[ERROR] No bucket name provided")
        if logger:
            logger.warning("Upload cancelled - no bucket name provided")
        return False

    # Get optional base path
    print("Enter base path in bucket (optional):")
    print("  Examples: 'chess-models', 'experiments/chess', 'models/2024'")
    base_path = input("Base path (press Enter for root): ").strip()

    # Optional credentials path
    creds_path = input("Service account JSON path (press Enter for default auth): ").strip()

    print(f"\n[UPLOAD] Starting upload to gs://{bucket_name}/")
    if logger:
        logger.info(f"Starting model upload to bucket: {bucket_name}, base_path: {base_path}")

    # Upload files
    success_count = 0
    for file_path, file_name, size_mb in files_info:
        if base_path:
            gcs_path = f"{base_path.rstrip('/')}/{file_name}"
        else:
            gcs_path = file_name

        try:
            if upload_to_gcs_bucket(file_path, bucket_name, gcs_path, creds_path if creds_path else None):
                success_count += 1
            else:
                print(f"[FAILED] Could not upload {file_name}")
        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] Upload cancelled by user")
            if logger:
                logger.info("Upload interrupted by user")
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error uploading {file_name}: {e}")
            if logger:
                logger.error(f"Unexpected error uploading {file_name}: {e}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"[COMPLETE] Upload Summary")
    print(f"{'=' * 50}")
    print(f"Successfully uploaded: {success_count}/{len(files_info)} files")
    if success_count > 0:
        print(f"Destination: gs://{bucket_name}/{base_path if base_path else ''}")
        print(f"Total uploaded: {sum(info[2] for info in files_info[:success_count]):.1f} MB")
    print(f"")

    if logger:
        logger.info(f"Upload completed - {success_count}/{len(files_info)} files successful")

    return success_count > 0


def check_gcs_upload_requirements() -> bool:
    """Check if GCS upload requirements are met."""
    try:
        from google.cloud import storage

        # Try to create a client to test authentication
        try:
            client = storage.Client()
            # Test authentication by listing accessible projects
            _ = client.project
            return True
        except Exception as e:
            print(f"[WARNING] GCS authentication issue: {e}")
            print("[HINT] Run: gcloud auth application-default login")
            if logger:
                logger.warning(f"GCS authentication issue: {e}")
            return False

    except ImportError:
        print("[WARNING] google-cloud-storage not installed")
        print("[INSTALL] Run: pip install google-cloud-storage")
        if logger:
            logger.warning("google-cloud-storage library not installed")
        return False