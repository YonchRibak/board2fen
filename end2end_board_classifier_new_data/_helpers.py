# board2fen/end2end_board_classifier/_helpers.py
import os
import math
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np


# =========================
# Configuration & Utilities
# =========================

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
        p.mkdir(parents=True, exist_ok=True)


# =========================
# Model: ResNeXt-101 (32x8d)
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
            # put into logs so CSVLogger captures it
            if logs is not None:
                logs[f"{self.prefix}_perfect_pct"] = s["perfect"]
                logs[f"{self.prefix}_le1_pct"] = s["le1"]
                logs[f"{self.prefix}_mean_wrong"] = s["mean_wrong"]
                logs[f"{self.prefix}_per_square_err_pct"] = s["per_square_err"]


def make_callbacks(cfg: Dict[str, Any], output_dir: Path, analytics_dir: Path, val_ds: tf.data.Dataset):
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

    return [lr_cb, ckpt_cb, es_cb, csv_cb, tb_cb, board_eval]


# =========================
# Data loading - OPTIMIZED
# =========================

def load_class_order(pp_dir: Path) -> List[str]:
    """Load class_order.json (written by preprocess.py) and return the list of class names in order."""
    p = pp_dir / "class_order.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run preprocess.py first.")
    return json.loads(p.read_text(encoding="utf-8"))


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
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# OPTIMIZED: Pure TensorFlow implementation without tf.py_function
# Removed create_sparse_to_dense_map as we're using tf.py_function approach for variable-length data

def get_datasets(cfg: Dict[str, Any]) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    pp_dir = Path(r"C:/datasets/ChessReD/chessred_preprocessed")
    idx_train = pp_dir / "index_train.jsonl"
    idx_val = pp_dir / "index_val.jsonl"
    class_order_path = pp_dir / "class_order.json"

    if not idx_train.exists() or not idx_val.exists() or not class_order_path.exists():
        raise FileNotFoundError(f"Missing preprocessed files under {pp_dir}. Run preprocess.py first.")

    # --- class order ---
    class_order = load_class_order(pp_dir)
    num_classes_file = len(class_order)
    empty_idx = find_empty_index(class_order)
    if cfg.get("num_classes") != num_classes_file:
        print(
            f"[WARNING] Overriding cfg['num_classes'] from {cfg.get('num_classes')} -> {num_classes_file} (from class_order.json)")
        cfg["num_classes"] = num_classes_file

    # --- read JSONL into memory ---
    train_rows = list(_read_jsonl(idx_train))
    val_rows = list(_read_jsonl(idx_val))

    print(f"Loading {len(train_rows)} training samples and {len(val_rows)} validation samples")

    bs = cfg["batch_size"]
    size = cfg["input_shape"]
    num_squares = cfg["num_squares"]
    num_classes = cfg["num_classes"]
    shuffle_buffer = cfg["dataset"].get("shuffle_buffer", 512)

    # FIXED: Use generators to handle variable-length sparse labels
    def train_generator():
        for row in train_rows:
            yield (row["file_path"], row["labels_sparse"])

    def val_generator():
        for row in val_rows:
            yield (row["file_path"], row["labels_sparse"])

    # Define output signature for variable-length data
    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),  # file_path
        tf.TensorSpec(shape=(None, 2), dtype=tf.int32),  # variable-length sparse labels
    )

    # Create datasets from generators
    train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=output_signature)
    val_ds = tf.data.Dataset.from_generator(val_generator, output_signature=output_signature)

    # FIXED: Simplified parsing function using tf.py_function for sparse-to-dense conversion
    def parse_sample(file_path, labels_sparse):
        # Load and preprocess image
        image_bytes = tf.io.read_file(file_path)
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, size[:2], method="bilinear")
        image = tf.cast(image, tf.float32) / 255.0

        # Convert sparse labels to dense using tf.py_function
        def sparse_to_dense_py(sparse_pairs):
            sparse_pairs = sparse_pairs.numpy()
            dense = np.zeros((num_squares, num_classes), dtype=np.float32)
            # Initialize all squares as empty
            dense[:, empty_idx] = 1.0
            # Set pieces
            for sq_idx, cls_idx in sparse_pairs:
                if 0 <= sq_idx < num_squares and 0 <= cls_idx < num_classes:
                    dense[sq_idx, :] = 0.0
                    dense[sq_idx, cls_idx] = 1.0
            return dense

        dense_labels = tf.py_function(
            sparse_to_dense_py,
            [labels_sparse],
            tf.float32
        )
        dense_labels.set_shape((num_squares, num_classes))

        return image, dense_labels

    # Apply parsing with controlled parallelism
    train_ds = train_ds.map(parse_sample, num_parallel_calls=2)
    val_ds = val_ds.map(parse_sample, num_parallel_calls=2)

    # Apply augmentations
    if cfg["dataset"]["augment"]:
        train_ds = train_ds.map(_augmentor, num_parallel_calls=2)

    # Shuffle, batch, and prefetch
    if cfg["dataset"]["shuffle"]:
        train_ds = train_ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    train_ds = train_ds.batch(bs, drop_remainder=False)
    val_ds = val_ds.batch(bs, drop_remainder=False)

    # Prefetch with limited buffer
    train_ds = train_ds.prefetch(2)
    val_ds = val_ds.prefetch(2)

    steps_per_epoch = max(1, len(train_rows) // bs)
    val_steps = max(1, len(val_rows) // bs)
    print(f"[SUCCESS] Train samples: {len(train_rows)} | Val samples: {len(val_rows)}")
    print(f"[SUCCESS] Steps/epoch: {steps_per_epoch} | Val steps: {val_steps}")
    return train_ds, val_ds, steps_per_epoch, val_steps