# board2fen/end2end_board_classifier/train.py
import os
import sys
import time
import json
import gc
from pathlib import Path

import tensorflow as tf

from _helpers import (
    get_mode_config,
    seed_everything,
    build_resnext101_32x8d_head,
    make_dirs,
    make_callbacks,
    get_datasets,
    pretty_cfg,
    BoardLevelEvalCallback,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "end2end_board_classifier_new_data" / "outputs" / "best_model"
DEFAULT_ANALYTICS_DIR = PROJECT_ROOT / "end2end_board_classifier_new_data" / "analytics"


def configure_gpu_memory():
    """Configure GPU memory growth to prevent OOM errors."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] Configured memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"[WARNING] GPU memory configuration failed: {e}")
    else:
        print("[INFO] No GPUs detected, using CPU")


def prompt_mode():
    print("\nSelect training mode:")
    print("  [q] quick      -> sanity check / tiny run (fast)")
    print("  [t] thorough   -> solid baseline (default)")
    print("  [p] production -> full training (slow)\n")
    ans = input("Enter q / t / p (or press Enter for thorough): ").strip().lower()
    mapping = {"q": "quick", "t": "thorough", "p": "production", "": "thorough"}
    if ans not in mapping:
        print("Unrecognized input. Falling back to 'thorough'.")
        ans = ""
    return mapping[ans]


def main():
    print("=" * 70)
    print("End-to-End Chess Recognition - ResNeXt101 (32x8d) Training")
    print("=" * 70)

    # Configure GPU memory early
    configure_gpu_memory()

    # Set TensorFlow logging level to reduce noise
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    mode = prompt_mode()
    cfg = get_mode_config(mode)

    # Paths
    output_dir = Path(cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    analytics_dir = Path(cfg.get("analytics_dir", DEFAULT_ANALYTICS_DIR))

    # Prep environment & folders
    seed_everything(cfg["seed"])
    make_dirs([output_dir, analytics_dir])

    print("\n[CONFIG] Configuration:")
    print(pretty_cfg(cfg))

    try:
        # Datasets
        print("\n[DATA] Loading datasets...")
        train_ds, val_ds, steps_per_epoch, val_steps = get_datasets(cfg)

        # Force garbage collection after dataset loading
        gc.collect()

        # Build model
        print("\n[MODEL] Building model...")
        model = build_resnext101_32x8d_head(
            input_shape=tuple(cfg["input_shape"]),
            cardinality=cfg["cardinality"],
            base_width=cfg["base_width"],
            num_squares=cfg["num_squares"],
            num_classes=cfg["num_classes"],
        )

        # Compile with mixed precision for better memory efficiency
        print("\n[COMPILE] Compiling model...")

        # Enable mixed precision if available
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("[INFO] Mixed precision enabled")
        except:
            print("[WARNING] Mixed precision not available, using float32")

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

        # Callbacks: checkpoint, early stop, LR scheduler, CSV log, tensorboard, board-level eval
        print("\n[CALLBACKS] Preparing callbacks...")
        callbacks = make_callbacks(
            cfg=cfg,
            output_dir=output_dir,
            analytics_dir=analytics_dir,
            val_ds=val_ds
        )

        # Test the first batch to catch errors early
        print("\n[TEST] Testing first batch...")
        try:
            for x_batch, y_batch in train_ds.take(1):
                print(f"Input shape: {x_batch.shape}")
                print(f"Label shape: {y_batch.shape}")
                _ = model(x_batch, training=False)  # Test forward pass
                print("[SUCCESS] First batch test successful")
                break
        except Exception as e:
            print(f"[ERROR] First batch test failed: {e}")
            raise

        print("\n[TRAIN] Starting training...")
        t0 = time.time()

        # Use try-except to catch memory errors during training
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
            print(f"\n[ERROR] Out of memory error: {e}")
            print("Try reducing batch size or image resolution")
            return
        except Exception as e:
            print(f"\n[ERROR] Training error: {e}")
            raise

        t1 = time.time()
        print(f"\n[DONE] Training finished in {(t1 - t0) / 60:.1f} min")

        # Always save a final copy (even if best checkpoint already saved)
        final_path = output_dir / "final_model.keras"
        print(f"[SAVE] Saving final model to: {final_path}")
        try:
            model.save(final_path)
        except Exception as e:
            print(f"[WARNING] Could not save final model: {e}")

        # Save history JSON for later analytics
        hist_path = analytics_dir / f"history_{mode}.json"
        try:
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(history.history, f, indent=2)
            print(f"[SAVE] Training history saved at: {hist_path}")
        except Exception as e:
            print(f"[WARNING] Could not save training history: {e}")

        print("\n[SUCCESS] Done. Best model is under outputs/best_model/best_model.keras\n")

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("Make sure you have run the preprocessing step first.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Nicer error if user runs from project root accidentally
    try:
        main()
    except NotImplementedError as e:
        print("\n[ERROR] NotImplementedError:", e)
        print("   Please implement get_datasets() in _helpers.py for your data.\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user")
        sys.exit(0)