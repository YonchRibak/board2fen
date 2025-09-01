# board2fen/end2end_board_classifier/train.py
import os
import sys
import time
import json
import gc
import logging
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from _helpers import (
    get_mode_config,
    get_light_model_config,
    seed_everything,
    build_resnext101_32x8d_head,
    build_light_chess_classifier,
    make_dirs,
    make_callbacks,
    get_datasets,
    pretty_cfg,
    setup_logging,
    check_preprocessed_data_exists,
    check_gcs_upload_requirements,
    prompt_upload_models,
    BoardLevelEvalCallback,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "end2end_board_classifier_new_data" / "outputs" / "best_model"
DEFAULT_ANALYTICS_DIR = PROJECT_ROOT / "end2end_board_classifier_new_data" / "analytics"

# Global logger will be initialized in main()
logger = None


def configure_gpu_memory():
    """Configure GPU memory growth to prevent OOM errors."""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                message = f"[GPU] Configured memory growth for {len(gpus)} GPU(s)"
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
        # Don't raise - training can continue on CPU


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
    print("End-to-End Chess Recognition - GCS Training")
    print("=" * 70)

    # Setup logging first
    try:
        log_dir = Path(__file__).resolve().parent / "logs"
        logger = setup_logging(log_dir)
        # Also set the global logger in _helpers
        import _helpers
        _helpers.logger = logger

        logger.info("=" * 50)
        logger.info("Starting chess recognition training")
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

        # Paths - update output dir based on model type
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

        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Analytics directory: {analytics_dir}")

        # Prep environment & folders
        seed_everything(cfg["seed"])
        make_dirs([output_dir, analytics_dir])

        print(f"\n[CONFIG] Configuration:")
        print(pretty_cfg(cfg))
        logger.info(f"Training configuration: {json.dumps(cfg, indent=2)}")

        # Datasets
        print("\n[DATA] Loading datasets from GCS...")
        logger.info("Starting dataset loading from GCS")

        try:
            train_ds, val_ds, steps_per_epoch, val_steps = get_datasets(cfg)
            logger.info(
                f"Successfully loaded datasets - {steps_per_epoch} training steps, {val_steps} validation steps")
        except Exception as e:
            error_msg = f"Failed to load datasets: {e}"
            print(f"[ERROR] {error_msg}")
            logger.error(error_msg)
            raise

        # Force garbage collection after dataset loading
        gc.collect()

        # Build model based on selection
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

        # Compile with mixed precision for better memory efficiency (mainly for large model)
        print("\n[COMPILE] Compiling model...")
        logger.info("Compiling model")

        try:
            # Enable mixed precision if available and using large model
            if model_type == "resnext":
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    message = "Mixed precision enabled"
                    print(f"[INFO] {message}")
                    logger.info(message)
                except Exception as mp_error:
                    message = f"Mixed precision not available: {mp_error}, using float32"
                    print(f"[WARNING] {message}")
                    logger.warning(message)
            else:
                message = "Mixed precision disabled for light model"
                print(f"[INFO] {message}")
                logger.info(message)

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
            print(f"Model type: {model_type}")
            print(f"Input shape: {cfg['input_shape']}")

            logger.info(
                f"Model compiled successfully - {model.count_params():,} parameters, ~{model.count_params() * 4 / 1024 / 1024:.1f} MB")

        except Exception as e:
            error_msg = f"Failed to compile model: {e}"
            print(f"[ERROR] {error_msg}")
            logger.error(error_msg)
            raise

        # Callbacks: checkpoint, early stop, LR scheduler, CSV log, tensorboard, board-level eval
        print("\n[CALLBACKS] Preparing callbacks...")
        logger.info("Preparing training callbacks")

        try:
            callbacks = make_callbacks(
                cfg=cfg,
                output_dir=output_dir,
                analytics_dir=analytics_dir,
                val_ds=val_ds
            )
            logger.info("Successfully created all training callbacks")
        except Exception as e:
            error_msg = f"Failed to create callbacks: {e}"
            print(f"[ERROR] {error_msg}")
            logger.error(error_msg)
            raise

        # Test the first batch to catch errors early
        print("\n[TEST] Testing first batch...")
        logger.info("Testing first batch")

        try:
            test_count = 0
            for x_batch, y_batch in train_ds.take(1):
                print(f"Input shape: {x_batch.shape}")
                print(f"Label shape: {y_batch.shape}")
                logger.info(f"First batch - Input: {x_batch.shape}, Labels: {y_batch.shape}")

                _ = model(x_batch, training=False)  # Test forward pass
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

        print(f"\n[TRAIN] Starting training ({model_type} model, {mode} mode)...")
        logger.info(f"Starting training - {model_type} model, {mode} mode, {cfg['epochs']} epochs")

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

        # Always save a final copy (even if best checkpoint already saved)
        # Create descriptive filename with current date
        current_date = datetime.now().strftime("%Y%m%d")
        model_type_name = "large" if model_type == "resnext" else model_type
        final_filename = f"final_{model_type_name}_{mode}_{current_date}.keras"
        final_path = output_dir / final_filename

        print(f"[SAVE] Saving final model to: {final_path}")
        logger.info(f"Saving final model to: {final_path}")

        try:
            model.save(final_path)
            print(f"[SUCCESS] Final model saved as: {final_filename}")
            logger.info(f"Final model saved successfully as: {final_filename}")
        except Exception as e:
            error_msg = f"Could not save final model: {e}"
            print(f"[WARNING] {error_msg}")
            logger.warning(error_msg)

        # Save history JSON for later analytics
        hist_filename = f"history_{model_type_name}_{mode}_{current_date}.json"
        hist_path = analytics_dir / hist_filename
        try:
            with open(hist_path, "w", encoding="utf-8") as f:
                # Add model info to history
                history_dict = history.history.copy()
                history_dict["model_info"] = {
                    "model_type": model_type_name,
                    "mode": mode,
                    "total_params": int(model.count_params()),
                    "input_shape": cfg["input_shape"],
                    "batch_size": cfg["batch_size"],
                    "epochs": cfg["epochs"],
                    "training_time_minutes": training_time,
                    "training_date": current_date,
                    "final_model_filename": final_filename,
                }
                json.dump(history_dict, f, indent=2)

            print(f"[SAVE] Training history saved as: {hist_filename}")
            logger.info(f"Training history saved as: {hist_filename}")

        except Exception as e:
            error_msg = f"Could not save training history: {e}"
            print(f"[WARNING] {error_msg}")
            logger.warning(error_msg)

        # Optional: Upload models to GCS
        print(f"\n{'=' * 70}")
        print(f"[UPLOAD] Model Upload to Google Cloud Storage")
        print(f"{'=' * 70}")

        try:
            # Check if upload is available
            if check_gcs_upload_requirements():
                upload_success = prompt_upload_models(
                    output_dir,
                    analytics_dir,
                    final_filename,
                    hist_filename
                )
                if upload_success:
                    logger.info("Model upload completed successfully")
                else:
                    logger.info("Model upload skipped or failed")
            else:
                print("[INFO] GCS upload not available")
                print("[INSTALL] To enable uploads, run: pip install google-cloud-storage")
                print("[AUTH] Then run: gcloud auth application-default login")
                logger.info("GCS upload requirements not met")

        except KeyboardInterrupt:
            print(f"\n[SKIP] Upload cancelled by user")
            logger.info("Upload cancelled by user")
        except Exception as e:
            error_msg = f"Upload error: {e}"
            print(f"[WARNING] {error_msg}")
            logger.warning(error_msg)

        print(f"\n[SUCCESS] Training completed successfully!")
        print(f"[INFO] Model type: {model_type_name} | Training mode: {mode}")
        print(f"[INFO] Best model: {output_dir}/best_model.keras")
        print(f"[INFO] Final model: {final_filename}")
        print(f"[INFO] Parameters: {model.count_params():,}")
        print(f"[INFO] Training time: {training_time:.1f} minutes\n")

        logger.info("=" * 50)
        logger.info("Training completed successfully")
        logger.info(f"Model: {model_type_name}, Mode: {mode}, Parameters: {model.count_params():,}")
        logger.info(f"Final model filename: {final_filename}")
        logger.info(f"Training time: {training_time:.1f} minutes")
        logger.info("=" * 50)

        return True

    except FileNotFoundError as e:
        error_msg = f"File not found: {e}"
        print(f"\n[ERROR] {error_msg}")
        if logger:
            logger.error(error_msg)
        print("Make sure you have run the preprocessing step first.")
        return False

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"\n[ERROR] {error_msg}")
        if logger:
            logger.error(error_msg)

        import traceback
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")
        if logger:
            logger.error(f"Traceback: {traceback_str}")
        return False


if __name__ == "__main__":
    # Nicer error handling
    try:
        success = main()
        sys.exit(0 if success else 1)

    except NotImplementedError as e:
        error_msg = f"NotImplementedError: {e}"
        print(f"\n[ERROR] {error_msg}")
        print("   Please implement get_datasets() in _helpers.py for your data.\n")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user")
        if logger:
            logger.info("Training interrupted by user")
        sys.exit(0)

    except Exception as e:
        error_msg = f"Fatal error: {e}"
        print(f"\n[FATAL] {error_msg}")
        sys.exit(1)