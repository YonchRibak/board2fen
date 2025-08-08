# cnn_corner_detector_fixed.py - Memory-efficient CNN-based chess board corner detection

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class CornerDetectionConfig:
    """Configuration for CNN corner detection"""

    # Dataset paths
    dataset_path: str = "c:/datasets/ChessRender360"
    rgb_subdir: str = "rgb"
    annotations_subdir: str = "annotations"

    # Model parameters
    input_size: Tuple[int, int] = (224, 224)
    output_size: int = 8  # 4 corners × 2 coordinates

    # Training parameters
    batch_size: int = 16  # Reduced from 32 to save memory
    epochs: int = 50
    learning_rate: float = 1e-3
    validation_split: float = 0.2

    # Data augmentation
    enable_augmentation: bool = True
    rotation_range: float = 5.0  # Small rotations
    zoom_range: float = 0.1
    brightness_range: Tuple[float, float] = (0.8, 1.2)

    # Output paths
    output_dir: str = "outputs/corner_detector"
    model_name: str = "chess_corner_detector.keras"


class ChessCornerDataset:
    """Memory-efficient dataset class for chess board corner detection"""

    def __init__(self, config: CornerDetectionConfig):
        self.config = config
        self.dataset_path = Path(config.dataset_path)
        self.file_pairs = []  # Store file paths instead of loaded images

    def load_file_list(self, max_samples: int = None) -> List[Tuple[Path, Path]]:
        """Load list of valid image-annotation file pairs without loading images into memory"""

        print("📥 Loading ChessRender360 corner detection file list...")

        rgb_dir = self.dataset_path / self.config.rgb_subdir
        annotations_dir = self.dataset_path / self.config.annotations_subdir

        print(f"🔍 Looking for data in:")
        print(f"  RGB dir: {rgb_dir}")
        print(f"  Annotations dir: {annotations_dir}")
        print(f"  RGB dir exists: {rgb_dir.exists()}")
        print(f"  Annotations dir exists: {annotations_dir.exists()}")

        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
        if not annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

        # Get all RGB files
        rgb_files = []
        for ext in ["*.jpeg", "*.jpg", "*.png"]:
            found_files = list(rgb_dir.glob(ext))
            rgb_files.extend(found_files)
            print(f"  Found {len(found_files)} files with extension {ext}")

        rgb_files = sorted(rgb_files)
        print(f"  Total RGB files found: {len(rgb_files)}")

        if max_samples:
            rgb_files = rgb_files[:max_samples]

        print(f"Processing {len(rgb_files)} files...")

        valid_pairs = []

        for i, rgb_file in enumerate(rgb_files):
            # Get corresponding annotation file
            sample_id = rgb_file.stem.split('_')[1]  # Extract number from rgb_X.jpeg
            annotation_file = annotations_dir / f"annotation_{sample_id}.json"

            if not annotation_file.exists():
                if i < 5:  # Only print first 5 missing files to avoid spam
                    print(f"  ⚠️ Missing annotation: {annotation_file}")
                continue

            # Verify annotation file is valid JSON
            try:
                with open(annotation_file, 'r') as f:
                    annotation = json.load(f)

                # Check if required fields exist
                if 'board_corners' not in annotation:
                    continue

                valid_pairs.append((rgb_file, annotation_file))

                if len(valid_pairs) % 100 == 0:
                    print(f"Found {len(valid_pairs)} valid pairs...")

            except Exception as e:
                if i < 5:
                    print(f"Error reading annotation {annotation_file}: {e}")
                continue

        print(f"✅ Found {len(valid_pairs)} valid image-annotation pairs")
        self.file_pairs = valid_pairs
        return valid_pairs

    def load_and_preprocess_sample(self, rgb_file: Path, annotation_file: Path):
        """Load and preprocess a single sample"""

        # Load image and resize immediately to save memory
        image = cv2.imread(str(rgb_file))
        if image is None:
            raise ValueError(f"Could not load image: {rgb_file}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]

        # Resize to target size immediately
        image = cv2.resize(image, self.config.input_size)
        image = image.astype(np.float32) / 255.0

        # Load annotation
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)

        # Extract board corners
        board_corners = annotation['board_corners']

        # Convert to array format: [x1, y1, x2, y2, x3, y3, x4, y4]
        corner_array = [
            board_corners['white_left'][0], board_corners['white_left'][1],
            board_corners['white_right'][0], board_corners['white_right'][1],
            board_corners['black_right'][0], board_corners['black_right'][1],
            board_corners['black_left'][0], board_corners['black_left'][1]
        ]

        # Normalize coordinates to [0, 1] range based on original image size
        normalized_corners = [
            corner_array[0] / original_width, corner_array[1] / original_height,  # white_left
            corner_array[2] / original_width, corner_array[3] / original_height,  # white_right
            corner_array[4] / original_width, corner_array[5] / original_height,  # black_right
            corner_array[6] / original_width, corner_array[7] / original_height  # black_left
        ]

        return image, np.array(normalized_corners, dtype=np.float32)

    def create_tf_dataset(self, file_pairs: List[Tuple[Path, Path]], is_training: bool = True) -> tf.data.Dataset:
        """Create memory-efficient TensorFlow dataset using generator"""

        def data_generator():
            """Generator that yields one sample at a time and cycles indefinitely"""
            while True:  # Infinite loop to repeat data
                indices = list(range(len(file_pairs)))
                if is_training:
                    np.random.shuffle(indices)  # Shuffle on each epoch

                for idx in indices:
                    rgb_file, annotation_file = file_pairs[idx]
                    try:
                        image, corners = self.load_and_preprocess_sample(rgb_file, annotation_file)
                        yield image, corners
                    except Exception as e:
                        print(f"Error loading {rgb_file}: {e}")
                        continue

        # Define output signature
        output_signature = (
            tf.TensorSpec(shape=self.config.input_size + (3,), dtype=tf.float32),
            tf.TensorSpec(shape=(8,), dtype=tf.float32)
        )

        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=output_signature
        )

        # Add data augmentation for training
        if is_training and self.config.enable_augmentation:
            dataset = dataset.map(self._augment_data, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _augment_data(self, image, corners):
        """Apply data augmentation"""

        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)

        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Random saturation
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)

        # Ensure image values stay in [0, 1] range
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, corners


class CNNCornerDetector:
    """CNN-based chess board corner detector"""

    def __init__(self, config: CornerDetectionConfig):
        self.config = config
        self.model = None
        self.history = None

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def build_model(self) -> tf.keras.Model:
        """Build CNN model for corner detection"""

        # Use ResNet50 as backbone (reliable, well-tested, no weight loading issues)
        backbone = tf.keras.applications.ResNet50(
            input_shape=self.config.input_size + (3,),
            include_top=False,
            weights='imagenet'
        )

        # Initially freeze backbone
        backbone.trainable = False

        model = tf.keras.Sequential([
            backbone,

            # Global feature extraction
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Corner regression head
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),

            # Output: 8 coordinates (4 corners × 2 coords)
            layers.Dense(self.config.output_size, activation='sigmoid')  # Sigmoid for normalized coords
        ])

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',  # Mean squared error for regression
            metrics=['mae']  # Mean absolute error
        )

        self.model = model
        return model

    def train(self,
              train_dataset: tf.data.Dataset,
              val_dataset: tf.data.Dataset,
              steps_per_epoch: int,
              validation_steps: int) -> tf.keras.callbacks.History:
        """Train the corner detection model"""

        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("🚀 Starting corner detection training...")

        # Callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                filepath=Path(self.config.output_dir) / self.config.model_name,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1
        )

        return self.history

    def fine_tune(self,
                  train_dataset: tf.data.Dataset,
                  val_dataset: tf.data.Dataset,
                  steps_per_epoch: int,
                  validation_steps: int,
                  fine_tune_epochs: int = 20):
        """Fine-tune the model with unfrozen backbone"""

        print("🔓 Fine-tuning with unfrozen backbone...")

        # Unfreeze backbone
        self.model.layers[0].trainable = True

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate * 0.1),
            loss='mse',
            metrics=['mae']
        )

        # Fine-tune
        fine_tune_history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=fine_tune_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[
                callbacks.ModelCheckpoint(
                    filepath=Path(self.config.output_dir) / f"fine_tuned_{self.config.model_name}",
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                )
            ],
            verbose=1
        )

        return fine_tune_history

    def predict_corners(self, image: np.ndarray) -> np.ndarray:
        """Predict corners for a single image"""

        if self.model is None:
            raise ValueError("Model not trained. Load a trained model first.")

        # Preprocess image
        original_shape = image.shape[:2]
        processed_image = cv2.resize(image, self.config.input_size)
        processed_image = processed_image.astype(np.float32) / 255.0
        processed_image = np.expand_dims(processed_image, axis=0)

        # Predict normalized coordinates
        normalized_corners = self.model.predict(processed_image, verbose=0)[0]

        # Convert back to original image coordinates
        h, w = original_shape
        corners = [
            normalized_corners[0] * w, normalized_corners[1] * h,  # white_left
            normalized_corners[2] * w, normalized_corners[3] * h,  # white_right
            normalized_corners[4] * w, normalized_corners[5] * h,  # black_right
            normalized_corners[6] * w, normalized_corners[7] * h  # black_left
        ]

        # Reshape to corner format
        corner_points = np.array([
            [corners[0], corners[1]],  # white_left
            [corners[2], corners[3]],  # white_right
            [corners[4], corners[5]],  # black_right
            [corners[6], corners[7]]  # black_left
        ])

        return corner_points

    def visualize_prediction(self, image: np.ndarray, predicted_corners: np.ndarray):
        """Visualize corner predictions"""

        img_vis = image.copy()
        corners = predicted_corners.astype(int)

        # Draw corners
        cv2.polylines(img_vis, [corners], True, (0, 255, 0), 3)

        # Draw corner points
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        labels = ['WL', 'WR', 'BR', 'BL']

        for i, (corner, color, label) in enumerate(zip(corners, colors, labels)):
            cv2.circle(img_vis, tuple(corner), 8, color, -1)
            cv2.putText(img_vis, label, (corner[0] + 10, corner[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(img_vis)
        plt.title("CNN Corner Detection")
        plt.axis('off')
        plt.show()

    def evaluate_model(self, test_dataset: tf.data.Dataset, steps: int) -> Dict:
        """Evaluate trained model"""

        if self.model is None:
            raise ValueError("Model not trained.")

        # Evaluate
        results = self.model.evaluate(test_dataset, steps=steps, verbose=1)

        metrics = {
            'test_loss': results[0],
            'test_mae': results[1]
        }

        print(f"📊 Model Evaluation:")
        print(f"Test Loss (MSE): {metrics['test_loss']:.6f}")
        print(f"Test MAE: {metrics['test_mae']:.6f}")

        return metrics

    def plot_training_history(self):
        """Plot training history"""

        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


def main():
    """Main training and evaluation pipeline"""

    # Configuration
    config = CornerDetectionConfig(
        dataset_path="c:/datasets/ChessRender360",
        input_size=(224, 224),
        batch_size=16,  # Reduced batch size to save memory
        epochs=50,
        learning_rate=1e-3
    )

    print("🎯 Memory-Efficient CNN Chess Board Corner Detection Training")
    print("=" * 60)

    # Load file list (not actual images)
    dataset = ChessCornerDataset(config)
    file_pairs = dataset.load_file_list(max_samples=5000)  # Use subset for faster training

    # Check if we have data
    if len(file_pairs) == 0:
        print("❌ No training data found. Please check dataset path and file structure.")
        return

    print(f"📊 Dataset loaded: {len(file_pairs)} file pairs")

    # Split file pairs
    train_pairs, val_pairs = train_test_split(
        file_pairs, test_size=config.validation_split, random_state=42
    )

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")

    # Calculate steps per epoch
    steps_per_epoch = len(train_pairs) // config.batch_size
    validation_steps = len(val_pairs) // config.batch_size

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Create TF datasets (memory-efficient)
    train_dataset = dataset.create_tf_dataset(train_pairs, is_training=True)
    val_dataset = dataset.create_tf_dataset(val_pairs, is_training=False)

    # Build and train model
    detector = CNNCornerDetector(config)
    detector.build_model()

    print(f"📊 Model Architecture:")
    detector.model.summary()

    # Train
    detector.train(train_dataset, val_dataset, steps_per_epoch, validation_steps)

    # Fine-tune
    detector.fine_tune(train_dataset, val_dataset, steps_per_epoch, validation_steps)

    # Evaluate
    metrics = detector.evaluate_model(val_dataset, validation_steps)

    # Plot results
    detector.plot_training_history()

    # Test on sample image
    test_rgb_file, test_annotation_file = train_pairs[0]
    test_image = cv2.imread(str(test_rgb_file))
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    predicted_corners = detector.predict_corners(test_image)
    detector.visualize_prediction(test_image, predicted_corners)

    print(f"✅ Training complete! Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()