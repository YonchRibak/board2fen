# hybrid_chess_cnn_classifier.py
# End-to-end CNN for chess position prediction using real ChessReD + synthetic data

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import tensorflow as tf
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import Dict, Tuple, List, Optional
import time
from tqdm import tqdm
import chess
import re
import random
from collections import Counter

# CPU optimizations
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)


class FENProcessor:
    """Utility class for processing FEN strings to grid labels"""

    def __init__(self):
        # Mapping from FEN piece notation to class IDs
        self.fen_to_class = {
            'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k': 6,  # black pieces
            'P': 7, 'R': 8, 'N': 9, 'B': 10, 'Q': 11, 'K': 12  # white pieces
        }

        # Class 0 is empty square
        self.class_to_name = {
            0: 'empty', 1: 'black_pawn', 2: 'black_rook', 3: 'black_knight',
            4: 'black_bishop', 5: 'black_queen', 6: 'black_king',
            7: 'white_pawn', 8: 'white_rook', 9: 'white_knight',
            10: 'white_bishop', 11: 'white_queen', 12: 'white_king'
        }

    def fen_to_grid(self, fen_string: str) -> np.ndarray:
        """Convert FEN string to 8x8 grid of class IDs"""
        try:
            # Extract just the piece placement part (before first space)
            board_part = fen_string.split()[0]

            # Initialize 8x8 grid
            grid = np.zeros((8, 8), dtype=np.int32)

            ranks = board_part.split('/')
            if len(ranks) != 8:
                raise ValueError(f"Invalid FEN: expected 8 ranks, got {len(ranks)}")

            for rank_idx, rank in enumerate(ranks):
                file_idx = 0
                for char in rank:
                    if char.isdigit():
                        # Empty squares
                        file_idx += int(char)
                    elif char in self.fen_to_class:
                        # Piece
                        grid[rank_idx, file_idx] = self.fen_to_class[char]
                        file_idx += 1
                    else:
                        raise ValueError(f"Invalid FEN character: {char}")

            return grid

        except Exception as e:
            print(f"Error processing FEN '{fen_string}': {e}")
            return np.zeros((8, 8), dtype=np.int32)

    def extract_fen_from_filename(self, filename: str) -> Optional[str]:
        """Extract FEN string from filename, handling various formats"""
        filename = Path(filename).stem  # Remove extension

        # Try different patterns that might be used in filenames
        patterns = [
            r'^(.+)$',  # Entire filename is FEN
            r'fen_(.+)',  # fen_<FEN>
            r'position_(.+)',  # position_<FEN>
        ]

        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                fen_candidate = match.group(1)
                # Replace URL encoding or special characters that might be in filename
                fen_candidate = fen_candidate.replace('_', ' ')
                fen_candidate = fen_candidate.replace('%20', ' ')

                # Validate FEN format roughly
                if self.is_valid_fen_format(fen_candidate):
                    return fen_candidate

        return None

    def is_valid_fen_format(self, fen_string: str) -> bool:
        """Quick validation of FEN format"""
        try:
            parts = fen_string.split()
            if len(parts) < 1:
                return False

            board_part = parts[0]
            ranks = board_part.split('/')
            return len(ranks) == 8

        except:
            return False


class HybridChessDataset:
    """Dataset loader that combines real ChessReD data with synthetic data"""

    def __init__(self,
                 real_data_dir: str,
                 synthetic_data_dir: Optional[str] = None,
                 img_size: Tuple[int, int] = (224, 224),
                 real_synthetic_ratio: float = 0.8):
        """
        Initialize hybrid dataset

        Args:
            real_data_dir: Path to ChessReD dataset
            synthetic_data_dir: Path to synthetic dataset (optional)
            img_size: Target image size
            real_synthetic_ratio: Fraction of data that should be real (0.8 = 80% real, 20% synthetic)
        """
        self.real_data_dir = Path(real_data_dir)
        self.synthetic_data_dir = Path(synthetic_data_dir) if synthetic_data_dir else None
        self.img_size = img_size
        self.real_synthetic_ratio = real_synthetic_ratio
        self.fen_processor = FENProcessor()

        print(f"📂 Loading hybrid dataset:")
        print(f"   Real data: {real_data_dir}")
        print(f"   Synthetic data: {synthetic_data_dir}")
        print(f"   Target ratio: {real_synthetic_ratio:.1%} real, {1 - real_synthetic_ratio:.1%} synthetic")

        # Load datasets
        self.real_samples = self._load_real_data()
        self.synthetic_samples = self._load_synthetic_data() if synthetic_data_dir else []

        print(f"✅ Dataset loaded:")
        print(f"   Real samples: {len(self.real_samples):,}")
        print(f"   Synthetic samples: {len(self.synthetic_samples):,}")

        # Create combined dataset with proper ratio
        self.combined_samples = self._create_combined_dataset()
        print(f"   Combined total: {len(self.combined_samples):,}")

    def _load_real_data(self) -> List[Dict]:
        """Load real ChessReD data"""
        print("📸 Loading real ChessReD data...")

        samples = []

        # Look for image files in the real data directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(list(self.real_data_dir.rglob(f'*{ext}')))
            image_files.extend(list(self.real_data_dir.rglob(f'*{ext.upper()}')))

        print(f"   Found {len(image_files)} potential image files")

        valid_samples = 0
        invalid_samples = 0

        for image_path in tqdm(image_files, desc="Processing real images"):
            # Extract FEN from filename
            fen_string = self.fen_processor.extract_fen_from_filename(image_path.name)

            if fen_string:
                # Convert FEN to grid
                grid = self.fen_processor.fen_to_grid(fen_string)

                if grid is not None:
                    samples.append({
                        'image_path': str(image_path),
                        'label': grid,
                        'fen': fen_string,
                        'source': 'real'
                    })
                    valid_samples += 1
                else:
                    invalid_samples += 1
            else:
                invalid_samples += 1

        print(f"   Valid samples: {valid_samples:,}")
        print(f"   Invalid samples: {invalid_samples:,}")

        return samples

    def _load_synthetic_data(self) -> List[Dict]:
        """Load synthetic data"""
        if not self.synthetic_data_dir or not self.synthetic_data_dir.exists():
            return []

        print("🎨 Loading synthetic data...")

        try:
            # Load metadata
            with open(self.synthetic_data_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)

            # Load image paths
            with open(self.synthetic_data_dir / "image_paths.txt", 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]

            # Load labels
            labels = np.load(self.synthetic_data_dir / "labels_complete.npy")

            samples = []
            for i, (image_path, label) in enumerate(zip(image_paths, labels)):
                samples.append({
                    'image_path': image_path,
                    'label': label,
                    'fen': None,  # FEN not available for synthetic data
                    'source': 'synthetic'
                })

            print(f"   Loaded {len(samples):,} synthetic samples")
            return samples

        except Exception as e:
            print(f"   ⚠️ Error loading synthetic data: {e}")
            return []

    def _create_combined_dataset(self) -> List[Dict]:
        """Create combined dataset with proper real/synthetic ratio"""

        if not self.real_samples:
            print("⚠️ No real samples available, using synthetic only")
            return self.synthetic_samples

        if not self.synthetic_samples:
            print("⚠️ No synthetic samples available, using real only")
            return self.real_samples

        # Calculate target counts
        total_real = len(self.real_samples)

        # Calculate how many synthetic samples to include
        if self.real_synthetic_ratio >= 1.0:
            target_synthetic = 0
        else:
            target_synthetic = int(total_real * (1 - self.real_synthetic_ratio) / self.real_synthetic_ratio)

        target_synthetic = min(target_synthetic, len(self.synthetic_samples))

        print(f"📊 Creating combined dataset:")
        print(f"   Real samples: {total_real:,}")
        print(f"   Synthetic samples: {target_synthetic:,}")

        # Sample synthetic data
        if target_synthetic > 0:
            synthetic_subset = random.sample(self.synthetic_samples, target_synthetic)
        else:
            synthetic_subset = []

        # Combine datasets
        combined = self.real_samples + synthetic_subset
        random.shuffle(combined)

        return combined

    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess single image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0

        return image

    def create_generators(self, train_indices: List[int], val_indices: List[int],
                          batch_size: int = 4, augment_training: bool = True):
        """Create data generators for training and validation"""

        def train_generator():
            """Training data generator with augmentation"""
            while True:
                indices = train_indices.copy()
                random.shuffle(indices)

                batch_images = []
                batch_labels = []

                for idx in indices:
                    try:
                        sample = self.combined_samples[idx]

                        # Load image
                        image = self.load_and_preprocess_image(sample['image_path'])

                        # Apply augmentation for training
                        if augment_training:
                            image = self.augment_image(image, sample['source'])

                        # Get label
                        label = sample['label']

                        batch_images.append(image)
                        batch_labels.append(label)

                        if len(batch_images) == batch_size:
                            yield np.array(batch_images), np.array(batch_labels)
                            batch_images = []
                            batch_labels = []

                    except Exception as e:
                        print(f"Error loading sample {idx}: {e}")
                        continue

        def val_generator():
            """Validation data generator (no augmentation)"""
            while True:
                indices = val_indices.copy()

                batch_images = []
                batch_labels = []

                for idx in indices:
                    try:
                        sample = self.combined_samples[idx]

                        # Load image (no augmentation)
                        image = self.load_and_preprocess_image(sample['image_path'])
                        label = sample['label']

                        batch_images.append(image)
                        batch_labels.append(label)

                        if len(batch_images) == batch_size:
                            yield np.array(batch_images), np.array(batch_labels)
                            batch_images = []
                            batch_labels = []

                    except Exception as e:
                        print(f"Error loading validation sample {idx}: {e}")
                        continue

        return train_generator(), val_generator()

    def augment_image(self, image: np.ndarray, source: str) -> np.ndarray:
        """Apply source-appropriate augmentations"""

        # Real images need more conservative augmentation
        if source == 'real':
            # Real images already have natural variation

            # Mild brightness/contrast (real images have natural lighting variation)
            if np.random.random() > 0.7:
                brightness = np.random.uniform(0.9, 1.1)
                image = np.clip(image * brightness, 0, 1)

            if np.random.random() > 0.7:
                contrast = np.random.uniform(0.9, 1.1)
                image = np.clip((image - 0.5) * contrast + 0.5, 0, 1)

            # Very small rotations (real images might be slightly tilted)
            if np.random.random() > 0.8:
                angle = np.random.uniform(-2, 2)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, matrix, (w, h))

        else:  # Synthetic images
            # Synthetic images can handle more aggressive augmentation

            # Random horizontal flip (valid for chess)
            if np.random.random() > 0.5:
                image = np.fliplr(image)

            # Rotation
            if np.random.random() > 0.5:
                angle = np.random.uniform(-5, 5)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, matrix, (w, h))

            # Brightness and contrast
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.7, 1.3)
                image = np.clip(image * brightness, 0, 1)

            if np.random.random() > 0.5:
                contrast = np.random.uniform(0.7, 1.3)
                image = np.clip((image - 0.5) * contrast + 0.5, 0, 1)

            # Gaussian noise
            if np.random.random() > 0.7:
                noise = np.random.normal(0, 0.02, image.shape)
                image = np.clip(image + noise, 0, 1)

        return image

    def analyze_dataset(self):
        """Analyze the combined dataset"""
        print("\n📊 DATASET ANALYSIS")
        print("=" * 50)

        # Source distribution
        source_counts = Counter(sample['source'] for sample in self.combined_samples)
        print(f"Source distribution:")
        for source, count in source_counts.items():
            percentage = count / len(self.combined_samples) * 100
            print(f"   {source}: {count:,} samples ({percentage:.1f}%)")

        # Piece distribution (from real data with FEN)
        print(f"\nPiece distribution analysis:")
        piece_counts = Counter()

        for sample in self.combined_samples[:1000]:  # Sample for analysis
            label = sample['label']
            unique, counts = np.unique(label, return_counts=True)
            for piece_id, count in zip(unique, counts):
                piece_counts[piece_id] += count

        total_squares = sum(piece_counts.values())
        print(f"Analyzed {total_squares:,} squares from sample:")

        for piece_id in sorted(piece_counts.keys()):
            count = piece_counts[piece_id]
            percentage = count / total_squares * 100
            piece_name = self.fen_processor.class_to_name[piece_id]
            print(f"   {piece_name}: {count:,} ({percentage:.1f}%)")


class ChessBoardCNN:
    """Enhanced CNN for hybrid real+synthetic chess data"""

    def __init__(self, num_classes: int = 13, img_size: Tuple[int, int] = (224, 224)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None

    def build_model(self, backbone: str = 'efficientnet') -> tf.keras.Model:
        """Build CNN model optimized for real chess images"""

        print(f"🏗️ Building {backbone} model for real chess data...")

        # Input
        inputs = tf.keras.Input(shape=self.img_size + (3,), name='chess_image')

        # Backbone selection - optimized for real image variations
        if backbone == 'efficientnet':
            backbone_model = tf.keras.applications.EfficientNetB1(  # B1 for better real-world performance
                input_shape=self.img_size + (3,),
                include_top=False,
                weights='imagenet'
            )
            x = backbone_model(inputs)

        elif backbone == 'resnet':
            backbone_model = tf.keras.applications.ResNet50(
                input_shape=self.img_size + (3,),
                include_top=False,
                weights='imagenet'
            )
            x = backbone_model(inputs)

        else:  # mobilenet - fastest option
            backbone_model = tf.keras.applications.MobileNetV2(
                input_shape=self.img_size + (3,),
                include_top=False,
                weights='imagenet'
            )
            x = backbone_model(inputs)

        # Start with frozen backbone
        backbone_model.trainable = False

        # Enhanced spatial processing for real chess images
        # Add spatial attention to focus on chess squares
        spatial_attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
        x = layers.Multiply()([x, spatial_attention])

        # Global features
        global_features = layers.GlobalAveragePooling2D()(x)
        global_features = layers.BatchNormalization()(global_features)
        global_features = layers.Dropout(0.3)(global_features)

        # Dense layers for chess understanding
        features = layers.Dense(1024, activation='relu')(global_features)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(0.4)(features)

        features = layers.Dense(512, activation='relu')(features)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(0.3)(features)

        # Spatial grid reconstruction
        grid_features = layers.Dense(8 * 8 * 128, activation='relu')(features)
        grid_features = layers.Reshape((8, 8, 128))(grid_features)

        # Spatial convolutions with residual connections
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(grid_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Residual block
        residual = x
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])

        # Final layers
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Output prediction
        predictions = layers.Conv2D(self.num_classes, (1, 1),
                                    activation='softmax', name='position_prediction')(x)

        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        print(f"📊 Model built with {model.count_params():,} parameters")

        self.model = model
        return model

    def compile_model(self, learning_rate: float = 1e-3):
        """Compile model with class weighting for chess positions"""

        # Chess-specific loss weighting (pieces are rarer than empty squares)
        class_weights = {
            0: 1.0,  # empty (most common)
            1: 2.0, 2: 3.0, 3: 4.0, 4: 3.0, 5: 5.0, 6: 5.0,  # black pieces
            7: 2.0, 8: 3.0, 9: 4.0, 10: 3.0, 11: 5.0, 12: 5.0  # white pieces
        }

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Model compiled with learning rate: {learning_rate}")

    def progressive_train(self, train_gen, val_gen, steps_per_epoch: int,
                          validation_steps: int, config: dict):
        """Progressive training optimized for real chess data"""

        print("\n🚀 PROGRESSIVE TRAINING FOR REAL CHESS DATA")
        print("=" * 50)

        total_epochs = config['epochs']
        phase1_epochs = max(8, total_epochs // 2)  # More frozen epochs for real data
        phase2_epochs = total_epochs - phase1_epochs

        # Phase 1: Frozen backbone - learn chess-specific features
        print(f"\n📚 PHASE 1: Learning chess features ({phase1_epochs} epochs)")

        # Find and freeze backbone
        backbone_layer = None
        for layer in self.model.layers:
            if hasattr(layer, 'trainable') and layer.count_params() > 1000000:
                backbone_layer = layer
                break

        if backbone_layer is not None:
            backbone_layer.trainable = False
            print(f"✅ Frozen backbone: {backbone_layer.name}")

        self.compile_model(config['learning_rate'])

        callbacks_phase1 = self.create_callbacks(config, "phase1")

        try:
            history1 = self.model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_gen,
                validation_steps=validation_steps,
                epochs=phase1_epochs,
                callbacks=callbacks_phase1,
                verbose=1
            )

            phase1_acc = max(history1.history['val_accuracy'])
            print(f"\n📊 Phase 1 Results: Best validation accuracy = {phase1_acc:.4f}")

        except Exception as e:
            print(f"❌ Phase 1 training failed: {e}")
            return None

        # Phase 2: Fine-tuning for real image variations
        if phase2_epochs > 0:
            print(f"\n🔓 PHASE 2: Fine-tuning for real images ({phase2_epochs} epochs)")

            try:
                if backbone_layer is not None:
                    backbone_layer.trainable = True
                    print(f"✅ Unfrozen backbone: {backbone_layer.name}")

                # Lower learning rate for fine-tuning real data
                self.compile_model(config['learning_rate'] * 0.05)  # More conservative for real data

                callbacks_phase2 = self.create_callbacks(config, "phase2")

                history2 = self.model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    epochs=phase2_epochs,
                    initial_epoch=phase1_epochs,
                    callbacks=callbacks_phase2,
                    verbose=1
                )

                try:
                    self.history = self.combine_histories([history1, history2])
                except Exception as e:
                    print(f"⚠️ Could not combine histories: {e}")
                    self.history = history2

            except Exception as e:
                print(f"❌ Phase 2 training failed: {e}")
                self.history = history1
        else:
            self.history = history1

        return self.history

    def create_callbacks(self, config: dict, phase: str) -> List:
        """Create training callbacks optimized for real data"""

        output_dir = Path(config['output_dir'])

        callbacks_list = [
            callbacks.ModelCheckpoint(
                filepath=output_dir / f"best_model_{phase}.keras",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,  # More aggressive LR reduction for real data
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Early stopping for longer training
        if config['epochs'] > 20:
            callbacks_list.append(
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=max(6, config['epochs'] // 8),
                    restore_best_weights=True,
                    verbose=1
                )
            )

        return callbacks_list

    def combine_histories(self, histories: List) -> tf.keras.callbacks.History:
        """Combine multiple training histories"""
        combined = tf.keras.callbacks.History()
        combined.history = {}

        if not histories:
            return combined

        all_keys = set()
        for hist in histories:
            if hasattr(hist, 'history') and hist.history:
                all_keys.update(hist.history.keys())

        for key in all_keys:
            combined.history[key] = []

        for hist in histories:
            if hasattr(hist, 'history') and hist.history:
                for key in all_keys:
                    if key in hist.history:
                        combined.history[key].extend(hist.history[key])

        return combined


def create_training_config(mode: str, **kwargs) -> dict:
    """Create training configuration for real data"""

    if mode == 'quick':
        config = {
            'mode': 'quick',
            'epochs': 15,
            'batch_size': 6,
            'learning_rate': 1e-3,
            'backbone': 'mobilenet',
            'augmentation': True,
            'real_synthetic_ratio': 0.9,  # Favor real data
            'description': 'Quick training on real data (15 epochs, MobileNet)'
        }

    elif mode == 'thorough':
        config = {
            'mode': 'thorough',
            'epochs': 40,
            'batch_size': 4,
            'learning_rate': 8e-4,
            'backbone': 'efficientnet',
            'augmentation': True,
            'real_synthetic_ratio': 0.8,  # 80% real, 20% synthetic
            'description': 'Thorough training on real data (40 epochs, EfficientNet)'
        }

    elif mode == 'production':
        config = {
            'mode': 'production',
            'epochs': 60,
            'batch_size': 3,
            'learning_rate': 5e-4,
            'backbone': 'efficientnet',
            'augmentation': True,
            'real_synthetic_ratio': 0.85,  # Mostly real data
            'description': 'Production training on real data (60 epochs, best quality)'
        }

    elif mode == 'custom':
        config = {
            'mode': 'custom',
            'epochs': kwargs.get('epochs', 25),
            'batch_size': kwargs.get('batch_size', 4),
            'learning_rate': kwargs.get('learning_rate', 1e-3),
            'backbone': kwargs.get('backbone', 'efficientnet'),
            'augmentation': kwargs.get('augmentation', True),
            'real_synthetic_ratio': kwargs.get('real_synthetic_ratio', 0.8),
            'description': f"Custom training ({kwargs.get('epochs', 25)} epochs)"
        }

    else:
        raise ValueError(f"Unknown mode: {mode}")

    config['output_dir'] = f"outputs/hybrid_chess_cnn_{mode}"
    return config


def main():
    """Main training function for hybrid real+synthetic data"""

    parser = argparse.ArgumentParser(description='Hybrid Real+Synthetic Chess CNN Classifier')
    parser.add_argument('--mode', type=str, default='thorough',
                        choices=['quick', 'thorough', 'production', 'custom'],
                        help='Training mode (default: thorough)')
    parser.add_argument('--real_data', type=str,
                        default='/path/to/ChessReD',  # PLACEHOLDER - update when download complete
                        help='Path to ChessReD real dataset')
    parser.add_argument('--synthetic_data', type=str,
                        default='C:/datasets/ChessRender360_GridLabels',
                        help='Path to synthetic dataset (optional)')
    parser.add_argument('--real_ratio', type=float, default=0.8,
                        help='Fraction of data that should be real (0.8 = 80% real, 20% synthetic)')

    # Custom mode parameters
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--backbone', type=str, default='efficientnet',
                        choices=['efficientnet', 'resnet', 'mobilenet'])

    args = parser.parse_args()

    print("♟️  HYBRID REAL+SYNTHETIC CHESS CNN CLASSIFIER")
    print("=" * 60)
    print("🎯 Training on real ChessReD data + synthetic augmentation")

    # Create training configuration
    config = create_training_config(
        args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        backbone=args.backbone,
        real_synthetic_ratio=args.real_ratio
    )

    print(f"🎯 Training Mode: {config['mode'].upper()}")
    print(f"📝 {config['description']}")
    print(f"⚙️  Configuration:")
    for key, value in config.items():
        if key != 'description':
            print(f"   {key}: {value}")

    # Check datasets
    if not Path(args.real_data).exists():
        print(f"❌ Real dataset not found: {args.real_data}")
        print("   Please update --real_data path when ChessReD download is complete")
        return

    # Load hybrid dataset
    print(f"\n📂 Loading hybrid dataset...")
    dataset = HybridChessDataset(
        real_data_dir=args.real_data,
        synthetic_data_dir=args.synthetic_data if Path(args.synthetic_data).exists() else None,
        real_synthetic_ratio=config['real_synthetic_ratio']
    )

    # Analyze dataset
    dataset.analyze_dataset()

    # Split dataset
    num_samples = len(dataset.combined_samples)
    indices = list(range(num_samples))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(train_indices):,} samples")
    print(f"   Validation: {len(val_indices):,} samples")

    # Calculate steps
    steps_per_epoch = len(train_indices) // config['batch_size']
    validation_steps = len(val_indices) // config['batch_size']

    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")

    # Create data generators
    print(f"\n🔄 Creating data generators...")
    train_gen, val_gen = dataset.create_generators(
        train_indices, val_indices,
        batch_size=config['batch_size'],
        augment_training=config['augmentation']
    )

    # Build model
    print(f"\n🏗️ Building model...")
    cnn = ChessBoardCNN()
    model = cnn.build_model(backbone=config['backbone'])

    print(f"\n📋 Model Summary:")
    model.summary()

    # Start training
    print(f"\n🚀 Starting hybrid training...")
    estimated_time = config['epochs'] * steps_per_epoch * config['batch_size'] / 200
    print(f"⏱️  Estimated time: {estimated_time:.1f} minutes")

    start_time = time.time()

    history = cnn.progressive_train(
        train_gen, val_gen,
        steps_per_epoch, validation_steps,
        config
    )

    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time / 60:.1f} minutes")

    # Save final model
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = output_dir / "final_hybrid_chess_cnn.keras"
    cnn.model.save(final_model_path)

    print(f"\n💾 Model saved to: {final_model_path}")
    print(f"📁 All outputs saved to: {output_dir}")

    # Save config
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n🎉 Hybrid training complete!")
    print(f"🎯 Real data ratio achieved: {config['real_synthetic_ratio']:.1%}")


if __name__ == "__main__":
    main()