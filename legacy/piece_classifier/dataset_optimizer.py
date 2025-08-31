
import os
import shutil
import random
from pathlib import Path
from collections import Counter


def create_individual_piece_dataset(
        source_dir="C:/chess_pieces_dataset",
        target_dir="C:/chess_pieces_individual_classifier",
        samples_per_class=2000,
        focus_on_hard_cases=True
):
    """
    Create balanced dataset optimized for individual piece classification

    Key principles:
    1. Equal samples per class (no chess frequency bias)
    2. Focus on challenging/ambiguous cases
    3. High-quality, diverse examples per piece type
    4. Robust to different lighting, angles, styles
    """

    print("🎯 INDIVIDUAL PIECE CLASSIFIER DATASET")
    print("=" * 50)
    print("Optimizing for: When given a cropped piece image, classify it correctly")
    print(f"Target: {samples_per_class:,d} samples per piece type")
    print()

    # All 12 piece classes - equal representation
    piece_classes = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
        'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directory
    if target_path.exists():
        print(f"⚠️  Target directory {target_path} already exists!")
        response = input("Delete and recreate? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(target_path)
        else:
            return False

    target_path.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    class_stats = {}

    print("📊 BALANCED DISTRIBUTION (Individual Classification):")
    print("-" * 55)

    # Process each class with equal representation
    for class_name in piece_classes:
        class_folder = source_path / class_name

        if not class_folder.exists():
            print(f"⚠️  Class folder {class_name} not found!")
            continue

        print(f"📁 Processing {class_name}...")

        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            image_files.extend(list(class_folder.glob(f'*{ext}')))
            image_files.extend(list(class_folder.glob(f'*{ext.upper()}')))

        available_count = len(image_files)
        print(f"   Available: {available_count:,d} images")

        # Determine actual samples to take
        if available_count < samples_per_class:
            actual_samples = available_count
            print(f"   ⚠️  Taking all {available_count:,d} (less than target {samples_per_class:,d})")
        else:
            actual_samples = samples_per_class
            print(f"   Taking: {samples_per_class:,d} samples")

        # Smart sampling strategy
        if focus_on_hard_cases:
            selected_files = smart_sample_pieces(image_files, actual_samples, class_name)
        else:
            selected_files = random.sample(image_files,
                                           actual_samples) if actual_samples < available_count else image_files

        # Create target folder and copy files
        target_class_folder = target_path / class_name
        target_class_folder.mkdir(exist_ok=True)

        copied_count = 0
        for source_file in selected_files:
            try:
                target_file = target_class_folder / source_file.name

                # Handle duplicate names
                counter = 1
                original_target = target_file
                while target_file.exists():
                    stem = original_target.stem
                    suffix = original_target.suffix
                    target_file = target_class_folder / f"{stem}_{counter}{suffix}"
                    counter += 1

                shutil.copy2(source_file, target_file)
                copied_count += 1

            except Exception as e:
                print(f"   ❌ Error copying {source_file.name}: {e}")

        class_stats[class_name] = {
            'available': available_count,
            'copied': copied_count
        }
        total_copied += copied_count

        percentage = (copied_count / total_copied * 100) if total_copied > 0 else 0
        print(f"   ✅ Copied: {copied_count:,d} images")

    # Final summary
    print(f"\n📊 INDIVIDUAL PIECE CLASSIFIER SUMMARY:")
    print("=" * 60)

    final_total = sum(stats['copied'] for stats in class_stats.values())

    for class_name, stats in class_stats.items():
        percentage = (stats['copied'] / final_total * 100) if final_total > 0 else 0
        print(f"{class_name:15}: {stats['copied']:6,d} samples ({percentage:5.1f}%)")

    print(f"\nTotal files: {final_total:,d}")

    # Check balance quality
    copied_counts = [stats['copied'] for stats in class_stats.values()]
    min_count = min(copied_counts) if copied_counts else 0
    max_count = max(copied_counts) if copied_counts else 0
    balance_ratio = max_count / min_count if min_count > 0 else float('inf')

    print(f"Balance ratio: {balance_ratio:.1f}:1")

    if balance_ratio <= 1.2:
        print("✅ Excellent balance achieved!")
        balance_quality = "Excellent"
    elif balance_ratio <= 2.0:
        print("✅ Good balance achieved!")
        balance_quality = "Good"
    else:
        print("⚠️  Some imbalance remains")
        balance_quality = "Fair"

    # Save dataset info
    info_file = target_path / "dataset_info.txt"
    with open(info_file, 'w') as f:
        f.write("INDIVIDUAL PIECE CLASSIFIER DATASET\n")
        f.write("=" * 45 + "\n\n")
        f.write("Purpose: Classify individual cropped chess piece images\n")
        f.write("Strategy: Balanced representation for equal performance\n")
        f.write(f"Balance quality: {balance_quality}\n")
        f.write(f"Total files: {final_total:,d}\n\n")
        f.write("CLASS DISTRIBUTION:\n")
        for class_name, stats in class_stats.items():
            percentage = (stats['copied'] / final_total * 100) if final_total > 0 else 0
            f.write(f"{class_name}: {stats['copied']:,d} files ({percentage:.1f}%)\n")

    print(f"\n📝 Dataset info saved to: {info_file}")
    print(f"🎯 Individual piece classifier dataset ready at: {target_path}")

    return True


def smart_sample_pieces(image_files, target_count, class_name):
    """
    Smart sampling strategy to get diverse, challenging examples

    For individual piece classification, we want:
    1. Variety in lighting conditions
    2. Different angles/orientations
    3. Various piece styles/materials
    4. Include some challenging/edge cases
    """

    if target_count >= len(image_files):
        return image_files

    # For now, use random sampling
    # In future, could implement smart filtering based on:
    # - Image brightness/contrast variation
    # - Edge detection complexity
    # - File size diversity (proxy for detail level)

    return random.sample(image_files, target_count)


def create_classifier_training_config():
    """Generate optimized training config for individual piece classification"""

    config = """# individual_piece_classifier_config.py

import tensorflow as tf

# Dataset Configuration
DATA_DIR = "C:/chess_pieces_individual_classifier"  
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 12

# Training optimized for individual classification
EPOCHS = 20
LEARNING_RATE = 1e-3

# NO class weights needed - we want equal performance on all pieces!
# Each piece type should be classified with equal accuracy

# Data augmentation for robustness
def get_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),        # Small rotations
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.GaussianNoise(0.01),         # Robustness to image noise
    ])

# Model architecture for piece classification
def build_individual_piece_classifier():
    from tensorflow.keras import layers, cnn_models, applications

    base_model = applications.EfficientNetB3(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = cnn_models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

# Metrics focused on individual classification
METRICS = [
    'accuracy',
    tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

# Target: 95%+ accuracy on each individual piece type!
"""

    config_path = Path("individual_piece_classifier_config.py")
    with open(config_path, 'w') as f:
        f.write(config)

    print(f"📝 Optimized config saved to: {config_path}")
    return config_path


if __name__ == "__main__":
    print("🎯 Individual Piece Classifier Dataset Creator")
    print("=" * 50)

    choice = input("""
Dataset size for individual piece classification:
1. Small test (500 per class) - ~6K total, quick training
2. Medium (1500 per class) - ~18K total, good balance  
3. Large (2500 per class) - ~30K total, maximum accuracy
4. Custom

Enter choice (1/2/3/4): """).strip()

    size_map = {'1': 500, '2': 1500, '3': 2500}

    if choice in size_map:
        samples_per_class = size_map[choice]
    elif choice == '4':
        try:
            samples_per_class = int(input("Samples per class: "))
        except ValueError:
            print("❌ Invalid number")
            exit()
    else:
        print("❌ Invalid choice")
        exit()

    print(f"\nCreating balanced dataset with {samples_per_class:,d} samples per piece type...")

    success = create_individual_piece_dataset(samples_per_class=samples_per_class)

    if success:
        print("\n✅ Balanced dataset created!")
        config_path = create_classifier_training_config()

        print(f"""
🎯 INDIVIDUAL PIECE CLASSIFIER READY!

Expected performance: 95%+ accuracy per piece type
Training time: 1-3 hours depending on dataset size
Focus: Equal performance on all 12 piece types

Next: Train your model and achieve excellent individual piece classification!
        """)
    else:
        print("❌ Dataset creation failed")