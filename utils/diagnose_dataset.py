# diagnose_dataset.py - Find out what's wrong with dataset loading

from pathlib import Path
import tensorflow as tf

# Your config imports
from legacy.piece_classifier.config import DATASET_DIR


def count_actual_files(data_dir):
    """Count actual files in each class folder"""
    data_path = Path(data_dir)

    print("🔍 ACTUAL FILE COUNTS IN FOLDERS:")
    print("=" * 50)

    total_files = 0
    class_counts = {}

    for class_folder in sorted(data_path.iterdir()):
        if class_folder.is_dir():
            # Count image files in this folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            file_count = 0

            for ext in image_extensions:
                file_count += len(list(class_folder.glob(f'*{ext}')))
                file_count += len(list(class_folder.glob(f'*{ext.upper()}')))

            class_counts[class_folder.name] = file_count
            total_files += file_count

            print(f"{class_folder.name:15}: {file_count:8,d} files")

    print(f"\nTotal files found: {total_files:,d}")
    return class_counts, total_files


def check_tensorflow_dataset_loading(data_dir):
    """Check how TensorFlow is loading the dataset"""
    print("\n🔍 TENSORFLOW DATASET LOADING:")
    print("=" * 50)

    # Try different loading methods

    # Method 1: image_dataset_from_directory with different settings
    try:
        print("\nTrying tf.keras.utils.image_dataset_from_directory...")

        ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(224, 224),
            batch_size=32  # Small batch size for testing
        )

        print(f"Dataset created successfully!")
        print(f"Dataset spec: {ds.element_spec}")

        # Count samples in first few batches
        sample_count = 0
        batch_count = 0

        for images, labels in ds.take(5):  # Just first 5 batches
            batch_count += 1
            batch_size = len(labels)
            sample_count += batch_size
            print(f"Batch {batch_count}: {batch_size} samples")

        print(f"Samples in first 5 batches: {sample_count}")

        # Get all class names
        class_names = ds.class_names
        print(f"Classes detected: {class_names}")

        return ds, class_names

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None


def check_data_directory_structure(data_dir):
    """Check the directory structure"""
    data_path = Path(data_dir)

    print(f"\n🔍 DIRECTORY STRUCTURE CHECK:")
    print("=" * 50)
    print(f"Data directory: {data_path}")
    print(f"Directory exists: {data_path.exists()}")

    if not data_path.exists():
        print("❌ Data directory doesn't exist!")
        return False

    # Check subdirectories
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} subdirectories:")

    for subdir in sorted(subdirs):
        file_count = len(list(subdir.iterdir()))
        print(f"  {subdir.name}: {file_count} items")

    return True


def main_diagnosis():
    """Main diagnosis function"""

    # You'll need to update this path to your actual data directory
    # Based on your config, it might be something like:
    data_dir = DATASET_DIR


    if data_dir is None:
        print("❌ Could not find data directory!")
        print("Please update the data_dir variable in this script.")
        print("Current working directory:", Path.cwd())
        return

    print(f"✅ Found data directory: {data_dir}")

    # 1. Check directory structure
    check_data_directory_structure(data_dir)

    # 2. Count actual files
    class_counts, total_files = count_actual_files(data_dir)

    # 3. Check TensorFlow loading
    ds, class_names = check_tensorflow_dataset_loading(data_dir)

    # 4. Analysis and recommendations
    print("\n📊 ANALYSIS & RECOMMENDATIONS:")
    print("=" * 50)

    if total_files > 100000:
        print(f"⚠️  VERY LARGE DATASET: {total_files:,d} files")
        print("   - Consider using a subset for initial training")
        print("   - Use smaller batch sizes")
        print("   - Consider data streaming/prefetching")

    # Check for extreme imbalance
    if class_counts:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        print(f"📈 Class imbalance ratio: {imbalance_ratio:.1f}:1")

        if imbalance_ratio > 50:
            print("   ⚠️  SEVERE CLASS IMBALANCE DETECTED!")
            print("   - Consider class balancing techniques")
            print("   - Use weighted loss functions")
            print("   - Oversample minority classes")
            print("   - Undersample majority classes")

    # Memory recommendations
    available_memory_gb = 16  # Estimate - adjust based on your system
    estimated_memory_need = total_files * 224 * 224 * 3 * 4 / (1024 ** 3)  # Rough estimate

    print(f"💾 Estimated memory need: {estimated_memory_need:.1f} GB")
    if estimated_memory_need > available_memory_gb * 0.8:
        print("   ⚠️  May exceed available memory!")
        print("   - Use .cache() and .prefetch() carefully")
        print("   - Consider training on subset first")
        print("   - Reduce image size or batch size")


if __name__ == "__main__":
    main_diagnosis()