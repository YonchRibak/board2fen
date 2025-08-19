# inspect_npy_files.py
# Inspect and visualize the generated chess grid labels

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import cv2


def load_dataset_info(dataset_dir: str):
    """Load dataset metadata and files"""
    dataset_path = Path(dataset_dir)

    print("📊 DATASET INSPECTION")
    print("=" * 50)

    # Load metadata
    with open(dataset_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    print(f"📁 Dataset: {metadata['dataset_source']}")
    print(f"📈 Samples: {metadata['num_samples']}")
    print(f"📋 Label types: {metadata['label_types']}")
    print(f"🎯 Grid shape: {metadata['grid_shape']}")

    # Load image paths
    with open(dataset_path / "image_paths.txt", 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]

    print(f"🖼️  Images: {len(image_paths)}")

    # Load all label arrays
    labels = {}
    for label_type in metadata['label_types']:
        labels[label_type] = np.load(dataset_path / f"labels_{label_type}.npy")
        shape = labels[label_type].shape
        dtype = labels[label_type].dtype
        unique_vals = np.unique(labels[label_type])
        print(f"  📊 {label_type}: {shape}, {dtype}, unique values: {unique_vals}")

    return metadata, image_paths, labels


def visualize_chess_position(labels_dict: dict, sample_idx: int = 0):
    """Visualize a chess position from the labels"""

    # Piece symbols for visualization
    piece_symbols = {
        0: '.',  # Empty
        1: 'P', 2: 'R', 3: 'N', 4: 'B', 5: 'Q', 6: 'K',  # White
        7: 'p', 8: 'r', 9: 'n', 10: 'b', 11: 'q', 12: 'k'  # Black
    }

    print(f"\n🔍 SAMPLE {sample_idx} VISUALIZATION")
    print("=" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for i, (label_type, labels) in enumerate(labels_dict.items()):
        if i >= 4:  # Only show first 4 label types
            break

        row, col = i // 2, i % 2
        ax = axes[row, col]

        # Get the grid for this sample
        grid = labels[sample_idx]

        # Create visualization
        im = ax.imshow(grid, cmap='tab20', vmin=0, vmax=12)
        ax.set_title(f'{label_type.title()} (Classes: {np.unique(grid)})')

        # Add grid lines
        for x in range(9):
            ax.axhline(x - 0.5, color='black', linewidth=1)
            ax.axvline(x - 0.5, color='black', linewidth=1)

        # Add text annotations for complete grid
        if label_type == 'complete':
            for row_idx in range(8):
                for col_idx in range(8):
                    value = grid[row_idx, col_idx]
                    symbol = piece_symbols.get(value, str(value))
                    ax.text(col_idx, row_idx, symbol, ha='center', va='center',
                            fontsize=12, fontweight='bold',
                            color='white' if value > 0 else 'gray')

        # Set proper ticks
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])

    plt.tight_layout()
    plt.show()

    # Print the complete position in readable format
    complete_grid = labels_dict['complete'][sample_idx]
    print(f"\n📋 Complete Position (Sample {sample_idx}):")
    print("   a b c d e f g h")
    for row in range(8):
        rank = 8 - row
        row_str = f"{rank}  "
        for col in range(8):
            value = complete_grid[row, col]
            symbol = piece_symbols.get(value, '?')
            row_str += f"{symbol} "
        print(row_str)


def analyze_label_distributions(labels_dict: dict):
    """Analyze the distribution of labels across the dataset"""

    print(f"\n📊 LABEL DISTRIBUTION ANALYSIS")
    print("=" * 50)

    for label_type, labels in labels_dict.items():
        print(f"\n🎯 {label_type.upper()}:")

        # Flatten all grids to get overall distribution
        flat_labels = labels.flatten()
        unique, counts = np.unique(flat_labels, return_counts=True)

        print(f"  Shape: {labels.shape}")
        print(f"  Data type: {labels.dtype}")
        print(f"  Total squares: {len(flat_labels):,}")

        print(f"  Class distribution:")
        for val, count in zip(unique, counts):
            percentage = count / len(flat_labels) * 100
            print(f"    Class {val}: {count:,} squares ({percentage:.1f}%)")

        # Check for any obvious issues
        if label_type == 'complete':
            empty_squares = counts[0] if 0 in unique else 0
            occupied_squares = len(flat_labels) - empty_squares
            print(f"  📈 Empty squares: {empty_squares:,} ({empty_squares / len(flat_labels) * 100:.1f}%)")
            print(f"  📈 Occupied squares: {occupied_squares:,} ({occupied_squares / len(flat_labels) * 100:.1f}%)")


def validate_label_consistency(labels_dict: dict, num_samples_to_check: int = 10):
    """Validate that different label types are consistent with each other"""

    print(f"\n✅ LABEL CONSISTENCY VALIDATION")
    print("=" * 50)

    if not all(key in labels_dict for key in ['complete', 'occupancy', 'color', 'piece_type']):
        print("⚠️  Not all required label types available for consistency check")
        return

    complete = labels_dict['complete']
    occupancy = labels_dict['occupancy']
    color = labels_dict['color']
    piece_type = labels_dict['piece_type']

    inconsistencies = 0
    samples_checked = min(num_samples_to_check, len(complete))

    for sample_idx in range(samples_checked):
        for row in range(8):
            for col in range(8):
                c_val = complete[sample_idx, row, col]
                o_val = occupancy[sample_idx, row, col]
                color_val = color[sample_idx, row, col]
                piece_val = piece_type[sample_idx, row, col]

                # Check occupancy consistency
                if c_val == 0:  # Empty
                    if o_val != 0 or color_val != 0 or piece_val != 0:
                        inconsistencies += 1
                else:  # Occupied
                    if o_val != 1:
                        inconsistencies += 1

                    # Check color consistency
                    if 1 <= c_val <= 6:  # White pieces
                        if color_val != 1:
                            inconsistencies += 1
                    elif 7 <= c_val <= 12:  # Black pieces
                        if color_val != 2:
                            inconsistencies += 1

    total_squares_checked = samples_checked * 64

    if inconsistencies == 0:
        print(f"✅ All {total_squares_checked:,} squares consistent across label types!")
    else:
        print(f"❌ Found {inconsistencies} inconsistencies in {total_squares_checked:,} squares")
        print(f"   Consistency rate: {(total_squares_checked - inconsistencies) / total_squares_checked * 100:.2f}%")


def show_sample_images_with_labels(dataset_dir: str, image_paths: list,
                                   labels_dict: dict, num_samples: int = 3):
    """Show actual chess images alongside their generated labels"""

    print(f"\n🖼️  SAMPLE IMAGES WITH LABELS")
    print("=" * 50)

    # Piece symbols for overlay
    piece_symbols = {
        0: '',  # Empty
        1: 'P', 2: 'R', 3: 'N', 4: 'B', 5: 'Q', 6: 'K',  # White
        7: 'p', 8: 'r', 9: 'n', 10: 'b', 11: 'q', 12: 'k'  # Black
    }

    for sample_idx in range(min(num_samples, len(image_paths))):
        print(f"\n📸 Sample {sample_idx}: {Path(image_paths[sample_idx]).name}")

        # Load and display image
        try:
            image = cv2.imread(image_paths[sample_idx])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Original image
                ax1.imshow(image)
                ax1.set_title(f"Original Image {sample_idx}")
                ax1.axis('off')

                # Complete labels as chess position
                complete_grid = labels_dict['complete'][sample_idx]

                # Create chess board visualization
                board_vis = np.ones((8, 8, 3))
                for row in range(8):
                    for col in range(8):
                        # Alternate square colors
                        if (row + col) % 2 == 0:
                            board_vis[row, col] = [0.9, 0.9, 0.8]  # Light square
                        else:
                            board_vis[row, col] = [0.6, 0.4, 0.3]  # Dark square

                ax2.imshow(board_vis)
                ax2.set_title(f"Generated Labels {sample_idx}")

                # Add piece symbols
                for row in range(8):
                    for col in range(8):
                        value = complete_grid[row, col]
                        if value > 0:  # Not empty
                            symbol = piece_symbols.get(value, '?')
                            color = 'white' if value <= 6 else 'black'
                            ax2.text(col, row, symbol, ha='center', va='center',
                                     fontsize=16, fontweight='bold', color=color)

                # Set proper ticks
                ax2.set_xticks(range(8))
                ax2.set_yticks(range(8))
                ax2.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
                ax2.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])

                # Add grid
                for x in range(9):
                    ax2.axhline(x - 0.5, color='black', linewidth=1)
                    ax2.axvline(x - 0.5, color='black', linewidth=1)

                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"❌ Error loading image {sample_idx}: {e}")


def main():
    """Main inspection function"""

    # Configuration
    DATASET_DIR = "C:/datasets/ChessRender360_GridLabels"  # Update this path

    if not Path(DATASET_DIR).exists():
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        print("Please update the DATASET_DIR variable")
        return

    try:
        # Load dataset info
        metadata, image_paths, labels = load_dataset_info(DATASET_DIR)

        if len(labels) == 0:
            print("❌ No label files found!")
            return

        # Analyze distributions
        analyze_label_distributions(labels)

        # Validate consistency
        validate_label_consistency(labels)

        # Visualize sample positions
        if len(labels) > 0:
            # Show first few samples
            for i in range(min(3, list(labels.values())[0].shape[0])):
                visualize_chess_position(labels, sample_idx=i)

        # Show images with labels (if images exist)
        if image_paths:
            show_sample_images_with_labels(DATASET_DIR, image_paths, labels)

        print(f"\n🎉 Dataset inspection complete!")
        print(f"✅ Dataset looks good with {metadata['num_samples']} samples")

    except Exception as e:
        print(f"❌ Error during inspection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()