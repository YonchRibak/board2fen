# variance_analysis.py - Analyze why your model has inconsistent performance

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import seaborn as sns


def analyze_model_variance(model_path: str,
                           dataset_path: str = "c:/datasets/ChessRender360",
                           num_samples: int = 50):
    """
    Analyze why your model has inconsistent performance on training data

    Your debugger showed 21.7px error, but original diagnostic showed 48.8px error.
    This suggests high variance - model works well on some samples, poorly on others.
    """

    print("📊 MODEL PERFORMANCE VARIANCE ANALYSIS")
    print("=" * 50)
    print("Investigating why your model has inconsistent performance...")

    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return

    # Test on multiple samples to find patterns
    dataset_path = Path(dataset_path)
    rgb_dir = dataset_path / "rgb"
    ann_dir = dataset_path / "annotations"

    # Collect results from many samples
    results = []

    print(f"\n🔍 Testing {num_samples} training samples...")

    for i in range(num_samples):
        rgb_file = rgb_dir / f"rgb_{i}.jpeg"
        ann_file = ann_dir / f"annotation_{i}.json"

        if not rgb_file.exists() or not ann_file.exists():
            continue

        try:
            # Load and process sample
            image = cv2.imread(str(rgb_file))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]

            with open(ann_file) as f:
                annotation = json.load(f)

            corners = annotation['board_corners']

            # Ground truth (normalized)
            gt_normalized = [
                corners['white_left'][0] / w, corners['white_left'][1] / h,
                corners['white_right'][0] / w, corners['white_right'][1] / h,
                corners['black_right'][0] / w, corners['black_right'][1] / h,
                corners['black_left'][0] / w, corners['black_left'][1] / h
            ]

            # Model prediction
            processed = cv2.resize(image_rgb, (224, 224)).astype(np.float32) / 255.0
            prediction = model.predict(np.expand_dims(processed, 0), verbose=0)[0]

            # Calculate errors
            errors = np.abs(np.array(gt_normalized) - np.array(prediction))
            pixel_errors = errors * np.array([w, h] * 4)
            mean_pixel_error = np.mean(pixel_errors)

            # Analyze image characteristics that might affect performance
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)

            # Board characteristics
            all_corners = [corners['white_left'], corners['white_right'],
                           corners['black_right'], corners['black_left']]
            board_area = cv2.contourArea(np.array(all_corners, dtype=np.int32))
            board_area_ratio = board_area / (w * h)

            # Corner spread (how spread out are the corners)
            corner_coords = np.array(all_corners)
            corner_spread = np.std(corner_coords, axis=0).mean()

            results.append({
                'sample_id': i,
                'mean_pixel_error': mean_pixel_error,
                'max_pixel_error': np.max(pixel_errors),
                'brightness': brightness,
                'contrast': contrast,
                'board_area_ratio': board_area_ratio,
                'corner_spread': corner_spread,
                'image_size': f"{w}x{h}"
            })

            # Print progress for worst cases
            if mean_pixel_error > 40:
                print(f"  Sample {i}: {mean_pixel_error:.1f}px error (HIGH)")
            elif i < 10:
                print(f"  Sample {i}: {mean_pixel_error:.1f}px error")

        except Exception as e:
            print(f"  ❌ Error with sample {i}: {e}")

    if not results:
        print("❌ No results collected")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Statistical analysis
    print(f"\n📊 PERFORMANCE STATISTICS")
    print("=" * 30)
    print(f"Samples tested: {len(df)}")
    print(f"Mean pixel error: {df['mean_pixel_error'].mean():.1f} ± {df['mean_pixel_error'].std():.1f}")
    print(f"Median pixel error: {df['mean_pixel_error'].median():.1f}")
    print(f"Min pixel error: {df['mean_pixel_error'].min():.1f}")
    print(f"Max pixel error: {df['mean_pixel_error'].max():.1f}")

    # Identify problem categories
    good_samples = df[df['mean_pixel_error'] <= 25]
    poor_samples = df[df['mean_pixel_error'] > 50]

    print(f"\nPerformance categories:")
    print(f"  Good (≤25px):  {len(good_samples)}/{len(df)} ({len(good_samples) / len(df) * 100:.1f}%)")
    print(f"  Poor (>50px):  {len(poor_samples)}/{len(df)} ({len(poor_samples) / len(df) * 100:.1f}%)")

    # Find patterns in poor performance
    if len(poor_samples) > 0:
        print(f"\n🚨 POOR PERFORMANCE ANALYSIS")
        print("=" * 35)
        print("Worst performing samples:")
        worst_samples = poor_samples.nlargest(5, 'mean_pixel_error')[
            ['sample_id', 'mean_pixel_error', 'brightness', 'contrast', 'board_area_ratio']]
        print(worst_samples.to_string(index=False))

        # Statistical comparison
        print(f"\nCharacteristics comparison:")
        print(f"                     Good Samples    Poor Samples")
        print(
            f"Brightness:          {good_samples['brightness'].mean():.1f}           {poor_samples['brightness'].mean():.1f}")
        print(
            f"Contrast:            {good_samples['contrast'].mean():.1f}            {poor_samples['contrast'].mean():.1f}")
        print(
            f"Board area ratio:    {good_samples['board_area_ratio'].mean():.3f}         {poor_samples['board_area_ratio'].mean():.3f}")

    # Visualizations
    create_performance_analysis_plots(df)

    # Specific recommendations
    print(f"\n💡 DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 35)

    variance = df['mean_pixel_error'].std()
    mean_error = df['mean_pixel_error'].mean()

    if variance > 20:
        print("🚨 HIGH VARIANCE DETECTED!")
        print("   Your model has inconsistent performance across training data")
        print("   This explains the discrepancy between diagnostics")

    if mean_error > 40:
        print("⚠️  OVERALL POOR PERFORMANCE")
        print("   Model needs more training or architecture improvements")

    if len(poor_samples) > len(df) * 0.3:
        print("⚠️  MANY DIFFICULT SAMPLES")
        print("   30%+ of training data performs poorly")

    # Specific recommendations
    print(f"\nRecommendations:")

    if variance > 20:
        print("1. 🔧 REDUCE VARIANCE:")
        print("   • Add more regularization (dropout, weight decay)")
        print("   • Use ensemble methods")
        print("   • Increase training data augmentation")

    if len(poor_samples) > 0:
        print("2. 🎯 TARGET DIFFICULT CASES:")
        print("   • Analyze worst-performing samples")
        print("   • Add specific augmentation for challenging scenarios")
        print("   • Consider weighted loss to focus on hard examples")

    print("3. 🔄 TRAINING IMPROVEMENTS:")
    print("   • Use progressive training (easy → hard samples)")
    print("   • Implement curriculum learning")
    print("   • Add focal loss for hard examples")

    return df


def create_performance_analysis_plots(df):
    """Create visualizations of performance variance"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Error distribution histogram
    ax1.hist(df['mean_pixel_error'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(df['mean_pixel_error'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["mean_pixel_error"].mean():.1f}px')
    ax1.axvline(df['mean_pixel_error'].median(), color='green', linestyle='--',
                label=f'Median: {df["mean_pixel_error"].median():.1f}px')
    ax1.set_xlabel('Mean Pixel Error')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Distribution of Pixel Errors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Error vs Brightness
    ax2.scatter(df['brightness'], df['mean_pixel_error'], alpha=0.6, color='orange')
    ax2.set_xlabel('Image Brightness')
    ax2.set_ylabel('Mean Pixel Error')
    ax2.set_title('Error vs Image Brightness')
    ax2.grid(True, alpha=0.3)

    # Add correlation info
    correlation = df['brightness'].corr(df['mean_pixel_error'])
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

    # 3. Error vs Contrast
    ax3.scatter(df['contrast'], df['mean_pixel_error'], alpha=0.6, color='purple')
    ax3.set_xlabel('Image Contrast (std)')
    ax3.set_ylabel('Mean Pixel Error')
    ax3.set_title('Error vs Image Contrast')
    ax3.grid(True, alpha=0.3)

    correlation = df['contrast'].corr(df['mean_pixel_error'])
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

    # 4. Error vs Board Area Ratio
    ax4.scatter(df['board_area_ratio'], df['mean_pixel_error'], alpha=0.6, color='green')
    ax4.set_xlabel('Board Area Ratio')
    ax4.set_ylabel('Mean Pixel Error')
    ax4.set_title('Error vs Board Size')
    ax4.grid(True, alpha=0.3)

    correlation = df['board_area_ratio'].corr(df['mean_pixel_error'])
    ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

    plt.tight_layout()
    plt.show()


def visualize_worst_cases(model_path: str,
                          dataset_path: str = "c:/datasets/ChessRender360",
                          num_worst: int = 4):
    """Visualize the worst-performing training samples"""

    print(f"\n🔍 VISUALIZING WORST CASES")
    print("=" * 30)

    # Run quick analysis to find worst cases
    df = analyze_model_variance(model_path, dataset_path, num_samples=100)

    if df is None or len(df) == 0:
        return

    # Get worst performing samples
    worst_samples = df.nlargest(num_worst, 'mean_pixel_error')

    model = tf.keras.models.load_model(model_path)
    dataset_path = Path(dataset_path)

    fig, axes = plt.subplots(2, num_worst // 2, figsize=(20, 10))
    axes = axes.flatten() if num_worst > 1 else [axes]

    for i, (_, row) in enumerate(worst_samples.iterrows()):
        if i >= len(axes):
            break

        sample_id = int(row['sample_id'])
        rgb_file = dataset_path / "rgb" / f"rgb_{sample_id}.jpeg"
        ann_file = dataset_path / "annotations" / f"annotation_{sample_id}.json"

        # Load and process
        image = cv2.imread(str(rgb_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(ann_file) as f:
            annotation = json.load(f)

        corners = annotation['board_corners']
        gt_corners = np.array([corners['white_left'], corners['white_right'],
                               corners['black_right'], corners['black_left']])

        # Model prediction
        processed = cv2.resize(image_rgb, (224, 224)).astype(np.float32) / 255.0
        prediction = model.predict(np.expand_dims(processed, 0), verbose=0)[0]

        h, w = image_rgb.shape[:2]
        pred_corners = []
        for j in range(0, 8, 2):
            x = int(prediction[j] * w)
            y = int(prediction[j + 1] * h)
            pred_corners.append([x, y])
        pred_corners = np.array(pred_corners)

        # Visualization
        vis_image = image_rgb.copy()

        # Ground truth in green
        cv2.polylines(vis_image, [gt_corners.astype(int)], True, (0, 255, 0), 3)
        for corner in gt_corners:
            cv2.circle(vis_image, tuple(corner.astype(int)), 8, (0, 255, 0), -1)

        # Predictions in red
        cv2.polylines(vis_image, [pred_corners.astype(int)], True, (255, 0, 0), 3)
        for corner in pred_corners:
            cv2.circle(vis_image, tuple(corner.astype(int)), 8, (255, 0, 0), -1)

        axes[i].imshow(vis_image)
        axes[i].set_title(f'Sample {sample_id}: {row["mean_pixel_error"]:.1f}px error\n'
                          f'Brightness: {row["brightness"]:.0f}, Contrast: {row["contrast"]:.0f}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(worst_samples), len(axes)):
        axes[i].axis('off')

    plt.suptitle('Worst Performing Training Samples', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("📊 MODEL VARIANCE ANALYSIS")
    print("=" * 30)

    model_path = input("Enter model path (or press Enter for default): ").strip()
    if not model_path:
        model_path = "outputs/corner_detector/fine_tuned_chess_corner_detector.keras"

    # Run analysis
    df = analyze_model_variance(model_path, num_samples=100)

    if df is not None:
        # Ask if user wants to see worst cases
        show_worst = input("\nVisualize worst-performing samples? (y/N): ").strip().lower()
        if show_worst == 'y':
            visualize_worst_cases(model_path)