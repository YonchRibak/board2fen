# validate_robust_model.py - Test and validate the robust corner detector

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
from typing import List


# # ADD this after the imports:
# @tf.keras.saving.register_keras_serializable()
# def robust_corner_loss(y_true, y_pred):
#     """Robust loss function for loading saved cnn_models"""
#     huber_loss = tf.keras.losses.Huber(delta=0.1)
#     base_loss = huber_loss(y_true, y_pred)
#
#     large_error_threshold = 0.2
#     error_magnitude = tf.abs(y_true - y_pred)
#     catastrophic_mask = error_magnitude > large_error_threshold
#
#     catastrophic_penalty = tf.where(
#         catastrophic_mask,
#         error_magnitude * 5.0,
#         0.0
#     )
#
#     return base_loss + tf.reduce_mean(catastrophic_penalty)

class RobustModelValidator:
    """
    Validate the improved robust corner detector

    Expected improvements:
    - Variance reduction: ±66.8px → ±20px
    - Good samples: 25% → 70%+
    - Eliminate catastrophic failures (>200px)
    - Dramatic real-world performance improvement
    """

    def __init__(self,
                 old_model_path: str,
                 new_model_path: str,
                 dataset_path: str = "c:/datasets/ChessRender360"):

        self.old_model_path = old_model_path
        self.new_model_path = new_model_path
        self.dataset_path = Path(dataset_path)

        self.old_model = None
        self.new_model = None

        # Load cnn_models
        self._load_models()

    def _load_models(self):
        """Load both old and new cnn_models for comparison"""

        try:
            print("📥 Loading cnn_models for comparison...")
            self.old_model = tf.keras.models.load_model(self.old_model_path)
            print(f"✅ Old model loaded: {self.old_model_path}")
        except Exception as e:
            print(f"⚠️  Could not load old model: {e}")

        try:
            # Load with custom_objects to handle custom loss
            custom_objects = {
                'robust_corner_loss': lambda y_true, y_pred: tf.keras.losses.Huber(delta=0.1)(y_true, y_pred)}
            self.new_model = tf.keras.models.load_model(self.new_model_path, custom_objects=custom_objects)
            print(f"✅ New model loaded: {self.new_model_path}")
        except Exception as e:
            print(f"❌ Could not load new model: {e}")
            raise

    def comprehensive_comparison(self, num_test_samples: int = 200):
        """
        Comprehensive comparison of old vs new model performance
        """

        print(f"\n📊 COMPREHENSIVE MODEL COMPARISON")
        print("=" * 50)
        print(f"Testing {num_test_samples} samples...")

        # Test both cnn_models on same samples
        old_results = self._test_model(self.old_model, num_test_samples, "Old Model")
        new_results = self._test_model(self.new_model, num_test_samples, "New Robust Model")

        # Compare results
        self._compare_results(old_results, new_results)

        # Visualize improvements
        self._visualize_improvements(old_results, new_results)

        return old_results, new_results

    def _test_model(self, model, num_samples: int, model_name: str):
        """Test a model on training samples"""

        print(f"\n🧪 Testing {model_name}...")

        rgb_dir = self.dataset_path / "rgb"
        ann_dir = self.dataset_path / "annotations"

        results = []

        for i in range(num_samples):
            rgb_file = rgb_dir / f"rgb_{i}.jpeg"
            ann_file = ann_dir / f"annotation_{i}.json"

            if not rgb_file.exists() or not ann_file.exists():
                continue

            try:
                # Load sample
                image = cv2.imread(str(rgb_file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]

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
                processed = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
                prediction = model.predict(np.expand_dims(processed, 0), verbose=0)[0]

                # Calculate errors
                errors = np.abs(np.array(gt_normalized) - np.array(prediction))
                pixel_errors = errors * np.array([w, h] * 4)
                mean_pixel_error = np.mean(pixel_errors)
                max_pixel_error = np.max(pixel_errors)

                # Classify sample difficulty
                board_area = cv2.contourArea(np.array([
                    corners['white_left'], corners['white_right'],
                    corners['black_right'], corners['black_left']
                ], dtype=np.int32))
                board_area_ratio = board_area / (w * h)

                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray)

                # Determine if this was a "hard" sample
                is_hard_sample = (board_area_ratio < 0.01 or brightness < 80)
                is_catastrophic = mean_pixel_error > 200

                results.append({
                    'sample_id': i,
                    'mean_error': mean_pixel_error,
                    'max_error': max_pixel_error,
                    'brightness': brightness,
                    'board_area_ratio': board_area_ratio,
                    'is_hard_sample': is_hard_sample,
                    'is_catastrophic': is_catastrophic
                })

                # Print progress for notable cases
                if mean_pixel_error > 100:
                    print(f"  Sample {i}: {mean_pixel_error:.1f}px error (HIGH)")
                elif i < 10 or i % 50 == 0:
                    print(f"  Sample {i}: {mean_pixel_error:.1f}px error")

            except Exception as e:
                continue

        return results

    def _compare_results(self, old_results, new_results):
        """Compare and analyze improvements"""

        print(f"\n📈 PERFORMANCE COMPARISON")
        print("=" * 35)

        # Convert to DataFrames
        old_df = pd.DataFrame(old_results)
        new_df = pd.DataFrame(new_results)

        # Align samples (only compare samples that exist in both)
        common_samples = set(old_df['sample_id']) & set(new_df['sample_id'])
        old_df = old_df[old_df['sample_id'].isin(common_samples)].sort_values('sample_id')
        new_df = new_df[new_df['sample_id'].isin(common_samples)].sort_values('sample_id')

        print(f"Comparing {len(common_samples)} samples...")

        # Overall statistics
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"                        Old Model    →    New Model    Improvement")
        print(
            f"Mean error:           {old_df['mean_error'].mean():6.1f}px    →   {new_df['mean_error'].mean():6.1f}px    {((old_df['mean_error'].mean() - new_df['mean_error'].mean()) / old_df['mean_error'].mean() * 100):+5.1f}%")
        print(
            f"Standard deviation:   {old_df['mean_error'].std():6.1f}px    →   {new_df['mean_error'].std():6.1f}px    {((old_df['mean_error'].std() - new_df['mean_error'].std()) / old_df['mean_error'].std() * 100):+5.1f}%")
        print(
            f"Max error:            {old_df['max_error'].max():6.1f}px    →   {new_df['max_error'].max():6.1f}px    {((old_df['max_error'].max() - new_df['max_error'].max()) / old_df['max_error'].max() * 100):+5.1f}%")
        print(
            f"Median error:         {old_df['mean_error'].median():6.1f}px    →   {new_df['mean_error'].median():6.1f}px    {((old_df['mean_error'].median() - new_df['mean_error'].median()) / old_df['mean_error'].median() * 100):+5.1f}%")

        # Performance categories
        old_good = len(old_df[old_df['mean_error'] <= 25])
        new_good = len(new_df[new_df['mean_error'] <= 25])
        old_poor = len(old_df[old_df['mean_error'] > 50])
        new_poor = len(new_df[new_df['mean_error'] > 50])
        old_catastrophic = len(old_df[old_df['mean_error'] > 200])
        new_catastrophic = len(new_df[new_df['mean_error'] > 200])

        total_samples = len(common_samples)

        print(f"\n🎯 PERFORMANCE CATEGORIES:")
        print(f"                        Old Model              →    New Model")
        print(
            f"Good (≤25px):          {old_good:3d}/{total_samples} ({old_good / total_samples * 100:4.1f}%)    →   {new_good:3d}/{total_samples} ({new_good / total_samples * 100:4.1f}%)")
        print(
            f"Poor (>50px):          {old_poor:3d}/{total_samples} ({old_poor / total_samples * 100:4.1f}%)    →   {new_poor:3d}/{total_samples} ({new_poor / total_samples * 100:4.1f}%)")
        print(
            f"Catastrophic (>200px): {old_catastrophic:3d}/{total_samples} ({old_catastrophic / total_samples * 100:4.1f}%)    →   {new_catastrophic:3d}/{total_samples} ({new_catastrophic / total_samples * 100:4.1f}%)")

        # Hard sample analysis
        hard_mask = old_df['is_hard_sample']
        if np.any(hard_mask):
            old_hard = old_df[hard_mask]
            new_hard = new_df[hard_mask]

            print(f"\n🔥 HARD SAMPLE PERFORMANCE:")
            print(f"Hard samples found: {len(old_hard)}")
            if len(old_hard) > 0:
                print(
                    f"Old model hard sample error: {old_hard['mean_error'].mean():.1f} ± {old_hard['mean_error'].std():.1f}px")
                print(
                    f"New model hard sample error: {new_hard['mean_error'].mean():.1f} ± {new_hard['mean_error'].std():.1f}px")

                improvement = (old_hard['mean_error'].mean() - new_hard['mean_error'].mean()) / old_hard[
                    'mean_error'].mean() * 100
                print(f"Hard sample improvement: {improvement:+.1f}%")

        # Success assessment
        print(f"\n🏆 SUCCESS ASSESSMENT:")

        variance_reduction = (old_df['mean_error'].std() - new_df['mean_error'].std()) / old_df[
            'mean_error'].std() * 100
        good_improvement = (new_good - old_good) / total_samples * 100
        catastrophic_reduction = (old_catastrophic - new_catastrophic)

        success_criteria = {
            'Variance reduction (target: >50%)': variance_reduction > 50,
            'Good samples increase (target: >30%)': good_improvement > 30,
            'Catastrophic failures eliminated': new_catastrophic == 0,
            'Overall error reduction (target: >25%)': ((old_df['mean_error'].mean() - new_df['mean_error'].mean()) /
                                                       old_df['mean_error'].mean() * 100) > 25
        }

        for criterion, achieved in success_criteria.items():
            status = "✅ ACHIEVED" if achieved else "❌ NOT MET"
            print(f"  {criterion}: {status}")

        success_rate = sum(success_criteria.values()) / len(success_criteria)
        print(f"\nOverall success rate: {success_rate * 100:.0f}%")

        if success_rate >= 0.75:
            print("🎉 EXCELLENT! Training succeeded on most objectives!")
        elif success_rate >= 0.5:
            print("✅ GOOD! Significant improvements achieved!")
        else:
            print("⚠️  PARTIAL SUCCESS - Some improvements, but more work needed")

    def _visualize_improvements(self, old_results, new_results):
        """Create visualizations showing improvements"""

        old_df = pd.DataFrame(old_results)
        new_df = pd.DataFrame(new_results)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # 1. Error distribution comparison
        ax1.hist(old_df['mean_error'], bins=30, alpha=0.7, label='Old Model', color='red', edgecolor='black')
        ax1.hist(new_df['mean_error'], bins=30, alpha=0.7, label='New Model', color='green', edgecolor='black')
        ax1.set_xlabel('Mean Pixel Error')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Error Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(25, color='blue', linestyle='--', alpha=0.8, label='Good threshold (25px)')
        ax1.axvline(50, color='orange', linestyle='--', alpha=0.8, label='Poor threshold (50px)')
        ax1.axvline(200, color='red', linestyle='--', alpha=0.8, label='Catastrophic threshold (200px)')

        # 2. Scatter plot: Old vs New errors
        common_samples = set(old_df['sample_id']) & set(new_df['sample_id'])
        old_aligned = old_df[old_df['sample_id'].isin(common_samples)].sort_values('sample_id')
        new_aligned = new_df[new_df['sample_id'].isin(common_samples)].sort_values('sample_id')

        ax2.scatter(old_aligned['mean_error'], new_aligned['mean_error'], alpha=0.6, s=20)
        max_error = max(old_aligned['mean_error'].max(), new_aligned['mean_error'].max())
        ax2.plot([0, max_error], [0, max_error], 'r--', alpha=0.8, label='No improvement')
        ax2.set_xlabel('Old Model Error (pixels)')
        ax2.set_ylabel('New Model Error (pixels)')
        ax2.set_title('Per-Sample Error Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add text showing improvements
        improvement_count = np.sum(new_aligned['mean_error'] < old_aligned['mean_error'])
        total_count = len(common_samples)
        ax2.text(0.05, 0.95,
                 f'Improved: {improvement_count}/{total_count} ({improvement_count / total_count * 100:.1f}%)',
                 transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))

        # 3. Performance categories
        categories = ['Good\n(≤25px)', 'Medium\n(25-50px)', 'Poor\n(50-200px)', 'Catastrophic\n(>200px)']

        old_counts = [
            len(old_df[old_df['mean_error'] <= 25]),
            len(old_df[(old_df['mean_error'] > 25) & (old_df['mean_error'] <= 50)]),
            len(old_df[(old_df['mean_error'] > 50) & (old_df['mean_error'] <= 200)]),
            len(old_df[old_df['mean_error'] > 200])
        ]

        new_counts = [
            len(new_df[new_df['mean_error'] <= 25]),
            len(new_df[(new_df['mean_error'] > 25) & (new_df['mean_error'] <= 50)]),
            len(new_df[(new_df['mean_error'] > 50) & (new_df['mean_error'] <= 200)]),
            len(new_df[new_df['mean_error'] > 200])
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax3.bar(x - width / 2, old_counts, width, label='Old Model', color='red', alpha=0.7)
        ax3.bar(x + width / 2, new_counts, width, label='New Model', color='green', alpha=0.7)

        ax3.set_xlabel('Performance Category')
        ax3.set_ylabel('Number of Samples')
        ax3.set_title('Performance Category Distribution')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Variance comparison
        old_var = old_df['mean_error'].std()
        new_var = new_df['mean_error'].std()

        variance_data = ['Old Model', 'New Model']
        variance_values = [old_var, new_var]
        colors = ['red', 'green']

        ax4.bar(variance_data, variance_values, color=colors, alpha=0.7)
        ax4.set_ylabel('Standard Deviation (pixels)')
        ax4.set_title('Error Variance Comparison')
        ax4.grid(True, alpha=0.3)

        # Add improvement percentage
        var_improvement = (old_var - new_var) / old_var * 100
        ax4.text(0.5, max(variance_values) * 0.8, f'Variance Reduction:\n{var_improvement:+.1f}%',
                 ha='center', bbox=dict(boxstyle="round", facecolor='lightblue'))

        plt.suptitle('Robust Corner Detector: Performance Improvements', fontsize=16)
        plt.tight_layout()
        plt.show()

    def test_real_world_images(self, real_image_paths: List[str]):
        """Test both cnn_models on real-world images to verify real-world improvement"""

        print(f"\n🌍 REAL-WORLD IMAGE TESTING")
        print("=" * 35)
        print(f"Testing {len(real_image_paths)} real images...")

        results = {'old': [], 'new': []}

        for i, image_path in enumerate(real_image_paths):
            print(f"\n--- Real Image {i + 1}: {Path(image_path).name} ---")

            try:
                # Load image
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Test both cnn_models
                old_corners = self._predict_corners(self.old_model, image_rgb)
                new_corners = self._predict_corners(self.new_model, image_rgb)

                # Calculate confidence scores
                old_confidence = self._calculate_confidence(image_rgb, old_corners)
                new_confidence = self._calculate_confidence(image_rgb, new_corners)

                print(f"Old model confidence: {old_confidence:.3f}")
                print(f"New model confidence: {new_confidence:.3f}")
                print(f"Improvement: {(new_confidence - old_confidence):+.3f}")

                results['old'].append(old_confidence)
                results['new'].append(new_confidence)

                # Visualize comparison for first few images
                if i < 3:
                    self._visualize_real_world_comparison(image_rgb, old_corners, new_corners,
                                                          old_confidence, new_confidence, image_path)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        # Summary
        if results['old'] and results['new']:
            old_avg = np.mean(results['old'])
            new_avg = np.mean(results['new'])
            improvement = new_avg - old_avg

            print(f"\n📊 REAL-WORLD SUMMARY:")
            print(f"Old model avg confidence: {old_avg:.3f}")
            print(f"New model avg confidence: {new_avg:.3f}")
            print(f"Average improvement: {improvement:+.3f}")

            improved_count = sum(1 for old, new in zip(results['old'], results['new']) if new > old)
            print(
                f"Images with improvement: {improved_count}/{len(results['old'])} ({improved_count / len(results['old']) * 100:.1f}%)")

    def _predict_corners(self, model, image):
        """Predict corners with preprocessing"""
        processed = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
        prediction = model.predict(np.expand_dims(processed, 0), verbose=0)[0]

        h, w = image.shape[:2]
        corners = []
        for i in range(0, 8, 2):
            x = int(prediction[i] * w)
            y = int(prediction[i + 1] * h)
            corners.append([x, y])

        return np.array(corners)

    def _calculate_confidence(self, image, corners):
        """Calculate confidence score for corner predictions"""
        h, w = image.shape[:2]

        confidence = 1.0

        # Check bounds
        for corner in corners:
            if corner[0] < 0 or corner[0] >= w or corner[1] < 0 or corner[1] >= h:
                confidence *= 0.1

        # Check area
        area = cv2.contourArea(corners.astype(np.int32))
        area_ratio = area / (h * w)

        if 0.1 <= area_ratio <= 0.8:
            confidence *= 1.0
        elif 0.05 <= area_ratio < 0.1 or 0.8 < area_ratio <= 0.95:
            confidence *= 0.7
        else:
            confidence *= 0.3

        return min(confidence, 1.0)

    def _visualize_real_world_comparison(self, image, old_corners, new_corners,
                                         old_conf, new_conf, image_path):
        """Visualize comparison on real world image"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Old model
        img_old = image.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for corner, color in zip(old_corners, colors):
            cv2.circle(img_old, tuple(corner.astype(int)), 12, color, -1)
        cv2.polylines(img_old, [old_corners.astype(int)], True, (255, 0, 0), 3)

        ax1.imshow(img_old)
        ax1.set_title(f'Old Model\nConfidence: {old_conf:.3f}')
        ax1.axis('off')

        # New model
        img_new = image.copy()
        for corner, color in zip(new_corners, colors):
            cv2.circle(img_new, tuple(corner.astype(int)), 12, color, -1)
        cv2.polylines(img_new, [new_corners.astype(int)], True, (0, 255, 0), 3)

        ax2.imshow(img_new)
        ax2.set_title(f'Robust Model\nConfidence: {new_conf:.3f}')
        ax2.axis('off')

        improvement = new_conf - old_conf
        fig.suptitle(f'{Path(image_path).name} - Improvement: {improvement:+.3f}', fontsize=16)

        plt.tight_layout()
        plt.show()


def main():
    """Main validation script"""

    print("📊 ROBUST CORNER DETECTOR VALIDATION")
    print("=" * 50)

    # Paths
    old_model = input("Enter path to old model (or press Enter for default): ").strip()
    if not old_model:
        old_model = "outputs/corner_detection/fine_tuned_chess_corner_detector.keras"

    new_model = input("Enter path to new robust model (or press Enter for default): ").strip()
    if not new_model:
        new_model = "outputs/robust_corner_detector/final_fine_tuned_model.keras"

    # Initialize validator
    validator = RobustModelValidator(old_model, new_model)

    # Run comprehensive comparison
    print("\n🔄 Starting comprehensive comparison...")
    old_results, new_results = validator.comprehensive_comparison(num_test_samples=100)

    # Test real-world images if available
    real_images_input = input("\nEnter real image paths (comma-separated, or press Enter to skip): ").strip()
    if real_images_input:
        real_images = [path.strip() for path in real_images_input.split(',')]
        real_images = [img for img in real_images if Path(img).exists()]

        if real_images:
            validator.test_real_world_images(real_images)

    print("\n✅ Validation complete!")
    print("Expected results with robust training:")
    print("  • Variance reduction: 50%+ improvement")
    print("  • Good samples: 30%+ increase")
    print("  • Catastrophic failures: Eliminated")
    print("  • Real-world performance: Dramatically improved")


if __name__ == "__main__":
    main()