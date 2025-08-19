# corner_detector_diagnostics.py - Comprehensive debugging for corner detection model

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import seaborn as sns
from typing import Tuple, List, Dict
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class CornerDetectorDiagnostics:
    """Comprehensive diagnostics for corner detection model"""

    def __init__(self,
                 model_path: str,
                 dataset_path: str = "c:/datasets/ChessRender360"):
        """
        Initialize diagnostics

        Args:
            model_path: Path to trained model
            dataset_path: Path to original training dataset
        """
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.model = None
        self.input_size = (224, 224)

        # Corner labels
        self.corner_labels = ['white_left', 'white_right', 'black_right', 'black_left']

        # Load model
        self.load_model()

        print("🔍 CORNER DETECTOR DIAGNOSTICS INITIALIZED")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Dataset: {self.dataset_path}")

    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            print(f"✅ Model loaded: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def diagnose_training_vs_real_data(self,
                                       real_test_images: List[str],
                                       num_training_samples: int = 10):
        """
        Compare model performance on training data vs real test images

        Args:
            real_test_images: List of paths to real chess board images
            num_training_samples: Number of training samples to test
        """

        print("\n🔬 TRAINING VS REAL DATA COMPARISON")
        print("=" * 50)

        # Test on training data samples
        print("📊 Testing on TRAINING data samples...")
        training_results = self._test_on_training_data(num_training_samples)

        # Test on real images
        print("📊 Testing on REAL test images...")
        real_results = self._test_on_real_images(real_test_images)

        # Compare results
        self._compare_performance(training_results, real_results)

        return training_results, real_results

    def _test_on_training_data(self, num_samples: int = 10) -> Dict:
        """Test model on original training data"""

        rgb_dir = self.dataset_path / "rgb"
        annotations_dir = self.dataset_path / "annotations"

        if not rgb_dir.exists() or not annotations_dir.exists():
            print(f"❌ Training data not found at {self.dataset_path}")
            return {"error": "Training data not found"}

        # Get sample files
        rgb_files = list(rgb_dir.glob("*.jpeg"))[:num_samples]

        results = {
            "predictions": [],
            "ground_truth": [],
            "errors": [],
            "images": [],
            "mean_error": 0,
            "max_error": 0
        }

        print(f"Testing {len(rgb_files)} training samples...")

        for rgb_file in rgb_files:
            try:
                # Load image
                image = cv2.imread(str(rgb_file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Load ground truth annotation
                sample_id = rgb_file.stem.split('_')[1]
                annotation_file = annotations_dir / f"annotation_{sample_id}.json"

                if not annotation_file.exists():
                    continue

                with open(annotation_file, 'r') as f:
                    annotation = json.load(f)

                if 'board_corners' not in annotation:
                    continue

                # Extract ground truth corners
                board_corners = annotation['board_corners']
                gt_corners = np.array([
                    board_corners['white_left'],
                    board_corners['white_right'],
                    board_corners['black_right'],
                    board_corners['black_left']
                ])

                # Predict corners
                pred_corners = self._predict_corners_with_preprocessing(image)

                # Calculate error
                error = np.mean(np.sqrt(np.sum((pred_corners - gt_corners) ** 2, axis=1)))

                results["predictions"].append(pred_corners)
                results["ground_truth"].append(gt_corners)
                results["errors"].append(error)
                results["images"].append(str(rgb_file))

                print(f"  {rgb_file.name}: Error = {error:.1f} pixels")

            except Exception as e:
                print(f"  ❌ Error processing {rgb_file.name}: {e}")

        if results["errors"]:
            results["mean_error"] = np.mean(results["errors"])
            results["max_error"] = np.max(results["errors"])

        return results

    def _test_on_real_images(self, image_paths: List[str]) -> Dict:
        """Test model on real images (no ground truth available)"""

        results = {
            "predictions": [],
            "images": [],
            "confidence_scores": [],
            "image_characteristics": []
        }

        print(f"Testing {len(image_paths)} real images...")

        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  ❌ Could not load {image_path}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Predict corners
                pred_corners = self._predict_corners_with_preprocessing(image)

                # Analyze image characteristics
                characteristics = self._analyze_image_characteristics(image)

                # Calculate prediction confidence (heuristic)
                confidence = self._estimate_prediction_confidence(image, pred_corners)

                results["predictions"].append(pred_corners)
                results["images"].append(image_path)
                results["confidence_scores"].append(confidence)
                results["image_characteristics"].append(characteristics)

                print(f"  {Path(image_path).name}: Confidence = {confidence:.3f}")

            except Exception as e:
                print(f"  ❌ Error processing {image_path}: {e}")

        return results

    def _predict_corners_with_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Predict corners with full preprocessing pipeline"""

        original_shape = image.shape[:2]

        # Resize to model input size
        processed = cv2.resize(image, self.input_size)
        processed = processed.astype(np.float32) / 255.0
        processed = np.expand_dims(processed, axis=0)

        # Predict normalized coordinates
        normalized_coords = self.model.predict(processed, verbose=0)[0]

        # Convert back to original coordinates
        h, w = original_shape
        corners = []

        for i in range(0, 8, 2):
            x = normalized_coords[i] * w
            y = normalized_coords[i + 1] * h
            corners.append([x, y])

        return np.array(corners)

    def _analyze_image_characteristics(self, image: np.ndarray) -> Dict:
        """Analyze characteristics of an image"""

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        characteristics = {
            "brightness": np.mean(gray),
            "contrast": np.std(gray),
            "resolution": image.shape[:2],
            "aspect_ratio": image.shape[1] / image.shape[0],
            "edge_density": self._calculate_edge_density(gray),
            "color_variance": np.var(image.reshape(-1, 3), axis=0).mean()
        }

        return characteristics

    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate edge density in image"""
        edges = cv2.Canny(gray_image, 50, 150)
        return np.sum(edges > 0) / edges.size

    def _estimate_prediction_confidence(self,
                                        image: np.ndarray,
                                        corners: np.ndarray) -> float:
        """Estimate confidence in corner predictions using heuristics"""

        confidence = 1.0
        h, w = image.shape[:2]

        # Check if corners are within image bounds
        for corner in corners:
            if corner[0] < 0 or corner[0] >= w or corner[1] < 0 or corner[1] >= h:
                confidence *= 0.1  # Severely penalize out-of-bounds

        # Check if corners form reasonable quadrilateral
        area = cv2.contourArea(corners.astype(np.int32))
        image_area = h * w
        area_ratio = area / image_area

        if 0.1 <= area_ratio <= 0.8:
            confidence *= 1.0  # Good area
        elif 0.05 <= area_ratio < 0.1 or 0.8 < area_ratio <= 0.95:
            confidence *= 0.7  # Acceptable area
        else:
            confidence *= 0.3  # Poor area

        # Check aspect ratio of detected board
        if corners.shape[0] == 4:
            rect = cv2.minAreaRect(corners.astype(np.float32))
            width, height = rect[1]
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 10

            if aspect_ratio <= 1.5:
                confidence *= 1.0  # Square-ish
            elif aspect_ratio <= 2.0:
                confidence *= 0.8
            else:
                confidence *= 0.5  # Too rectangular

        return min(confidence, 1.0)

    def _compare_performance(self, training_results: Dict, real_results: Dict):
        """Compare performance between training and real data"""

        print(f"\n📊 PERFORMANCE COMPARISON")
        print("=" * 40)

        # Training data performance
        if "mean_error" in training_results and training_results["mean_error"] > 0:
            print(f"🎯 TRAINING DATA:")
            print(f"  Mean pixel error: {training_results['mean_error']:.1f}")
            print(f"  Max pixel error:  {training_results['max_error']:.1f}")
            print(f"  Samples tested:   {len(training_results['errors'])}")

        # Real data performance (confidence-based)
        if real_results["confidence_scores"]:
            mean_confidence = np.mean(real_results["confidence_scores"])
            min_confidence = np.min(real_results["confidence_scores"])

            print(f"\n🌍 REAL DATA:")
            print(f"  Mean confidence:  {mean_confidence:.3f}")
            print(f"  Min confidence:   {min_confidence:.3f}")
            print(f"  Samples tested:   {len(real_results['confidence_scores'])}")

            # Analyze confidence distribution
            high_conf = sum(1 for c in real_results["confidence_scores"] if c > 0.7)
            med_conf = sum(1 for c in real_results["confidence_scores"] if 0.3 < c <= 0.7)
            low_conf = sum(1 for c in real_results["confidence_scores"] if c <= 0.3)

            print(f"\n📈 CONFIDENCE DISTRIBUTION:")
            print(f"  High (>0.7):      {high_conf}/{len(real_results['confidence_scores'])}")
            print(f"  Medium (0.3-0.7): {med_conf}/{len(real_results['confidence_scores'])}")
            print(f"  Low (≤0.3):       {low_conf}/{len(real_results['confidence_scores'])}")

        # Analysis and recommendations
        print(f"\n🔍 ANALYSIS:")

        if "mean_error" in training_results and training_results["mean_error"] < 20:
            print("✅ Model performs well on training data")
        else:
            print("❌ Model performs poorly even on training data!")

        if real_results["confidence_scores"] and np.mean(real_results["confidence_scores"]) < 0.5:
            print("❌ Model struggles with real-world images")
            print("\n💡 LIKELY ISSUES:")
            print("  • Training data too different from real images")
            print("  • Overfitting to synthetic data characteristics")
            print("  • Coordinate system/normalization mismatch")
            print("  • Preprocessing differences")

    def visualize_failure_cases(self,
                                real_test_images: List[str],
                                max_examples: int = 6):
        """Visualize worst-performing cases for analysis"""

        print(f"\n🔍 ANALYZING FAILURE CASES")
        print("=" * 40)

        # Test images and get confidence scores
        results = []
        for image_path in real_test_images[:max_examples * 2]:  # Test more than we'll show
            try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                corners = self._predict_corners_with_preprocessing(image)
                confidence = self._estimate_prediction_confidence(image, corners)

                results.append({
                    'path': image_path,
                    'image': image,
                    'corners': corners,
                    'confidence': confidence
                })

            except Exception as e:
                print(f"Error with {image_path}: {e}")

        # Sort by confidence (worst first)
        results.sort(key=lambda x: x['confidence'])

        # Visualize worst cases
        num_examples = min(max_examples, len(results))

        fig, axes = plt.subplots(2, num_examples // 2 if num_examples >= 2 else 1,
                                 figsize=(20, 10))
        if num_examples == 1:
            axes = [axes]
        elif num_examples <= 2:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()

        for i in range(num_examples):
            result = results[i]

            # Create visualization
            vis_image = result['image'].copy()
            corners = result['corners'].astype(int)

            # Draw predicted corners
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            for j, (corner, color, label) in enumerate(zip(corners, colors, self.corner_labels)):
                cv2.circle(vis_image, tuple(corner), 15, color, -1)
                cv2.putText(vis_image, label[:2], (corner[0] + 20, corner[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw board outline
            cv2.polylines(vis_image, [corners], True, (0, 255, 0), 3)

            # Show in subplot
            axes[i].imshow(vis_image)
            axes[i].set_title(f"Conf: {result['confidence']:.3f}\n{Path(result['path']).name}",
                              fontsize=10)
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(num_examples, len(axes)):
            axes[i].axis('off')

        plt.suptitle("Worst Performing Cases (Lowest Confidence)", fontsize=16)
        plt.tight_layout()
        plt.show()

        return results[:num_examples]

    def analyze_coordinate_predictions(self,
                                       real_test_images: List[str],
                                       training_samples: int = 10):
        """Analyze coordinate prediction patterns"""

        print(f"\n📊 COORDINATE PREDICTION ANALYSIS")
        print("=" * 45)

        # Get predictions from both datasets
        training_results = self._test_on_training_data(training_samples)
        real_results = self._test_on_real_images(real_test_images)

        # Analyze coordinate distributions
        if training_results.get("predictions") and real_results.get("predictions"):
            # Convert to arrays for analysis
            train_coords = np.array(training_results["predictions"])  # Shape: (n_samples, 4, 2)
            real_coords = np.array(real_results["predictions"])

            # Analyze coordinate ranges (normalized to [0,1])
            print(f"🎯 TRAINING DATA COORDINATE RANGES:")
            self._analyze_coordinate_ranges(train_coords, "Training")

            print(f"\n🌍 REAL DATA COORDINATE RANGES:")
            self._analyze_coordinate_ranges(real_coords, "Real")

            # Visualize coordinate distributions
            self._plot_coordinate_distributions(train_coords, real_coords)

    def _analyze_coordinate_ranges(self, coords: np.ndarray, dataset_name: str):
        """Analyze coordinate ranges for a dataset"""

        for i, label in enumerate(self.corner_labels):
            x_coords = coords[:, i, 0]
            y_coords = coords[:, i, 1]

            print(f"  {label:12} X: [{np.min(x_coords):6.1f}, {np.max(x_coords):6.1f}] "
                  f"Y: [{np.min(y_coords):6.1f}, {np.max(y_coords):6.1f}]")

    def _plot_coordinate_distributions(self, train_coords: np.ndarray, real_coords: np.ndarray):
        """Plot coordinate distribution comparison"""

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        for i, label in enumerate(self.corner_labels):
            # X coordinates
            ax_x = axes[0, i]
            if len(train_coords) > 0:
                ax_x.hist(train_coords[:, i, 0], alpha=0.7, label='Training', bins=15)
            if len(real_coords) > 0:
                ax_x.hist(real_coords[:, i, 0], alpha=0.7, label='Real', bins=15)
            ax_x.set_title(f'{label} - X coordinates')
            ax_x.legend()
            ax_x.grid(True, alpha=0.3)

            # Y coordinates
            ax_y = axes[1, i]
            if len(train_coords) > 0:
                ax_y.hist(train_coords[:, i, 1], alpha=0.7, label='Training', bins=15)
            if len(real_coords) > 0:
                ax_y.hist(real_coords[:, i, 1], alpha=0.7, label='Real', bins=15)
            ax_y.set_title(f'{label} - Y coordinates')
            ax_y.legend()
            ax_y.grid(True, alpha=0.3)

        plt.suptitle('Coordinate Distributions: Training vs Real Data', fontsize=16)
        plt.tight_layout()
        plt.show()

    def run_comprehensive_diagnosis(self, real_test_images: List[str]):
        """Run complete diagnostic suite"""

        print("🔬 COMPREHENSIVE CORNER DETECTOR DIAGNOSIS")
        print("=" * 60)

        # 1. Training vs Real comparison
        training_results, real_results = self.diagnose_training_vs_real_data(real_test_images)

        # 2. Coordinate analysis
        self.analyze_coordinate_predictions(real_test_images)

        # 3. Failure case visualization
        failure_cases = self.visualize_failure_cases(real_test_images)

        # 4. Generate recommendations
        self._generate_recommendations(training_results, real_results, failure_cases)

        return {
            "training_results": training_results,
            "real_results": real_results,
            "failure_cases": failure_cases
        }

    def _generate_recommendations(self,
                                  training_results: Dict,
                                  real_results: Dict,
                                  failure_cases: List):
        """Generate specific recommendations based on diagnosis"""

        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 30)

        # Check training performance
        if "mean_error" in training_results:
            if training_results["mean_error"] > 50:
                print("🔧 HIGH PRIORITY - Model fails on training data:")
                print("   • Check model architecture")
                print("   • Verify loss function and metrics")
                print("   • Increase training epochs")
                print("   • Check for data loading bugs")
                return

        # Check real-world performance
        if real_results.get("confidence_scores"):
            mean_conf = np.mean(real_results["confidence_scores"])

            if mean_conf < 0.3:
                print("🚨 CRITICAL - Very poor real-world performance:")
                print("   • DOMAIN MISMATCH: Training data too different from real images")
                print("   • Collect real chess board images for training")
                print("   • Apply data augmentation to bridge the gap")
                print("   • Consider fine-tuning on real data")

            elif mean_conf < 0.6:
                print("⚠️  MODERATE - Poor real-world performance:")
                print("   • Add data augmentation (lighting, perspective, noise)")
                print("   • Mix synthetic and real training data")
                print("   • Check preprocessing pipeline consistency")

        # Specific technical recommendations
        print(f"\n🔧 TECHNICAL RECOMMENDATIONS:")
        print("   1. Data Collection:")
        print("      • Photograph real chess boards in various conditions")
        print("      • Include different lighting, angles, board styles")
        print("      • Manual annotation of corner coordinates")

        print("   2. Model Improvements:")
        print("      • Add stronger data augmentation")
        print("      • Try different loss functions (Huber, smooth L1)")
        print("      • Implement progressive training (synthetic → real)")

        print("   3. Architecture Changes:")
        print("      • Try different backbones (EfficientNet, Vision Transformer)")
        print("      • Add attention mechanisms for corner localization")
        print("      • Multi-scale feature extraction")

        print("   4. Training Strategy:")
        print("      • Domain adaptation techniques")
        print("      • Self-supervised pre-training on unlabeled real images")
        print("      • Active learning to select most useful training samples")


# Convenience function for quick diagnosis
def quick_diagnosis(model_path: str,
                    real_test_images: List[str],
                    dataset_path: str = "c:/datasets/ChessRender360"):
    """
    Quick diagnosis function

    Args:
        model_path: Path to trained model
        real_test_images: List of real chess board image paths
        dataset_path: Path to training dataset
    """

    diagnostics = CornerDetectorDiagnostics(model_path, dataset_path)
    results = diagnostics.run_comprehensive_diagnosis(real_test_images)

    return results


if __name__ == "__main__":
    print("🔬 CORNER DETECTOR DIAGNOSTICS")
    print("=" * 40)

    # Example usage
    model_path = "outputs/corner_detector/fine_tuned_chess_corner_detector.keras"

    # Get real test images from user
    test_images = []
    print("Enter paths to real chess board images (one per line, empty line to finish):")
    while True:
        path = input().strip()
        if not path:
            break
        if Path(path).exists():
            test_images.append(path)
        else:
            print(f"❌ File not found: {path}")

    if test_images:
        print(f"\n🧪 Running diagnosis on {len(test_images)} test images...")
        results = quick_diagnosis(model_path, test_images)
        print("✅ Diagnosis complete!")
    else:
        print("❌ No valid test images provided")