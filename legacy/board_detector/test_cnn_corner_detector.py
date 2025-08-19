# test_cnn_corner_detector.py - Test the trained CNN corner detection model

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import json


class CNNCornerDetectorTester:
    """Test trained CNN corner detector model"""

    def __init__(self, model_path: str):
        """
        Initialize tester with trained model

        Args:
            model_path: Path to saved keras model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.input_size = (224, 224)  # Model input size

        # Corner labels in prediction order
        self.corner_labels = [
            'white_left', 'white_right', 'black_right', 'black_left'
        ]

        # Corner colors for visualization
        self.corner_colors = {
            'white_left': (255, 0, 0),  # Red
            'white_right': (0, 255, 0),  # Green
            'black_right': (0, 0, 255),  # Blue
            'black_left': (255, 255, 0)  # Yellow
        }

        self.load_model()

    def load_model(self):
        """Load the trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"🔄 Loading model from: {self.model_path}")

        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            print("✅ Model loaded successfully!")

            # Print model info
            print(f"📊 Model input shape: {self.model.input_shape}")
            print(f"📊 Model output shape: {self.model.output_shape}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for model input

        Args:
            image: Input image (RGB)

        Returns:
            Preprocessed image and original shape
        """
        original_shape = image.shape[:2]  # (height, width)

        # Resize to model input size
        processed = cv2.resize(image, self.input_size)

        # Normalize to [0, 1]
        processed = processed.astype(np.float32) / 255.0

        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)

        return processed, original_shape

    def predict_corners(self, image: np.ndarray) -> np.ndarray:
        """
        Predict corner coordinates for an image

        Args:
            image: Input RGB image

        Returns:
            Corner coordinates as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Preprocess
        processed_image, original_shape = self.preprocess_image(image)

        # Predict normalized coordinates [0, 1]
        normalized_coords = self.model.predict(processed_image, verbose=0)[0]

        # Convert back to original image coordinates
        h, w = original_shape
        corners = []

        for i in range(0, 8, 2):  # Process pairs (x, y)
            x_norm = normalized_coords[i]
            y_norm = normalized_coords[i + 1]

            # Convert to original image coordinates
            x = int(x_norm * w)
            y = int(y_norm * h)

            corners.append([x, y])

        return np.array(corners)

    def test_single_image(self,
                          image_path: str,
                          show_plot: bool = True,
                          save_result: bool = False,
                          output_dir: str = "test_results") -> dict:
        """
        Test corner detection on a single image

        Args:
            image_path: Path to test image
            show_plot: Whether to display visualization
            save_result: Whether to save result image
            output_dir: Directory to save results

        Returns:
            Dictionary with test results
        """

        print(f"🧪 Testing image: {Path(image_path).name}")

        # Load image
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict corners
        try:
            predicted_corners = self.predict_corners(image_rgb)
            print("✅ Corner prediction successful!")

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return {"success": False, "error": str(e)}

        # Create visualization
        result_image = self.visualize_corners(image_rgb, predicted_corners)

        # Prepare results
        results = {
            "success": True,
            "image_path": str(image_path),
            "image_shape": image_rgb.shape,
            "predicted_corners": predicted_corners.tolist(),
            "corner_labels": self.corner_labels
        }

        # Print corner coordinates
        print(f"📍 Predicted corners:")
        for i, (corner, label) in enumerate(zip(predicted_corners, self.corner_labels)):
            print(f"  {label}: ({corner[0]}, {corner[1]})")

        # Save result if requested
        if save_result:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)

            result_filename = f"corner_test_{Path(image_path).stem}.jpg"
            result_path = output_path / result_filename

            # Convert RGB back to BGR for saving
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(result_path), result_bgr)

            # Save JSON with coordinates
            json_path = output_path / f"corner_test_{Path(image_path).stem}.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"💾 Results saved to: {result_path}")
            print(f"📄 Coordinates saved to: {json_path}")

            results["saved_path"] = str(result_path)

        # Show visualization
        if show_plot:
            self.show_result(image_rgb, result_image, predicted_corners)

        return results

    def visualize_corners(self,
                          image: np.ndarray,
                          corners: np.ndarray,
                          line_thickness: int = 3,
                          circle_radius: int = 12,
                          font_scale: float = 0.8) -> np.ndarray:
        """
        Create visualization with marked corners

        Args:
            image: Original RGB image
            corners: Predicted corner coordinates
            line_thickness: Thickness of board outline
            circle_radius: Radius of corner markers
            font_scale: Font size for labels

        Returns:
            Image with corners visualized
        """

        result_image = image.copy()

        # Draw board outline
        corners_int = corners.astype(int)
        cv2.polylines(result_image, [corners_int], True, (0, 255, 0), line_thickness)

        # Draw corners with labels
        for i, (corner, label) in enumerate(zip(corners_int, self.corner_labels)):
            color = self.corner_colors[label]

            # Draw corner circle
            cv2.circle(result_image, tuple(corner), circle_radius, color, -1)

            # Draw white border around circle for visibility
            cv2.circle(result_image, tuple(corner), circle_radius + 2, (255, 255, 255), 2)

            # Add label text
            text_pos = (corner[0] + 15, corner[1] - 15)

            # White background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            cv2.rectangle(result_image,
                          (text_pos[0] - 5, text_pos[1] - text_size[1] - 5),
                          (text_pos[0] + text_size[0] + 5, text_pos[1] + 5),
                          (255, 255, 255), -1)

            # Black text
            cv2.putText(result_image, label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

        # Add title
        title = "CNN Corner Detection Results"
        title_pos = (20, 40)
        cv2.rectangle(result_image,
                      (title_pos[0] - 10, title_pos[1] - 30),
                      (title_pos[0] + 400, title_pos[1] + 10),
                      (0, 0, 0), -1)  # Black background
        cv2.putText(result_image, title, title_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        return result_image

    def show_result(self,
                    original_image: np.ndarray,
                    result_image: np.ndarray,
                    corners: np.ndarray):
        """
        Display side-by-side comparison

        Args:
            original_image: Original RGB image
            result_image: Image with corners marked
            corners: Predicted corners
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Original image
        ax1.imshow(original_image)
        ax1.set_title('Original Image', fontsize=16)
        ax1.axis('off')

        # Result image
        ax2.imshow(result_image)
        ax2.set_title('Detected Corners', fontsize=16)
        ax2.axis('off')

        # Add corner coordinates as text
        corner_text = "Predicted Corners:\n"
        for corner, label in zip(corners, self.corner_labels):
            corner_text += f"{label}: ({corner[0]}, {corner[1]})\n"

        fig.text(0.02, 0.02, corner_text, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()
        plt.show()

    def batch_test(self,
                   image_dir: str,
                   extensions: list = None,
                   save_results: bool = True,
                   show_individual: bool = False) -> dict:
        """
        Test on multiple images in a directory

        Args:
            image_dir: Directory containing test images
            extensions: Image file extensions to look for
            save_results: Whether to save individual results
            show_individual: Whether to show each result individually

        Returns:
            Summary of batch test results
        """

        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
            image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))

        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return {}

        print(f"🧪 Testing {len(image_files)} images...")

        # Test each image
        results = {
            "total_images": len(image_files),
            "successful": 0,
            "failed": 0,
            "results": []
        }

        for i, image_file in enumerate(image_files):
            print(f"\n--- Testing {i + 1}/{len(image_files)}: {image_file.name} ---")

            try:
                result = self.test_single_image(
                    str(image_file),
                    show_plot=show_individual,
                    save_result=save_results,
                    output_dir="batch_test_results"
                )

                if result["success"]:
                    results["successful"] += 1
                else:
                    results["failed"] += 1

                results["results"].append(result)

            except Exception as e:
                print(f"❌ Error testing {image_file.name}: {e}")
                results["failed"] += 1
                results["results"].append({
                    "success": False,
                    "image_path": str(image_file),
                    "error": str(e)
                })

        # Print summary
        print(f"\n📊 BATCH TEST SUMMARY:")
        print(f"Total images: {results['total_images']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Success rate: {results['successful'] / results['total_images'] * 100:.1f}%")

        # Save batch results
        if save_results:
            batch_results_path = Path("batch_test_results") / "batch_summary.json"
            batch_results_path.parent.mkdir(exist_ok=True, parents=True)

            with open(batch_results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print(f"📄 Batch summary saved to: {batch_results_path}")

        return results


def main():
    """Main testing interface"""

    print("🎯 CNN CORNER DETECTOR TESTER")
    print("=" * 50)

    # Model path
    model_path = "outputs/corner_detector/fine_tuned_chess_corner_detector.keras"

    # Initialize tester
    try:
        tester = CNNCornerDetectorTester(model_path)
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        return

    # Interactive testing menu
    while True:
        print(f"\n🔧 TESTING OPTIONS:")
        print("1. Test single image")
        print("2. Test directory (batch)")
        print("3. Exit")

        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == "1":
            image_path = input("Enter image path: ").strip()

            try:
                result = tester.test_single_image(
                    image_path,
                    show_plot=True,
                    save_result=True
                )

                if result["success"]:
                    print("✅ Test completed successfully!")
                else:
                    print(f"❌ Test failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"❌ Error during testing: {e}")

        elif choice == "2":
            image_dir = input("Enter directory path: ").strip()
            show_individual = input("Show individual results? (y/N): ").strip().lower() == 'y'

            try:
                results = tester.batch_test(
                    image_dir,
                    save_results=True,
                    show_individual=show_individual
                )

                print("✅ Batch testing completed!")

            except Exception as e:
                print(f"❌ Error during batch testing: {e}")

        elif choice == "3":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please try again.")


# Quick test function for direct use
def quick_test(image_path: str):
    """Quick test function for direct use"""

    model_path = "board_detector/outputs/corner_detection/fine_tuned_chess_corner_detector.keras"

    try:
        tester = CNNCornerDetectorTester(model_path)
        result = tester.test_single_image(image_path, show_plot=True, save_result=True)
        return result
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return None


if __name__ == "__main__":
    main()