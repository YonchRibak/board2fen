
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import json
from chess_board_detector import ChessBoardDetector, BoardDetection


class BoardDetectionTester:
    """Utility class for testing board detection on multiple images"""
    def __init__(self, debug: bool = True):
        self.detector = ChessBoardDetector(debug=debug)
        self.results = []

    def test_single_image(self, image_path: str) -> Optional[BoardDetection]:
        """Test detection on a single image"""

        print(f"🧪 Testing: {Path(image_path).name}")

        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return None

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect board
        detection = self.detector.detect_board(image)

        if detection:
            print(f"✅ Success: {detection.method} (conf: {detection.confidence:.3f})")
        else:
            print("❌ Detection failed")

        return detection

    def test_directory(self, image_dir: str, extensions: List[str] = None) -> Dict:
        """Test detection on all images in a directory"""

        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"❌ Directory not found: {image_dir}")
            return {}

        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
            image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))

        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return {}

        print(f"🧪 Testing {len(image_files)} images in {image_dir}")

        results = {
            'successful': [],
            'failed': [],
            'methods_used': {},
            'confidence_scores': []
        }

        for image_file in image_files:
            detection = self.test_single_image(str(image_file))

            if detection:
                results['successful'].append(str(image_file))
                results['confidence_scores'].append(detection.confidence)

                method = detection.method
                results['methods_used'][method] = results['methods_used'].get(method, 0) + 1
            else:
                results['failed'].append(str(image_file))

        # Print summary
        total = len(image_files)
        success_rate = len(results['successful']) / total * 100

        print(f"\n📊 DETECTION SUMMARY:")
        print(f"Total images: {total}")
        print(f"Successful: {len(results['successful'])} ({success_rate:.1f}%)")
        print(f"Failed: {len(results['failed'])} ({100 - success_rate:.1f}%)")

        if results['confidence_scores']:
            avg_confidence = np.mean(results['confidence_scores'])
            print(f"Average confidence: {avg_confidence:.3f}")

        print(f"Methods used:")
        for method, count in results['methods_used'].items():
            print(f"  {method}: {count} times")

        return results

    def create_test_report(self, results: Dict, output_file: str = "detection_report.json"):
        """Save test results to a JSON report"""

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📄 Test report saved to: {output_file}")


def create_synthetic_test_image(size: Tuple[int, int] = (800, 600)) -> np.ndarray:
    """Create a synthetic chess board image for testing"""

    width, height = size
    image = np.ones((height, width, 3), dtype=np.uint8) * 200  # Gray background

    # Create checkerboard pattern
    board_size = min(width, height) // 2
    square_size = board_size // 8

    # Center the board
    start_x = (width - board_size) // 2
    start_y = (height - board_size) // 2

    for row in range(8):
        for col in range(8):
            x = start_x + col * square_size
            y = start_y + row * square_size

            # Alternate colors
            if (row + col) % 2 == 0:
                color = (240, 217, 181)  # Light square
            else:
                color = (181, 136, 99)  # Dark square

            cv2.rectangle(image, (x, y), (x + square_size, y + square_size), color, -1)

    # Add border
    board_rect = (start_x - 10, start_y - 10, board_size + 20, board_size + 20)
    cv2.rectangle(image, (board_rect[0], board_rect[1]),
                  (board_rect[0] + board_rect[2], board_rect[1] + board_rect[3]),
                  (139, 69, 19), 10)  # Brown border

    return image


def demo_board_detection():
    """Demo the board detection system"""

    print("🎯 Chess Board Detection Demo")
    print("=" * 50)

    # Create synthetic test image
    print("🎨 Creating synthetic test image...")
    test_image = create_synthetic_test_image()

    # Initialize detector
    detector = ChessBoardDetector(debug=True)

    # Test detection
    print("\n🔍 Testing board detection...")
    detection = detector.detect_board(test_image)

    if detection:
        print("✅ Detection successful!")

        # Display results
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(test_image)
        plt.title("Original Image")
        plt.axis('off')

        # Detection visualization
        plt.subplot(1, 3, 2)
        img_with_detection = test_image.copy()
        corners = detection.corners.astype(int)
        cv2.polylines(img_with_detection, [corners], True, (255, 0, 0), 3)
        for i, corner in enumerate(corners):
            cv2.circle(img_with_detection, tuple(corner), 8, (0, 255, 0), -1)
        plt.imshow(img_with_detection)
        plt.title(f"Detection: {detection.method}\nConf: {detection.confidence:.3f}")
        plt.axis('off')

        # Warped result (preview)
        plt.subplot(1, 3, 3)
        # Create perspective transformation
        board_size = 400
        dst_corners = np.array([
            [0, 0], [board_size, 0],
            [board_size, board_size], [0, board_size]
        ], dtype=np.float32)

        transform_matrix = cv2.getPerspectiveTransform(detection.corners, dst_corners)
        warped = cv2.warpPerspective(test_image, transform_matrix, (board_size, board_size))
        plt.imshow(warped)
        plt.title("Warped Board")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    else:
        print("❌ Detection failed")


def usage_example():
    """Show how to use the board detector in practice"""

    print("📚 USAGE EXAMPLE")
    print("=" * 30)

    # Example 1: Basic usage
    print("\n1. Basic Usage:")
    print("```python")
    print("from chess_board_detector import ChessBoardDetector")
    print()
    print("# Initialize detector")
    print("detector = ChessBoardDetector(debug=False)")
    print()
    print("# Load image")
    print("image = cv2.imread('chess_board.jpg')")
    print("image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)")
    print()
    print("# Detect board")
    print("detection = detector.detect_board(image)")
    print()
    print("if detection:")
    print("    print(f'Board detected: {detection.method}')")
    print("    print(f'Confidence: {detection.confidence:.3f}')")
    print("    print(f'Corners: {detection.corners}')")
    print("else:")
    print("    print('Board not detected')")
    print("```")

    # Example 2: Integration with pipeline
    print("\n2. Pipeline Integration:")
    print("```python")
    print("def process_chess_image(image_path):")
    print("    detector = ChessBoardDetector()")
    print("    image = cv2.imread(image_path)")
    print("    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)")
    print("    ")
    print("    detection = detector.detect_board(image)")
    print("    ")
    print("    if detection and detection.confidence > 0.5:")
    print("        # Use detected corners for further processing")
    print("        corners = detection.corners")
    print("        # ... continue with piece detection ...")
    print("        return True")
    print("    else:")
    print("        print('Could not detect chess board reliably')")
    print("        return False")
    print("```")

    # Example 3: Batch processing
    print("\n3. Batch Processing:")
    print("```python")
    print("from board_detection_utils import BoardDetectionTester")
    print()
    print("tester = BoardDetectionTester(debug=False)")
    print("results = tester.test_directory('chess_images/')")
    print("tester.create_test_report(results)")
    print("```")


if __name__ == "__main__":
    choice = input(
        "Choose demo option:\n1. Synthetic image demo\n2. Usage examples\n3. Test directory\nChoice (1/2/3): ")

    if choice == "1":
        demo_board_detection()
    elif choice == "2":
        usage_example()
    elif choice == "3":
        test_dir = input("Enter directory path with chess images: ")
        tester = BoardDetectionTester()
        results = tester.test_directory(test_dir)
        tester.create_test_report(results)
    else:
        print("Running synthetic demo...")
        demo_board_detection()