# double_yolo/corner_detection/main.py
# YOLOv8 Corner Detection for Chess Boards using ChessRender360 dataset

import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import yaml
from tqdm import tqdm
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Fixed import


class ChessCornerDatasetProcessor:
    """Process ChessRender360 dataset for YOLOv8 corner detection training"""

    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.rgb_dir = self.dataset_path / "rgb"
        self.annotations_dir = self.dataset_path / "annotations"

        # Corner class mapping for YOLO
        self.corner_classes = {
            'white_left': 0,
            'white_right': 1,
            'black_right': 2,
            'black_left': 3
        }

        print(f"📂 Dataset: {self.dataset_path}")
        print(f"🎯 Output: {self.output_path}")

    def _dataset_exists(self, dataset_dir: Path) -> bool:
        """Check if YOLO dataset already exists with reasonable amount of data"""
        try:
            # Check if main directories exist
            train_images = dataset_dir / 'train' / 'images'
            train_labels = dataset_dir / 'train' / 'labels'
            val_images = dataset_dir / 'val' / 'images'
            val_labels = dataset_dir / 'val' / 'labels'
            yaml_file = dataset_dir / 'data.yaml'

            if not all([train_images.exists(), train_labels.exists(),
                        val_images.exists(), val_labels.exists(), yaml_file.exists()]):
                return False

            # Check if directories contain files
            train_img_count = len(list(train_images.glob('*')))
            train_label_count = len(list(train_labels.glob('*.txt')))
            val_img_count = len(list(val_images.glob('*')))
            val_label_count = len(list(val_labels.glob('*.txt')))

            # Verify we have matching numbers of images and labels
            if (train_img_count == 0 or train_label_count == 0 or
                    val_img_count == 0 or val_label_count == 0):
                return False

            if (train_img_count != train_label_count or
                    val_img_count != val_label_count):
                print(f"⚠️ Mismatch in image/label counts:")
                print(f"   Train: {train_img_count} images, {train_label_count} labels")
                print(f"   Val: {val_img_count} images, {val_label_count} labels")
                return False

            print(f"📊 Found existing dataset:")
            print(f"   Train: {train_img_count} samples")
            print(f"   Val: {val_img_count} samples")

            return True

        except Exception as e:
            print(f"❌ Error checking dataset: {e}")
            return False

    def validate_dataset(self) -> bool:
        """Validate dataset structure"""
        if not self.dataset_path.exists():
            print(f"❌ Dataset path not found: {self.dataset_path}")
            return False

        if not self.rgb_dir.exists():
            print(f"❌ RGB directory not found: {self.rgb_dir}")
            return False

        if not self.annotations_dir.exists():
            print(f"❌ Annotations directory not found: {self.annotations_dir}")
            return False

        print("✅ Dataset structure validated")
        return True

    def find_image_annotation_pairs(self, max_samples: Optional[int] = None) -> List[Tuple[Path, Path]]:
        """Find matching image-annotation pairs"""
        print("🔍 Finding image-annotation pairs...")

        # Get all RGB files
        rgb_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            rgb_files.extend(list(self.rgb_dir.glob(ext)))

        pairs = []
        for rgb_file in tqdm(rgb_files, desc="Processing images"):
            # Extract sample ID from filename (e.g., rgb_1234.jpeg -> 1234)
            if '_' in rgb_file.stem:
                sample_id = rgb_file.stem.split('_')[1]
            else:
                sample_id = rgb_file.stem

            # Find corresponding annotation
            annotation_file = self.annotations_dir / f"annotation_{sample_id}.json"

            if annotation_file.exists():
                pairs.append((rgb_file, annotation_file))

            if max_samples and len(pairs) >= max_samples:
                break

        print(f"✅ Found {len(pairs)} valid image-annotation pairs")
        return pairs

    def convert_corners_to_yolo_format(self, corners: Dict, img_width: int, img_height: int) -> List[str]:
        """Convert corner coordinates to YOLO format"""
        yolo_annotations = []

        for corner_name, coords in corners.items():
            if corner_name in self.corner_classes:
                class_id = self.corner_classes[corner_name]
                x, y = coords

                # Normalize coordinates
                x_norm = x / img_width
                y_norm = y / img_height

                # Create small bounding box around corner point (5x5 pixels normalized)
                box_size = 5.0
                w_norm = box_size / img_width
                h_norm = box_size / img_height

                # YOLO format: class_id x_center y_center width height
                yolo_line = f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                yolo_annotations.append(yolo_line)

        return yolo_annotations

    def create_yolo_dataset(self, pairs: List[Tuple[Path, Path]], train_split: float = 0.8,
                            force_recreate: bool = False):
        """Create YOLO dataset structure"""
        dataset_dir = self.output_path / "yolo_dataset"
        yaml_path = dataset_dir / 'data.yaml'

        # Check if dataset already exists
        if not force_recreate and self._dataset_exists(dataset_dir):
            print(f"✅ YOLO dataset already exists at: {dataset_dir}")
            print(f"📄 Using existing configuration: {yaml_path}")
            return dataset_dir, yaml_path

        print(f"🏗️ Creating YOLO dataset structure...")

        # Create directory structure
        for split in ['train', 'val']:
            (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Split data
        np.random.shuffle(pairs)
        split_idx = int(len(pairs) * train_split)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        print(f"📊 Train samples: {len(train_pairs)}")
        print(f"📊 Validation samples: {len(val_pairs)}")

        # Process each split
        for split_name, split_pairs in [('train', train_pairs), ('val', val_pairs)]:
            print(f"\n🔄 Processing {split_name} split...")

            for img_path, ann_path in tqdm(split_pairs, desc=f"Processing {split_name}"):
                try:
                    # Load image to get dimensions
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    img_height, img_width = img.shape[:2]

                    # Load annotation
                    with open(ann_path, 'r') as f:
                        annotation = json.load(f)

                    if 'board_corners' not in annotation:
                        continue

                    corners = annotation['board_corners']

                    # Convert to YOLO format
                    yolo_annotations = self.convert_corners_to_yolo_format(
                        corners, img_width, img_height
                    )

                    if len(yolo_annotations) != 4:
                        print(f"⚠️ Skipping {img_path.name}: Expected 4 corners, got {len(yolo_annotations)}")
                        continue

                    # Copy image
                    dst_img = dataset_dir / split_name / 'images' / img_path.name
                    shutil.copy2(img_path, dst_img)

                    # Save label
                    label_name = img_path.stem + '.txt'
                    dst_label = dataset_dir / split_name / 'labels' / label_name

                    with open(dst_label, 'w') as f:
                        f.write('\n'.join(yolo_annotations))

                except Exception as e:
                    print(f"❌ Error processing {img_path.name}: {e}")
                    continue

        # Create data.yaml
        data_yaml = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 4,
            'names': ['white_left', 'white_right', 'black_right', 'black_left']
        }

        yaml_path = dataset_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"✅ Dataset created at: {dataset_dir}")
        print(f"📄 Configuration saved: {yaml_path}")

        return dataset_dir, yaml_path

    def _color_name_to_bgr(self, color_name: str) -> Tuple[int, int, int]:
        """Convert color name to BGR tuple for OpenCV"""
        # Convert matplotlib color name to RGB (0-1), then to BGR (0-255)
        rgb = mcolors.to_rgb(color_name)
        # Convert to BGR and scale to 0-255
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        return bgr

    def visualize_annotations(self, dataset_dir: Path, num_samples: int = 5):
        """Visualize sample annotations to verify correctness"""
        print(f"🎨 Visualizing {num_samples} sample annotations...")

        train_images = list((dataset_dir / 'train' / 'images').glob('*.jpg'))
        train_images.extend(list((dataset_dir / 'train' / 'images').glob('*.jpeg')))

        if len(train_images) == 0:
            print("❌ No training images found for visualization")
            return

        # Select random samples
        samples = np.random.choice(train_images, min(num_samples, len(train_images)), replace=False)

        fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
        if len(samples) == 1:
            axes = [axes]

        colors = ['red', 'green', 'blue', 'yellow']
        corner_names = ['white_left', 'white_right', 'black_right', 'black_left']

        for idx, img_path in enumerate(samples):
            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # Load corresponding label
            label_path = dataset_dir / 'train' / 'labels' / (img_path.stem + '.txt')

            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                # Draw corners
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)

                        # Convert back to pixel coordinates
                        x = int(x_center * w)
                        y = int(y_center * h)

                        # Draw corner point - Fixed color conversion
                        color_name = colors[int(class_id)]
                        color_bgr = self._color_name_to_bgr(color_name)
                        # Convert BGR to RGB for matplotlib display
                        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

                        cv2.circle(img, (x, y), 8, color_rgb, -1)

                        # Add label
                        cv2.putText(img, corner_names[int(class_id)], (x + 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)

            axes[idx].imshow(img)
            axes[idx].set_title(f"Sample {idx + 1}")
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_path / 'corner_annotations_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"💾 Visualization saved: {self.output_path / 'corner_annotations_visualization.png'}")


class CornerDetectionTrainer:
    """YOLOv8 trainer for corner detection"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _color_name_to_bgr(self, color_name: str) -> Tuple[int, int, int]:
        """Convert color name to BGR tuple for OpenCV"""
        # Convert matplotlib color name to RGB (0-1), then to BGR (0-255)
        rgb = mcolors.to_rgb(color_name)
        # Convert to BGR and scale to 0-255
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        return bgr

    def train_corner_detector(self, data_yaml_path: str,
                              model_size: str = 'n',
                              epochs: int = 100,
                              imgsz: int = 640,
                              batch_size: int = 16,
                              device: str = 'cpu'):
        """Train YOLOv8 corner detection model"""

        print(f"🚀 Starting YOLOv8 corner detection training...")
        print(f"📋 Configuration:")
        print(f"   Model size: YOLOv8{model_size}")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {device}")

        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')

        # Train model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=str(self.output_dir),
            name='corner_detection',
            save_period=10,  # Save checkpoint every 10 epochs
            patience=20,  # Early stopping patience
            verbose=True
        )

        print(f"✅ Training completed!")
        print(f"📁 Results saved to: {self.output_dir}")

        return results

    def evaluate_model(self, model_path: str, data_yaml_path: str):
        """Evaluate trained corner detection model"""
        print(f"📊 Evaluating corner detection model...")

        # Load trained model
        model = YOLO(model_path)

        # Run validation
        results = model.val(data=data_yaml_path)

        print(f"📈 Evaluation Results:")
        print(f"   mAP@0.5: {results.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {results.box.map:.4f}")

        return results

    def test_on_sample_images(self, model_path: str, test_images_dir: str, num_samples: int = 5):
        """Test corner detection on sample images"""
        print(f"🧪 Testing corner detection on sample images...")

        # Load model
        model = YOLO(model_path)

        # Get test images
        test_dir = Path(test_images_dir)
        if not test_dir.exists():
            print(f"❌ Test directory not found: {test_dir}")
            return

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(test_dir.glob(ext)))

        if len(image_files) == 0:
            print(f"❌ No images found in: {test_dir}")
            return

        # Select random samples
        samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)

        fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 5))
        if len(samples) == 1:
            axes = [axes]

        corner_names = ['white_left', 'white_right', 'black_right', 'black_left']
        colors = ['red', 'green', 'blue', 'yellow']

        for idx, img_path in enumerate(samples):
            # Load and predict
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = model(img_path)

            # Draw predictions
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        # Draw center point
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        # Fixed color conversion
                        color_name = colors[cls]
                        color_bgr = self._color_name_to_bgr(color_name)
                        # Convert BGR to RGB for matplotlib display
                        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

                        cv2.circle(img_rgb, (center_x, center_y), 10, color_rgb, -1)

                        # Add label
                        label = f"{corner_names[cls]} ({conf:.2f})"
                        cv2.putText(img_rgb, label, (center_x + 15, center_y - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rgb, 2)

            axes[idx].imshow(img_rgb)
            axes[idx].set_title(f"Test Sample {idx + 1}")
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'corner_detection_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"💾 Test results saved: {self.output_dir / 'corner_detection_test_results.png'}")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='YOLOv8 Corner Detection Training')
    parser.add_argument('--dataset', type=str, default='C:/datasets/ChessRender360',
                        help='Path to ChessRender360 dataset')
    parser.add_argument('--output', type=str, default='outputs/corner_detection',
                        help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use (None for all)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training (cpu or cuda)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only prepare dataset')
    parser.add_argument('--force_recreate', action='store_true',
                        help='Force recreation of YOLO dataset even if it already exists')

    args = parser.parse_args()

    print("🏁 YOLOv8 CORNER DETECTION TRAINING")
    print("=" * 50)

    # Initialize processor
    processor = ChessCornerDatasetProcessor(args.dataset, args.output)

    # Validate dataset
    if not processor.validate_dataset():
        return

    # Find image-annotation pairs
    pairs = processor.find_image_annotation_pairs(args.max_samples)

    if len(pairs) == 0:
        print("❌ No valid image-annotation pairs found!")
        return

    # Create YOLO dataset
    dataset_dir, yaml_path = processor.create_yolo_dataset(pairs, force_recreate=args.force_recreate)

    # Visualize annotations
    processor.visualize_annotations(dataset_dir)

    if args.skip_training:
        print("✅ Dataset preparation completed (training skipped)")
        return

    # Train model
    trainer = CornerDetectionTrainer(args.output)

    training_results = trainer.train_corner_detector(
        data_yaml_path=str(yaml_path),
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )

    # Find best model
    best_model_path = Path(args.output) / 'corner_detection' / 'weights' / 'best.pt'

    if best_model_path.exists():
        # Evaluate model
        trainer.evaluate_model(str(best_model_path), str(yaml_path))

        # Test on sample images
        test_images_dir = dataset_dir / 'val' / 'images'
        trainer.test_on_sample_images(str(best_model_path), str(test_images_dir))

        print(f"\n🎉 Corner Detection Training Complete!")
        print(f"📁 Best model: {best_model_path}")
        print(f"📊 Results: {Path(args.output) / 'corner_detection'}")
    else:
        print("❌ Training completed but best model not found!")


if __name__ == "__main__":
    main()