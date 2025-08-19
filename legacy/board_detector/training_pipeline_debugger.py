# training_pipeline_debugger.py - Debug your corner detection training pipeline

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def debug_training_pipeline(model_path: str, dataset_path: str = "c:/datasets/ChessRender360"):
    """
    Debug the entire training pipeline to find the root cause

    Based on your diagnostic results showing 48+ pixel errors on training data,
    there's likely a fundamental issue in the training setup.
    """

    print("🔧 TRAINING PIPELINE DEBUGGER")
    print("=" * 50)
    print("Investigating why your model has 48+ pixel errors on training data...")

    dataset_path = Path(dataset_path)

    # Step 1: Verify dataset structure and loading
    print("\n📁 STEP 1: Dataset Structure Verification")
    print("-" * 40)

    rgb_dir = dataset_path / "rgb"
    ann_dir = dataset_path / "annotations"

    if not rgb_dir.exists():
        print(f"❌ RGB directory not found: {rgb_dir}")
        return
    if not ann_dir.exists():
        print(f"❌ Annotations directory not found: {ann_dir}")
        return

    rgb_files = list(rgb_dir.glob("*.jpeg"))
    ann_files = list(ann_dir.glob("*.json"))

    print(f"✅ Found {len(rgb_files)} RGB files")
    print(f"✅ Found {len(ann_files)} annotation files")

    # Step 2: Check data loading and preprocessing
    print("\n🔍 STEP 2: Data Loading Analysis")
    print("-" * 35)

    # Test on first few samples
    test_samples = 3

    for i in range(test_samples):
        rgb_file = rgb_dir / f"rgb_{i}.jpeg"
        ann_file = ann_dir / f"annotation_{i}.json"

        if not rgb_file.exists() or not ann_file.exists():
            print(f"⚠️  Sample {i} missing files")
            continue

        print(f"\n--- Sample {i}: {rgb_file.name} ---")

        # Load image
        image = cv2.imread(str(rgb_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Image shape: {image_rgb.shape}")

        # Load annotation
        with open(ann_file) as f:
            annotation = json.load(f)

        if 'board_corners' not in annotation:
            print(f"❌ No board_corners in annotation")
            continue

        corners = annotation['board_corners']
        print(f"Corner coordinates:")
        print(f"  white_left:  {corners['white_left']}")
        print(f"  white_right: {corners['white_right']}")
        print(f"  black_right: {corners['black_right']}")
        print(f"  black_left:  {corners['black_left']}")

        # Check coordinate ranges
        all_coords = [
            corners['white_left'], corners['white_right'],
            corners['black_right'], corners['black_left']
        ]

        x_coords = [c[0] for c in all_coords]
        y_coords = [c[1] for c in all_coords]

        print(f"  X range: [{min(x_coords):.1f}, {max(x_coords):.1f}]")
        print(f"  Y range: [{min(y_coords):.1f}, {max(y_coords):.1f}]")

        # Verify coordinates are within image bounds
        h, w = image_rgb.shape[:2]
        valid_coords = all(0 <= x < w and 0 <= y < h for x, y in all_coords)
        print(f"  Coordinates valid: {valid_coords}")

        if not valid_coords:
            print(f"❌ PROBLEM: Coordinates outside image bounds!")
            print(f"   Image size: {w}×{h}")

    # Step 3: Test model loading and prediction
    print(f"\n🧠 STEP 3: Model Analysis")
    print("-" * 25)

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")

        # Check final layer activation
        final_layer = model.layers[-1]
        print(f"Final layer: {final_layer.__class__.__name__}")
        if hasattr(final_layer, 'activation'):
            activation = final_layer.activation
            print(f"Final activation: {activation}")

            if activation.__name__ != 'sigmoid':
                print(f"⚠️  WARNING: Final activation is {activation.__name__}, not sigmoid")
                print("   For normalized coordinates, should use sigmoid activation")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return

    # Step 4: Test coordinate normalization
    print(f"\n📐 STEP 4: Coordinate Normalization Test")
    print("-" * 40)

    # Test the exact preprocessing your model expects
    test_file = rgb_dir / "rgb_0.jpeg"
    test_ann = ann_dir / "annotation_0.json"

    if test_file.exists() and test_ann.exists():

        # Load test sample
        image = cv2.imread(str(test_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]

        with open(test_ann) as f:
            annotation = json.load(f)

        corners = annotation['board_corners']

        # Ground truth in original coordinates
        gt_original = [
            corners['white_left'][0], corners['white_left'][1],
            corners['white_right'][0], corners['white_right'][1],
            corners['black_right'][0], corners['black_right'][1],
            corners['black_left'][0], corners['black_left'][1]
        ]

        # Normalize to [0,1] (this is what model should predict)
        gt_normalized = [
            gt_original[0] / original_w, gt_original[1] / original_h,
            gt_original[2] / original_w, gt_original[3] / original_h,
            gt_original[4] / original_w, gt_original[5] / original_h,
            gt_original[6] / original_w, gt_original[7] / original_h
        ]

        # Preprocess image for model
        processed_image = cv2.resize(image_rgb, (224, 224))
        processed_image = processed_image.astype(np.float32) / 255.0
        processed_image = np.expand_dims(processed_image, axis=0)

        # Get model prediction
        prediction = model.predict(processed_image, verbose=0)[0]

        print(f"Ground truth (normalized): {[f'{x:.3f}' for x in gt_normalized]}")
        print(f"Model prediction:          {[f'{x:.3f}' for x in prediction]}")

        # Calculate errors
        errors = np.abs(np.array(gt_normalized) - np.array(prediction))
        pixel_errors = errors * np.array([original_w, original_h] * 4)

        print(f"Absolute differences:      {[f'{x:.3f}' for x in errors]}")
        print(f"Pixel errors:              {[f'{x:.1f}' for x in pixel_errors]}")
        print(f"Mean pixel error:          {np.mean(pixel_errors):.1f}")

        # Analysis
        if np.mean(pixel_errors) > 50:
            print(f"\n🚨 CONFIRMED: Large prediction errors!")

            if np.any(prediction < 0) or np.any(prediction > 1):
                print("❌ ISSUE: Predictions outside [0,1] range")
                print("   → Check final layer activation (should be sigmoid)")

            if np.all(prediction < 0.1) or np.all(prediction > 0.9):
                print("❌ ISSUE: Predictions clustered at edges")
                print("   → Model not learning properly")

            if np.std(prediction) < 0.05:
                print("❌ ISSUE: Very low prediction variance")
                print("   → Model might be stuck or undertrained")

        else:
            print(f"✅ Prediction errors reasonable for this sample")

    # Step 5: Visualize one sample
    print(f"\n🎨 STEP 5: Visualization")
    print("-" * 22)

    create_debug_visualization(test_file, test_ann, model)

    # Step 6: Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 20)

    print("Based on analysis:")
    print("1. If coordinates are outside [0,1]: Fix final layer activation")
    print("2. If large pixel errors: Check data loading pipeline")
    print("3. If clustered predictions: Increase model capacity or training time")
    print("4. If preprocessing issues: Verify normalization consistency")


def create_debug_visualization(image_path: str, annotation_path: str, model):
    """Create visualization showing ground truth vs prediction"""

    # Load data
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(annotation_path) as f:
        annotation = json.load(f)

    corners = annotation['board_corners']

    # Ground truth corners
    gt_corners = np.array([
        corners['white_left'],
        corners['white_right'],
        corners['black_right'],
        corners['black_left']
    ])

    # Model prediction
    processed = cv2.resize(image_rgb, (224, 224)).astype(np.float32) / 255.0
    prediction = model.predict(np.expand_dims(processed, 0), verbose=0)[0]

    # Convert prediction to pixel coordinates
    h, w = image_rgb.shape[:2]
    pred_corners = []
    for i in range(0, 8, 2):
        x = int(prediction[i] * w)
        y = int(prediction[i + 1] * h)
        pred_corners.append([x, y])
    pred_corners = np.array(pred_corners)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Ground truth
    img_gt = image_rgb.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    labels = ['WL', 'WR', 'BR', 'BL']

    for corner, color, label in zip(gt_corners, colors, labels):
        cv2.circle(img_gt, tuple(corner.astype(int)), 20, color, -1)
        cv2.putText(img_gt, label, (corner[0] + 25, corner[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.polylines(img_gt, [gt_corners.astype(int)], True, (0, 255, 0), 5)

    ax1.imshow(img_gt)
    ax1.set_title('Ground Truth Corners', fontsize=16)
    ax1.axis('off')

    # Prediction
    img_pred = image_rgb.copy()

    for corner, color, label in zip(pred_corners, colors, labels):
        cv2.circle(img_pred, tuple(corner.astype(int)), 20, color, -1)
        cv2.putText(img_pred, label, (corner[0] + 25, corner[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.polylines(img_pred, [pred_corners.astype(int)], True, (255, 0, 0), 5)

    ax2.imshow(img_pred)
    ax2.set_title('Model Predictions', fontsize=16)
    ax2.axis('off')

    # Add error information
    errors = np.sqrt(np.sum((gt_corners - pred_corners) ** 2, axis=1))
    error_text = f"Pixel Errors:\n"
    for i, (label, error) in enumerate(zip(labels, errors)):
        error_text += f"{label}: {error:.1f}px\n"
    error_text += f"Mean: {np.mean(errors):.1f}px"

    plt.figtext(0.02, 0.02, error_text, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))

    plt.tight_layout()
    plt.show()

    return np.mean(errors)


if __name__ == "__main__":
    print("🔧 TRAINING PIPELINE DEBUGGER")
    print("=" * 40)

    model_path = input("Enter model path (or press Enter for default): ").strip()
    if not model_path:
        model_path = "board_detector/outputs/corner_detection/fine_tuned_chess_corner_detector.keras"

    dataset_path = input("Enter dataset path (or press Enter for default): ").strip()
    if not dataset_path:
        dataset_path = "c:/datasets/ChessRender360"

    debug_training_pipeline(model_path, dataset_path)