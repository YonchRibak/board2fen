# Debug script to test chess pipeline loading
# Save this as debug_chess_pipeline.py and run it to identify the issue

import sys
import os
from pathlib import Path
import logging

# Set up logging to see detailed errors
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def debug_chess_pipeline():
    """Debug the chess pipeline loading step by step"""

    print("🔍 Debugging Chess Pipeline Loading")
    print("=" * 50)

    # Step 1: Check imports
    print("\n1. Testing imports...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return

    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return

    try:
        import chess
        print(f"✅ python-chess imported successfully")
    except ImportError as e:
        print(f"❌ python-chess import failed: {e}")
        return

    # Step 2: Check if board detector can be imported
    print("\n2. Testing board detector import...")
    try:
        from board_detector.chess_board_detector import ChessBoardDetector
        print("✅ ChessBoardDetector imported successfully")

        # Test instantiation
        detector = ChessBoardDetector(debug=False)
        print("✅ ChessBoardDetector instantiated successfully")
    except ImportError as e:
        print(f"❌ ChessBoardDetector import failed: {e}")
        print("   Make sure the board_detector module exists and is in your Python path")
        return
    except Exception as e:
        print(f"❌ ChessBoardDetector instantiation failed: {e}")
        return

    # Step 3: Check model path
    print("\n3. Testing model path...")

    # Import config to get model path
    try:
        from config import settings
        model_path = settings.absolute_model_path
        print(f"📁 Model path: {model_path}")

        if model_path.exists():
            print(f"✅ Model file exists")
            print(f"   File size: {model_path.stat().st_size / (1024 * 1024):.2f} MB")
        else:
            print(f"❌ Model file not found at: {model_path}")
            print("   Please check the model path in streamlit_config.py")

            # Show what files exist in the directory
            parent_dir = model_path.parent
            if parent_dir.exists():
                print(f"   Files in {parent_dir}:")
                for file in parent_dir.glob("*"):
                    print(f"     - {file.name}")
            else:
                print(f"   Parent directory {parent_dir} doesn't exist")
            return

    except Exception as e:
        print(f"❌ Error getting model path: {e}")
        return

    # Step 4: Test model loading
    print("\n4. Testing TensorFlow model loading...")
    try:
        model = tf.keras.models.load_model(str(model_path))
        print("✅ TensorFlow model loaded successfully")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")

        # Test prediction to make sure model works
        import numpy as np
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        prediction = model.predict(test_input, verbose=0)
        print(f"✅ Model prediction test successful, output shape: {prediction.shape}")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print(f"   Error type: {type(e).__name__}")

        # Additional checks for model file
        try:
            import h5py
            with h5py.File(str(model_path), 'r') as f:
                print(f"   Model file is readable as HDF5")
        except Exception as h5_error:
            print(f"   Model file is not readable as HDF5: {h5_error}")
        return

    # Step 5: Test full pipeline initialization
    print("\n5. Testing full ChessPipelineService initialization...")
    try:
        from _helpers import ChessPipelineService
        pipeline = ChessPipelineService(model_path=str(model_path))

        if pipeline.model_loaded:
            print("✅ ChessPipelineService initialized successfully")
            print(f"   Board detector ready: {pipeline.board_detector is not None}")
            print(f"   Piece classifier ready: {pipeline.piece_classifier is not None}")
        else:
            print("❌ ChessPipelineService initialization failed")

    except Exception as e:
        print(f"❌ ChessPipelineService initialization error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ All checks passed! Your chess pipeline should work.")


if __name__ == "__main__":
    debug_chess_pipeline()