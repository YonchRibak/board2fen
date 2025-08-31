
# Run this script to test your model directly

import numpy as np
import tensorflow as tf
from pathlib import Path


def test_model(model_path):
    """Test the model with dummy data to identify issues"""

    print(f"🔍 Testing model at: {model_path}")

    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Print model info
    print(f"\n📋 Model Information:")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Number of layers: {len(model.layers)}")

    # Print model summary
    print(f"\n📋 Model Summary:")
    model.summary()

    # Test with dummy data
    input_shape = model.input_shape[1:]  # Remove batch dimension
    print(f"\n🧪 Testing with dummy input shape: {input_shape}")

    # Create dummy input
    dummy_input = np.random.random((1,) + input_shape).astype(np.float32)
    print(f"   Dummy input shape: {dummy_input.shape}")
    print(f"   Dummy input dtype: {dummy_input.dtype}")
    print(f"   Dummy input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")

    # Test prediction
    try:
        print(f"\n🚀 Running prediction...")
        prediction = model.predict(dummy_input, verbose=1)

        print(f"✅ Prediction successful!")
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   Prediction dtype: {prediction.dtype}")
        print(f"   Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")

        # Check if it's multiple outputs
        if isinstance(prediction, (list, tuple)):
            print(f"   ⚠️ Multiple outputs detected: {len(prediction)}")
            for i, output in enumerate(prediction):
                print(f"      Output {i}: shape={output.shape}, dtype={output.dtype}")

        # Test accessing prediction[0]
        try:
            first_prediction = prediction[0]
            print(f"   prediction[0] shape: {first_prediction.shape}")

            # Test different interpretations
            if len(first_prediction.shape) == 1:
                print(f"   🔍 1D output - length: {len(first_prediction)}")
                if len(first_prediction) == 64 * 13:  # 64 squares * 13 piece classes
                    reshaped = first_prediction.reshape(64, 13)
                    print(f"   🔍 Can reshape to (64, 13): {reshaped.shape}")
                elif len(first_prediction) == 8 * 8 * 13:  # 8x8 board * 13 piece classes
                    reshaped = first_prediction.reshape(8, 8, 13)
                    print(f"   🔍 Can reshape to (8, 8, 13): {reshaped.shape}")

            elif len(first_prediction.shape) == 2:
                print(f"   🔍 2D output: {first_prediction.shape}")
                if first_prediction.shape == (64, 13):
                    print(f"   🔍 Perfect! (64, 13) format - flattened board")
                elif first_prediction.shape[0] == 64:
                    print(f"   🔍 64 squares detected, {first_prediction.shape[1]} classes")

            elif len(first_prediction.shape) == 3:
                print(f"   🔍 3D output: {first_prediction.shape}")
                if first_prediction.shape == (8, 8, 13):
                    print(f"   🔍 Perfect! (8, 8, 13) format - board format")

        except Exception as e:
            print(f"   ❌ Error accessing prediction[0]: {e}")

    except Exception as e:
        import traceback
        print(f"❌ Prediction failed: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    # Test your model
    model_path = "../api/cnn_models/final_model.keras"  # Adjust path as needed
    test_model(model_path)