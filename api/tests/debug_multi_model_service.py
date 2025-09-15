# Debug script to identify multi-model pipeline initialization issues
import sys
import logging
from pathlib import Path

# Add your project to path (adjust if needed)
sys.path.insert(0, str(Path(__file__).parent))

from api.config import settings


def debug_multi_model_pipeline():
    """Debug the multi-model pipeline initialization issues"""

    print("=== Multi-Model Pipeline Debug ===")

    # 1. Check ultralytics dependency
    print("\n1. Checking ultralytics dependency...")
    try:
        from ultralytics import YOLO
        print("✅ ultralytics is available")

        # Check YOLO version
        try:
            import ultralytics
            print(f"   Version: {ultralytics.__version__}")
        except:
            print("   Version: Unknown")

    except ImportError as e:
        print("❌ ultralytics is NOT available")
        print(f"   Error: {e}")
        print("   Solution: Install with 'pip install ultralytics'")
        return False

    # 2. Check configuration
    print("\n2. Checking configuration...")
    config = settings.get_service_config()
    print(f"   Service type: {settings.chess_service_type}")

    if settings.chess_service_type == "multi_model_pipeline":
        seg_path = settings.segmentation_model_path
        pieces_path = settings.pieces_model_path

        print(f"   Segmentation model: {seg_path}")
        print(f"   Pieces model: {pieces_path}")
    else:
        print("   Note: Current config is set to 'end_to_end', not 'multi_model_pipeline'")
        seg_path = settings.segmentation_model_path
        pieces_path = settings.pieces_model_path
        print(f"   But multi-model paths are: {seg_path}, {pieces_path}")

    # 3. Test model URL accessibility
    print("\n3. Testing model URL accessibility...")

    import requests

    for name, url in [("Segmentation", seg_path), ("Pieces", pieces_path)]:
        print(f"   Testing {name} model URL...")
        try:
            # Just test if URL is reachable (head request)
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                print(f"   ✅ {name} model URL is accessible")
                print(f"      Status: {response.status_code}")
                if 'content-length' in response.headers:
                    size_mb = int(response.headers['content-length']) / (1024 * 1024)
                    print(f"      Size: {size_mb:.1f} MB")
            else:
                print(f"   ❌ {name} model URL returned status {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"   ❌ {name} model URL timed out")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ {name} model URL error: {e}")

    # 4. Test YOLO model loading
    print("\n4. Testing YOLO model loading...")

    try:
        print("   Attempting to load segmentation model...")
        seg_model = YOLO(seg_path)
        print("   ✅ Segmentation model loaded successfully")
        print(f"      Model info: {seg_model.info()}")

        print("   Attempting to load pieces model...")
        pieces_model = YOLO(pieces_path)
        print("   ✅ Pieces model loaded successfully")
        print(f"      Model info: {pieces_model.info()}")

        return True

    except Exception as e:
        print(f"   ❌ YOLO model loading failed: {e}")

        # More detailed error info
        import traceback
        print("   Full traceback:")
        print(traceback.format_exc())

        return False


def test_service_creation():
    """Test creating the multi-model service directly"""
    print("\n=== Direct Service Creation Test ===")

    try:
        from api.services.base import ServiceFactory

        service_config = {
            'segmentation_model_path': settings.segmentation_model_path,
            'pieces_model_path': settings.pieces_model_path,
            'piece_classes': [
                'white-king', 'white-queen', 'white-rook', 'white-bishop', 'white-knight', 'white-pawn',
                'black-king', 'black-queen', 'black-rook', 'black-bishop', 'black-knight', 'black-pawn'
            ],
            'segmentation_confidence': settings.segmentation_confidence,
            'piece_confidence': settings.piece_detection_confidence,
            'iou_threshold': settings.iou_threshold
        }

        print("Creating multi-model service...")
        service = ServiceFactory.create_service("multi_model_pipeline", service_config)

        print(f"Service created: {service}")
        print(f"Service ready: {service.is_ready()}")
        print(f"Service info: {service.get_service_info()}")

        return True

    except Exception as e:
        print(f"❌ Service creation failed: {e}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return False


def provide_solutions():
    """Provide solutions for common issues"""
    print("\n=== Potential Solutions ===")

    print("1. Install ultralytics:")
    print("   pip install ultralytics")

    print("\n2. If models are not accessible, try downloading manually:")
    print("   # Download segmentation model")
    print(f"   curl -o seg_model.pt '{settings.segmentation_model_path}'")
    print("   # Download pieces model")
    print(f"   curl -o pieces_model.pt '{settings.pieces_model_path}'")

    print("\n3. Update config to use local paths:")
    print("   # In config.py or .env file:")
    print("   segmentation_model_path=./seg_model.pt")
    print("   pieces_model_path=./pieces_model.pt")

    print("\n4. Check firewall/network restrictions")
    print("   - Ensure access to storage.googleapis.com")
    print("   - Check if behind corporate firewall")

    print("\n5. For development, you can temporarily use smaller models:")
    print("   - Use 'yolov8n.pt' for testing (auto-downloaded)")
    print("   - Replace URLs with 'yolov8n-seg.pt' and 'yolov8n.pt' for testing")


if __name__ == "__main__":
    # Enable logging to see detailed error messages
    logging.basicConfig(level=logging.INFO)

    print("Chess Multi-Model Pipeline Debug Tool")
    print("=" * 40)

    # Run all debug checks
    models_accessible = debug_multi_model_pipeline()

    if models_accessible:
        print("\n✅ Models seem accessible, testing service creation...")
        service_created = test_service_creation()

        if service_created:
            print("\n🎉 Multi-model pipeline should work!")
        else:
            print("\n❌ Service creation failed - check logs above")
    else:
        print("\n❌ Model accessibility issues found")

    provide_solutions()