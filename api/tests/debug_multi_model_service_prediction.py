#!/usr/bin/env python3
"""
Chess Prediction Service Debug Script

This script helps diagnose issues with chess board position prediction,
particularly when getting empty board results. It tests various confidence
thresholds and analyzes the prediction pipeline in detail.

Usage:
    python debug_chess_prediction.py /path/to/test_image.jpg
"""

import requests
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import argparse
import time


class ChessPredictionDebugger:
    def __init__(self, api_base_url: str = "http://localhost:8081"):
        self.api_base_url = api_base_url.rstrip('/')
        self.session = requests.Session()

    def test_api_health(self) -> bool:
        """Test if the API is responding"""
        try:
            response = self.session.get(f"{self.api_base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print("🟢 API Health Check:")
                print(f"  Status: {health_data.get('status')}")
                print(f"  Model Ready: {health_data.get('model_ready')}")
                print(f"  Database Ready: {health_data.get('database_ready')}")
                return health_data.get('status') == 'healthy'
            else:
                print(f"🔴 API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"🔴 Cannot connect to API: {e}")
            return False

    def get_service_info(self) -> Dict[str, Any]:
        """Get current service information"""
        try:
            response = self.session.get(f"{self.api_base_url}/service/current")
            if response.status_code == 200:
                service_info = response.json()
                print("\n🔍 Current Service Info:")
                print(f"  Service Type: {service_info.get('service_type')}")
                print(f"  Service Loaded: {service_info.get('service_loaded')}")
                print(f"  Available Services: {service_info.get('available_services')}")
                return service_info
            else:
                print(f"🔴 Failed to get service info: {response.status_code}")
                return {}
        except Exception as e:
            print(f"🔴 Error getting service info: {e}")
            return {}

    def test_basic_prediction(self, image_path: str) -> Dict[str, Any]:
        """Test basic prediction with current settings"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.api_base_url}/predict", files=files)

            if response.status_code == 200:
                result = response.json()
                print(f"\n⚡ Basic Prediction Results:")
                print(f"  Success: {result.get('success')}")
                print(f"  Board Detected: {result.get('board_detected')}")
                print(f"  FEN: {result.get('fen', 'None')}")
                print(f"  Confidence: {result.get('confidence_score')}")
                print(f"  Processing Time: {result.get('processing_time_ms')}ms")

                # Count pieces in board matrix
                board_matrix = result.get('board_matrix', [])
                if board_matrix:
                    piece_count = sum(1 for row in board_matrix for cell in row if cell != '')
                    print(f"  Pieces on Board: {piece_count}/32 expected")

                    # Show board matrix
                    print(f"\n  Board Matrix:")
                    for i, row in enumerate(board_matrix):
                        row_str = ' '.join([f"{cell:2}" if cell else " ." for cell in row])
                        print(f"    {8 - i}: {row_str}")
                    print(f"       a  b  c  d  e  f  g  h")

                return result
            else:
                print(f"🔴 Prediction failed: {response.status_code}")
                print(f"Response: {response.text}")
                return {}
        except Exception as e:
            print(f"🔴 Error in basic prediction: {e}")
            return {}

    def test_detailed_prediction(self, image_path: str) -> Dict[str, Any]:
        """Test detailed prediction analysis"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.api_base_url}/debug/predict-detailed", files=files)

            if response.status_code == 200:
                result = response.json()
                print(f"\n🔬 Detailed Prediction Analysis:")
                print(f"  Service Type: {result.get('service_type')}")

                prediction_result = result.get('prediction_result', {})
                processing_steps = prediction_result.get('processing_steps', {})

                if processing_steps:
                    print(f"\n  Processing Steps Timing:")
                    for step, time_ms in processing_steps.items():
                        if isinstance(time_ms, (int, float)):
                            print(f"    {step}: {time_ms:.2f}ms")

                return result
            else:
                print(f"🔴 Detailed prediction failed: {response.status_code}")
                return {}
        except Exception as e:
            print(f"🔴 Error in detailed prediction: {e}")
            return {}

    def test_multi_model_pipeline(self, image_path: str) -> Dict[str, Any]:
        """Test multi-model pipeline specifically"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.api_base_url}/debug/multi-model-detailed", files=files)

            if response.status_code == 200:
                result = response.json()
                print(f"\n🔍 Multi-Model Pipeline Analysis:")

                # Board detection analysis
                board_detection = result.get('board_detection', {})
                print(f"  Board Detection:")
                print(f"    Corners Found: {board_detection.get('corners_found')}")
                print(f"    Segmentation Confidence: {board_detection.get('segmentation_confidence')}")

                # Piece detection analysis (CRITICAL)
                piece_detection = result.get('piece_detection', {})
                print(f"\n  Piece Detection (CRITICAL):")
                print(f"    Pieces Found: {piece_detection.get('pieces_found')}")

                pieces_details = piece_detection.get('pieces_details', [])
                if pieces_details:
                    print(f"    First few pieces:")
                    for i, piece in enumerate(pieces_details[:5]):
                        conf = piece.get('confidence', 0)
                        class_name = piece.get('class_name', 'unknown')
                        print(f"      {i + 1}: {class_name} (conf: {conf:.3f})")

                # Coordinate transformation
                coord_transform = result.get('coordinate_transformation', {})
                print(f"\n  Coordinate Transformation:")
                print(f"    Original Pieces: {coord_transform.get('original_pieces')}")
                print(f"    Warped Pieces: {coord_transform.get('warped_pieces')}")

                # Piece assignment
                piece_assignment = result.get('piece_assignment', {})
                print(f"\n  Piece Assignment:")
                print(f"    Pieces Assigned to Board: {piece_assignment.get('pieces_assigned')}")

                # Threshold analysis
                threshold_analysis = result.get('threshold_analysis', {})
                if 'low_threshold_pieces' in threshold_analysis:
                    low_thresh_info = threshold_analysis['low_threshold_pieces']
                    print(f"\n  Low Threshold Test (conf=0.1):")
                    print(f"    Pieces Found: {low_thresh_info.get('count')}")

                    low_pieces = low_thresh_info.get('pieces', [])
                    if low_pieces:
                        print(f"    Sample pieces:")
                        for piece in low_pieces[:8]:
                            conf = piece.get('confidence', 0)
                            class_name = piece.get('class_name', 'unknown')
                            print(f"      {class_name}: {conf:.3f}")

                return result
            else:
                print(f"🔴 Multi-model analysis failed: {response.status_code}")
                return {}
        except Exception as e:
            print(f"🔴 Error in multi-model analysis: {e}")
            return {}

    def test_raw_prediction(self) -> Dict[str, Any]:
        """Test raw model prediction (end-to-end only)"""
        try:
            response = self.session.get(f"{self.api_base_url}/debug/raw-prediction")

            if response.status_code == 200:
                result = response.json()
                print(f"\n🧠 Raw Model Analysis:")
                print(f"  Model Output Shape: {result.get('model_output_shape')}")
                print(f"  Output Range: {result.get('output_range')}")
                print(f"  Output Mean: {result.get('output_mean'):.4f}")
                print(f"  Output Std: {result.get('output_std'):.4f}")

                squares_analysis = result.get('squares_analysis', [])
                if squares_analysis:
                    print(f"\n  Sample Square Predictions:")
                    for square in squares_analysis[:8]:
                        pos = square.get('square_position')
                        class_name = square.get('predicted_class_name')
                        conf = square.get('confidence')
                        symbol = square.get('piece_symbol')
                        print(f"    {pos}: {class_name} (conf: {conf:.3f}) -> '{symbol}'")

                return result
            else:
                print(f"🔴 Raw prediction analysis failed: {response.status_code}")
                return {}
        except Exception as e:
            print(f"🔴 Error in raw prediction analysis: {e}")
            return {}

    def test_service_switching(self) -> Dict[str, Any]:
        """Test switching between available services"""
        results = {}

        # Get available services
        service_info = self.get_service_info()
        available_services = service_info.get('available_services', [])
        current_service = service_info.get('service_type')

        print(f"\n🔄 Testing Service Switching:")
        print(f"  Current Service: {current_service}")
        print(f"  Available Services: {available_services}")

        for service_type in available_services:
            if service_type == current_service:
                print(f"  ✅ Already using {service_type}")
                continue

            try:
                # Switch service
                switch_data = {"service_type": service_type}
                response = self.session.post(f"{self.api_base_url}/service/switch", json=switch_data)

                if response.status_code == 200:
                    switch_result = response.json()
                    print(f"  ✅ Switched to {service_type}: {switch_result.get('message')}")
                    results[service_type] = {"switch_success": True}
                else:
                    print(f"  ❌ Failed to switch to {service_type}: {response.status_code}")
                    results[service_type] = {"switch_success": False, "error": response.text}

            except Exception as e:
                print(f"  ❌ Error switching to {service_type}: {e}")
                results[service_type] = {"switch_success": False, "error": str(e)}

        return results

    def run_comprehensive_debug(self, image_path: str) -> Dict[str, Any]:
        """Run comprehensive debugging analysis"""
        print("=" * 60)
        print("🔍 CHESS PREDICTION SERVICE DEBUG ANALYSIS")
        print("=" * 60)

        results = {}

        # 1. Health check
        if not self.test_api_health():
            print("\n❌ API is not healthy, stopping debug analysis")
            return results

        # 2. Service info
        results['service_info'] = self.get_service_info()

        # 3. Basic prediction
        results['basic_prediction'] = self.test_basic_prediction(image_path)

        # 4. Detailed prediction
        results['detailed_prediction'] = self.test_detailed_prediction(image_path)

        # 5. Multi-model specific analysis (if applicable)
        current_service = results['service_info'].get('service_type')
        if current_service == "multi_model_pipeline":
            results['multi_model_analysis'] = self.test_multi_model_pipeline(image_path)

        # 6. Raw prediction analysis (if end-to-end)
        if current_service == "end_to_end":
            results['raw_prediction'] = self.test_raw_prediction()

        # 7. Test service switching
        results['service_switching'] = self.test_service_switching()

        # Generate recommendations
        self.generate_recommendations(results)

        return results

    def generate_recommendations(self, debug_results: Dict[str, Any]):
        """Generate recommendations based on debug results"""
        print(f"\n" + "=" * 60)
        print("📋 DIAGNOSTIC RECOMMENDATIONS")
        print("=" * 60)

        basic_pred = debug_results.get('basic_prediction', {})
        service_info = debug_results.get('service_info', {})
        current_service = service_info.get('service_type')

        # Check if board was detected
        if not basic_pred.get('board_detected', False):
            print("🔴 CRITICAL: Board not detected")
            print("  Recommendations:")
            print("  - Check image quality and lighting")
            print("  - Ensure chess board is clearly visible")
            print("  - Try lowering segmentation_confidence threshold")
            print("  - Test with different image angles")

        # Check if prediction was successful
        if not basic_pred.get('success', False):
            print("🔴 CRITICAL: Prediction failed")
            print("  Recommendations:")
            print("  - Check error message in prediction result")
            print("  - Verify model files are accessible")
            print("  - Check memory/processing resources")

        # Check piece count
        board_matrix = basic_pred.get('board_matrix', [])
        if board_matrix:
            piece_count = sum(1 for row in board_matrix for cell in row if cell != '')
            if piece_count == 0:
                print("🔴 CRITICAL: No pieces detected on board")
                print("  Recommendations:")
                if current_service == "multi_model_pipeline":
                    print("  - Lower piece_detection_confidence threshold (currently in config)")
                    print("  - Lower iou_threshold for non-maximum suppression")
                    print("  - Check if piece detection model is working")
                elif current_service == "end_to_end":
                    print("  - Lower end_to_end_piece_confidence threshold")
                    print("  - Check model training data compatibility")
            elif piece_count < 20:  # Typical game has 20-32 pieces
                print(f"⚠️  WARNING: Only {piece_count} pieces detected (expected 20-32)")
                print("  Recommendations:")
                print("  - Lower confidence thresholds slightly")
                print("  - Check for piece occlusion in image")

        # Service-specific recommendations
        if current_service == "multi_model_pipeline":
            multi_analysis = debug_results.get('multi_model_analysis', {})

            # Check segmentation confidence
            board_detection = multi_analysis.get('board_detection', {})
            seg_conf = board_detection.get('segmentation_confidence', 0)
            if seg_conf < 0.5:
                print(f"⚠️  LOW: Segmentation confidence: {seg_conf:.3f}")
                print("  Recommendations:")
                print("  - Image quality might be poor")
                print("  - Consider different camera angle")

            # Check piece detection
            piece_detection = multi_analysis.get('piece_detection', {})
            pieces_found = piece_detection.get('pieces_found', 0)
            if pieces_found == 0:
                print("🔴 CRITICAL: No pieces detected by YOLO model")
                print("  Recommendations:")
                print("  - Set piece_detection_confidence to 0.05 or lower")
                print("  - Check if pieces model file is corrupted")
                print("  - Verify piece model is trained on similar images")

            # Check threshold analysis
            threshold_analysis = multi_analysis.get('threshold_analysis', {})
            if 'low_threshold_pieces' in threshold_analysis:
                low_count = threshold_analysis['low_threshold_pieces'].get('count', 0)
                if low_count > pieces_found:
                    print(f"💡 INSIGHT: Low threshold finds {low_count} pieces vs {pieces_found} at normal threshold")
                    print("  Recommendations:")
                    print("  - Lower piece_detection_confidence in config")

        elif current_service == "end_to_end":
            raw_pred = debug_results.get('raw_prediction', {})
            if raw_pred:
                output_mean = raw_pred.get('output_mean', 0)
                output_std = raw_pred.get('output_std', 0)
                if output_std < 0.1:
                    print(f"⚠️  WARNING: Low output variance (std: {output_std:.4f})")
                    print("  Recommendations:")
                    print("  - Model might be undertrained")
                    print("  - Check input image preprocessing")

        # Configuration recommendations
        print(f"\n💡 CONFIGURATION TUNING SUGGESTIONS:")
        if current_service == "multi_model_pipeline":
            print("  In config.py, try lowering these values:")
            print("  - piece_detection_confidence: 0.1 -> 0.05")
            print("  - iou_threshold: 0.3 -> 0.2")
            print("  - segmentation_confidence: 0.2 -> 0.15")
        elif current_service == "end_to_end":
            print("  In config.py, try lowering these values:")
            print("  - end_to_end_piece_confidence: 0.3 -> 0.2")
            print("  - end_to_end_empty_confidence: 0.5 -> 0.4")

        print(f"\n🔧 IMMEDIATE TESTING STEPS:")
        print("  1. Try the suggested threshold changes")
        print("  2. Test with a different, high-quality chess image")
        print("  3. Check server logs for additional error details")
        print("  4. If using multi-model, try switching to end_to_end service")


def main():
    parser = argparse.ArgumentParser(description="Debug chess prediction service")
    parser.add_argument("image_path", help="Path to test chess image")
    parser.add_argument("--api-url", default="http://localhost:8081", help="API base URL")
    parser.add_argument("--output", help="Save debug results to JSON file")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"❌ Image file not found: {args.image_path}")
        sys.exit(1)

    debugger = ChessPredictionDebugger(args.api_url)
    results = debugger.run_comprehensive_debug(args.image_path)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Debug results saved to: {args.output}")


if __name__ == "__main__":
    main()