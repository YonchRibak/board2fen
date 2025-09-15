#!/usr/bin/env python3
"""
Chess Coordinate Transformation Debug Script

This script specifically debugs the coordinate transformation and piece assignment
issues identified in the multi-model pipeline where pieces are being lost during:
1. Coordinate transformation from original → warped space
2. Piece assignment to 8x8 board squares

Usage:
    python debug_coordinates.py /path/to/test_image.jpg
"""

import numpy as np
import cv2
from pathlib import Path
import sys
import json
import argparse
from typing import List, Dict, Tuple, Optional

# Add project root to path to import from api
# Navigate up from api/tests/ to project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from api._helpers import ImageProcessor
    from api.services.multi_model import MultiModelPipelineService
    from api.config import settings
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the project root or have the correct Python path")
    print("Try running: python -m api.tests.debug_coordinates <image_path>")
    sys.exit(1)


class CoordinateTransformationDebugger:
    def __init__(self):
        # Initialize multi-model service for debugging
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

        self.service = MultiModelPipelineService(service_config)
        if not self.service.is_ready():
            raise Exception("Failed to initialize MultiModelPipelineService")

    def debug_coordinate_transformation(self, image_path: str) -> Dict:
        """Debug the coordinate transformation pipeline step by step"""

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        image = ImageProcessor.load_image_from_bytes(image_bytes)
        if image is None:
            raise Exception("Failed to load image")

        print("=" * 60)
        print("🔍 COORDINATE TRANSFORMATION DEBUG")
        print("=" * 60)

        debug_info = {
            'image_shape': image.shape,
            'steps': {}
        }

        # Step 1: Board detection
        print("Step 1: Board Detection")
        corners, seg_confidence, mask = self.service._detect_board_with_segmentation(image)

        if not corners:
            raise Exception("Board detection failed")

        print(f"  ✅ Board detected with confidence: {seg_confidence:.3f}")
        print(f"  📍 Corners: {corners}")

        debug_info['steps']['board_detection'] = {
            'corners': corners,
            'confidence': seg_confidence
        }

        # Step 2: Piece detection on original image
        print("\nStep 2: Original Piece Detection")
        original_pieces = self.service._detect_pieces(image)
        print(f"  ✅ Detected {len(original_pieces)} pieces")

        # Analyze original piece positions
        print("  📍 Original piece positions (first 10):")
        for i, piece in enumerate(original_pieces[:10]):
            x, y = piece['x_center'], piece['y_center']
            conf = piece['confidence']
            class_name = piece.get('class_name', 'unknown')
            print(f"    {i + 1:2d}: {class_name:12s} at ({x:6.1f}, {y:6.1f}) conf={conf:.3f}")

        debug_info['steps']['original_pieces'] = {
            'count': len(original_pieces),
            'pieces': original_pieces
        }

        # Step 3: Board warping
        print("\nStep 3: Board Warping")
        warped_board, transform_matrix, final_corners = self.service._warp_board_from_mask(
            image, mask, corners, output_size=800
        )

        if warped_board is None or transform_matrix is None:
            raise Exception("Board warping failed")

        print(f"  ✅ Board warped to shape: {warped_board.shape}")
        print(f"  📐 Transform matrix shape: {transform_matrix.shape}")

        debug_info['steps']['board_warping'] = {
            'warped_shape': warped_board.shape,
            'output_size': 800
        }

        # Step 4: DETAILED coordinate transformation analysis
        print("\nStep 4: Coordinate Transformation Analysis")
        warped_pieces = []
        lost_pieces = []
        transformation_details = []

        for i, piece in enumerate(original_pieces):
            x_orig, y_orig = piece['x_center'], piece['y_center']

            try:
                # Transform coordinates
                point = np.array([[[x_orig, y_orig]]], dtype=np.float32)
                transformed_point = cv2.perspectiveTransform(point, transform_matrix)
                x_new, y_new = transformed_point[0][0]

                # Check bounds
                within_bounds = 0 <= x_new < 800 and 0 <= y_new < 800

                detail = {
                    'original_pos': (x_orig, y_orig),
                    'transformed_pos': (x_new, y_new),
                    'within_bounds': within_bounds,
                    'class_name': piece.get('class_name', 'unknown'),
                    'confidence': piece['confidence']
                }
                transformation_details.append(detail)

                if within_bounds:
                    warped_piece = piece.copy()
                    warped_piece['x_center'] = float(x_new)
                    warped_piece['y_center'] = float(y_new)
                    warped_pieces.append(warped_piece)
                else:
                    lost_pieces.append(detail)

            except Exception as e:
                lost_pieces.append({
                    'original_pos': (x_orig, y_orig),
                    'error': str(e),
                    'class_name': piece.get('class_name', 'unknown')
                })

        print(f"  ✅ Transformed {len(original_pieces)} → {len(warped_pieces)} pieces")
        print(f"  ❌ Lost {len(lost_pieces)} pieces during transformation")

        if lost_pieces:
            print("  📍 Lost pieces analysis:")
            for i, lost in enumerate(lost_pieces[:5]):  # Show first 5
                orig_pos = lost.get('original_pos', (0, 0))
                trans_pos = lost.get('transformed_pos', 'error')
                class_name = lost.get('class_name', 'unknown')
                if isinstance(trans_pos, tuple):
                    print(
                        f"    {i + 1}: {class_name} ({orig_pos[0]:.1f},{orig_pos[1]:.1f}) → ({trans_pos[0]:.1f},{trans_pos[1]:.1f}) [OUT OF BOUNDS]")
                else:
                    print(
                        f"    {i + 1}: {class_name} ({orig_pos[0]:.1f},{orig_pos[1]:.1f}) → ERROR: {lost.get('error', 'unknown')}")

        debug_info['steps']['coordinate_transformation'] = {
            'original_count': len(original_pieces),
            'transformed_count': len(warped_pieces),
            'lost_count': len(lost_pieces),
            'transformation_details': transformation_details
        }

        # Step 5: DETAILED piece assignment analysis
        print("\nStep 5: Piece Assignment Analysis")
        board_matrix = [[''] * 8 for _ in range(8)]
        assignment_details = []
        assignment_conflicts = []

        square_size = 800 / 8  # 100 pixels per square
        print(f"  📏 Square size: {square_size} pixels")

        for i, piece in enumerate(warped_pieces):
            x, y = piece['x_center'], piece['y_center']

            # Calculate which square
            col = int(x // square_size)
            row = int(y // square_size)

            # Check bounds
            valid_assignment = 0 <= row < 8 and 0 <= col < 8

            assignment_detail = {
                'piece_index': i,
                'class_name': piece.get('class_name', 'unknown'),
                'position': (x, y),
                'calculated_square': (row, col),
                'valid_assignment': valid_assignment,
                'confidence': piece['confidence']
            }

            if valid_assignment:
                square_notation = f"{chr(ord('a') + col)}{8 - row}"
                assignment_detail['square_notation'] = square_notation

                # Convert piece to symbol
                piece_symbol = self.service._convert_piece_category_to_symbol(piece)
                assignment_detail['piece_symbol'] = piece_symbol

                if piece_symbol:
                    # Check for conflicts
                    if board_matrix[row][col] != '':
                        assignment_conflicts.append({
                            'square': square_notation,
                            'existing_piece': board_matrix[row][col],
                            'new_piece': piece_symbol,
                            'new_piece_confidence': piece['confidence']
                        })
                        assignment_detail['conflict'] = True
                    else:
                        board_matrix[row][col] = piece_symbol
                        assignment_detail['assigned'] = True
                else:
                    assignment_detail['symbol_conversion_failed'] = True
            else:
                assignment_detail['out_of_grid'] = True

            assignment_details.append(assignment_detail)

        # Count final assignments
        assigned_pieces = sum(1 for row in board_matrix for cell in row if cell != '')

        print(f"  ✅ Assigned {assigned_pieces} pieces to board")
        print(f"  ⚠️  {len(assignment_conflicts)} assignment conflicts")

        # Show assignment details for first few pieces
        print("  📍 Assignment details (first 10 pieces):")
        for detail in assignment_details[:10]:
            class_name = detail['class_name']
            pos = detail['position']
            square = detail.get('square_notation', 'INVALID')
            symbol = detail.get('piece_symbol', 'NO_SYMBOL')
            assigned = detail.get('assigned', False)

            status = '✅' if assigned else '❌'
            print(f"    {status} {class_name:12s} at ({pos[0]:5.1f},{pos[1]:5.1f}) → {square} ({symbol})")

        if assignment_conflicts:
            print("  ⚠️  Assignment conflicts:")
            for conflict in assignment_conflicts:
                print(
                    f"    Square {conflict['square']}: {conflict['existing_piece']} vs {conflict['new_piece']} (conf={conflict['new_piece_confidence']:.3f})")

        debug_info['steps']['piece_assignment'] = {
            'warped_pieces_count': len(warped_pieces),
            'assigned_pieces_count': assigned_pieces,
            'conflicts_count': len(assignment_conflicts),
            'assignment_details': assignment_details,
            'conflicts': assignment_conflicts,
            'final_board_matrix': board_matrix
        }

        # Step 6: Generate diagnostic insights
        print("\n" + "=" * 60)
        print("📋 DIAGNOSTIC INSIGHTS")
        print("=" * 60)

        self._generate_coordinate_insights(debug_info)

        return debug_info

    def _generate_coordinate_insights(self, debug_info: Dict):
        """Generate insights about coordinate transformation issues"""

        coord_transform = debug_info['steps']['coordinate_transformation']
        piece_assignment = debug_info['steps']['piece_assignment']

        original_count = coord_transform['original_count']
        transformed_count = coord_transform['transformed_count']
        assigned_count = piece_assignment['assigned_pieces_count']

        # Analyze transformation losses
        if coord_transform['lost_count'] > 0:
            print("🔴 COORDINATE TRANSFORMATION ISSUES:")
            print(f"  Lost {coord_transform['lost_count']} pieces during perspective transformation")

            lost_details = [d for d in coord_transform['transformation_details'] if not d['within_bounds']]
            if lost_details:
                # Analyze why pieces are being lost
                out_of_bounds_positions = []
                for detail in lost_details:
                    if 'transformed_pos' in detail:
                        x, y = detail['transformed_pos']
                        out_of_bounds_positions.append((x, y))

                if out_of_bounds_positions:
                    x_coords = [pos[0] for pos in out_of_bounds_positions]
                    y_coords = [pos[1] for pos in out_of_bounds_positions]

                    print(f"  Transformed coordinates range:")
                    print(f"    X: {min(x_coords):.1f} to {max(x_coords):.1f} (valid: 0-800)")
                    print(f"    Y: {min(y_coords):.1f} to {max(y_coords):.1f} (valid: 0-800)")

                    # Specific recommendations
                    if any(x < 0 for x in x_coords) or any(x > 800 for x in x_coords):
                        print("  🔧 FIX: Perspective transformation is pushing pieces outside X bounds")
                    if any(y < 0 for y in y_coords) or any(y > 800 for y in y_coords):
                        print("  🔧 FIX: Perspective transformation is pushing pieces outside Y bounds")

        # Analyze assignment losses
        assignment_loss = transformed_count - assigned_count
        if assignment_loss > 0:
            print(f"\n🔴 PIECE ASSIGNMENT ISSUES:")
            print(f"  Lost {assignment_loss} pieces during square assignment")

            # Analyze assignment problems
            assignment_details = piece_assignment['assignment_details']

            no_symbol_count = sum(1 for d in assignment_details if d.get('symbol_conversion_failed', False))
            out_of_grid_count = sum(1 for d in assignment_details if d.get('out_of_grid', False))
            conflict_count = sum(1 for d in assignment_details if d.get('conflict', False))

            if no_symbol_count > 0:
                print(f"  ⚠️  {no_symbol_count} pieces failed symbol conversion")
            if out_of_grid_count > 0:
                print(f"  ⚠️  {out_of_grid_count} pieces fell outside 8x8 grid")
            if conflict_count > 0:
                print(f"  ⚠️  {conflict_count} pieces had square conflicts")

        # Recommendations
        print(f"\n💡 SPECIFIC FIXES:")

        if coord_transform['lost_count'] > 5:
            print("  1. PERSPECTIVE TRANSFORMATION ISSUE:")
            print("     - The board corner detection may be inaccurate")
            print("     - Try using a more robust corner detection method")
            print("     - Consider adding margin to the warped board size")

        if assignment_loss > 5:
            print("  2. PIECE ASSIGNMENT ISSUE:")
            print("     - Check if piece coordinates are being calculated correctly")
            print("     - Verify square_size calculation (should be board_size/8)")
            print("     - Debug the piece symbol conversion function")

        print(f"\n🔧 IMMEDIATE DEBUG STEPS:")
        print("  1. Save visualization images of:")
        print("     - Original image with detected pieces marked")
        print("     - Warped board with transformed pieces marked")
        print("     - Final board with assigned squares highlighted")
        print("  2. Check if corner detection is accurate by visual inspection")
        print("  3. Verify the piece classification names match expected format")


def create_debug_visualization(debugger: CoordinateTransformationDebugger,
                               image_path: str, output_dir: str = "debug_output"):
    """Create visualization images for debugging"""

    Path(output_dir).mkdir(exist_ok=True)

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    image = ImageProcessor.load_image_from_bytes(image_bytes)

    # Get debug info
    debug_info = debugger.debug_coordinate_transformation(image_path)

    # Create visualizations...
    # (Implementation would create actual images showing the transformation steps)
    print(f"\n💾 Debug visualizations would be saved to: {output_dir}/")
    print("  - original_with_pieces.jpg")
    print("  - warped_board.jpg")
    print("  - final_assignments.jpg")


def main():
    parser = argparse.ArgumentParser(description="Debug chess coordinate transformation")
    parser.add_argument("image_path", help="Path to test chess image")
    parser.add_argument("--output", help="Save debug results to JSON file")
    parser.add_argument("--visualize", action="store_true", help="Create debug visualization images")

    args = parser.parse_args()

    if not Path(args.image_path).exists():
        print(f"❌ Image file not found: {args.image_path}")
        sys.exit(1)

    try:
        debugger = CoordinateTransformationDebugger()
        debug_results = debugger.debug_coordinate_transformation(args.image_path)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(debug_results, f, indent=2, default=str)
            print(f"\n💾 Debug results saved to: {args.output}")

        if args.visualize:
            create_debug_visualization(debugger, args.image_path)

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()