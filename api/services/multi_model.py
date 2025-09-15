# api/services/multi_model.py - Multi-model chess prediction pipeline service

import time
import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import cv2
from pathlib import Path

from api.services.base import ChessPredictionService
from api._helpers import (
    PredictionResult,
    ImageProcessor,
    FENValidator,
    ModelDownloader
)
from api.config import settings

logger = logging.getLogger(__name__)


class MultiModelPipelineService(ChessPredictionService):
    """
    Chess prediction service using multiple models in a pipeline:
    1. YOLO segmentation model to detect chess board
    2. YOLO object detection model to detect pieces
    3. Computer vision algorithms for board warping and piece assignment
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.segmentation_model = None
        self.pieces_model = None
        self.piece_classes = config.get('piece_classes', [
            'white-king', 'white-queen', 'white-rook', 'white-bishop', 'white-knight', 'white-pawn',
            'black-king', 'black-queen', 'black-rook', 'black-bishop', 'black-knight', 'black-pawn'
        ])

        # Load models on initialization
        if self._load_models():
            self.service_loaded = True
        else:
            logger.error("Failed to load multi-model pipeline")

    def _load_models(self) -> bool:
        """Load both segmentation and piece detection models"""
        try:
            # Import YOLO here to avoid import errors if not available
            try:
                from ultralytics import YOLO
            except ImportError:
                logger.error("ultralytics package not available. Install with: pip install ultralytics")
                return False

            # Load segmentation model for board detection
            seg_model_path = self.config.get('segmentation_model_path')
            if not seg_model_path:
                logger.error("segmentation_model_path not specified in config")
                return False

            logger.info(f"Loading segmentation model from: {seg_model_path}")
            self.segmentation_model = YOLO(seg_model_path)
            logger.info("Segmentation model loaded successfully")

            # Load pieces detection model
            pieces_model_path = self.config.get('pieces_model_path')
            if not pieces_model_path:
                logger.error("pieces_model_path not specified in config")
                return False

            logger.info(f"Loading pieces model from: {pieces_model_path}")
            self.pieces_model = YOLO(pieces_model_path)
            logger.info("Pieces model loaded successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to load multi-model pipeline: {e}")
            return False

    def predict_from_image(self, image_bytes: bytes) -> PredictionResult:
        """
        Main prediction method using multi-model pipeline:
        1. Load and preprocess image
        2. Detect chess board using segmentation model
        3. Detect pieces using object detection model
        4. Warp board and transform coordinates
        5. Assign pieces to squares
        6. Generate FEN notation
        """

        if not self.service_loaded:
            return PredictionResult(
                success=False,
                error_message="Multi-model pipeline not loaded properly"
            )

        processing_steps = {}
        start_time = time.time()

        try:
            # Step 1: Load and preprocess image
            step_start = time.time()
            image = ImageProcessor.load_image_from_bytes(image_bytes)
            if image is None:
                return PredictionResult(
                    success=False,
                    error_message="Failed to load image"
                )
            processing_steps['image_loading'] = time.time() - step_start

            # Step 2: Board detection using segmentation
            step_start = time.time()
            corners, seg_confidence, mask = self._detect_board_with_segmentation(image)
            processing_steps['board_segmentation'] = time.time() - step_start

            if not corners:
                return PredictionResult(
                    success=False,
                    board_detected=False,
                    error_message="Failed to detect chess board"
                )

            # Step 3: Piece detection on original image
            step_start = time.time()
            original_pieces = self._detect_pieces(image)
            processing_steps['piece_detection'] = time.time() - step_start

            # Step 4: Board warping using detected corners
            step_start = time.time()
            warped_board, transform_matrix, final_corners = self._warp_board_from_mask(
                image, mask, corners
            )
            processing_steps['board_warping'] = time.time() - step_start

            if warped_board is None:
                return PredictionResult(
                    success=False,
                    board_detected=True,
                    error_message="Failed to warp chess board"
                )

            # Step 5: Transform piece coordinates to warped space
            step_start = time.time()
            warped_pieces = self._transform_pieces_to_warped_space(
                original_pieces, transform_matrix, warped_board.shape[:2]
            )
            processing_steps['coordinate_transformation'] = time.time() - step_start

            # Step 6: Assign pieces to squares
            step_start = time.time()
            board_matrix = self._assign_pieces_to_squares(warped_pieces)
            processing_steps['piece_assignment'] = time.time() - step_start

            # Step 7: Generate FEN notation
            step_start = time.time()
            fen = FENValidator.board_matrix_to_fen(board_matrix)
            processing_steps['fen_generation'] = time.time() - step_start

            if fen is None:
                return PredictionResult(
                    success=False,
                    board_detected=True,
                    error_message="Failed to generate valid FEN",
                    processing_steps=processing_steps
                )

            processing_steps['total_time'] = time.time() - start_time

            # Calculate overall confidence (combine segmentation and piece detection)
            piece_confidences = [p['confidence'] for p in warped_pieces if p.get('confidence')]
            avg_piece_confidence = np.mean(piece_confidences) if piece_confidences else 0.5
            overall_confidence = float((seg_confidence + avg_piece_confidence) / 2)

            return PredictionResult(
                success=True,
                fen=fen,
                board_matrix=board_matrix,
                confidence=overall_confidence,
                board_detected=True,
                processing_steps=processing_steps
            )

        except Exception as e:
            logger.error(f"Multi-model pipeline error: {e}")
            return PredictionResult(
                success=False,
                error_message=f"Multi-model pipeline error: {str(e)}",
                processing_steps=processing_steps
            )

    def _detect_board_with_segmentation(self, image: np.ndarray) -> Tuple[Optional[Dict], float, Optional[np.ndarray]]:
        """Detect chess board using YOLO segmentation model - SCALE FIX VERSION"""
        try:
            # Get segmentation confidence threshold from config
            seg_confidence = self.config.get('segmentation_confidence', 0.3)

            # Run segmentation
            results = self.segmentation_model(image, conf=seg_confidence)

            if not results or not results[0].masks:
                logger.warning("No chess board detected in segmentation")
                return None, 0.0, None

            # Get the best detection
            best_result = results[0]
            confidence = float(best_result.boxes.conf[0]) if best_result.boxes.conf is not None else 0.5

            # Extract mask - CRITICAL: Check if mask needs scaling
            mask = best_result.masks.data[0].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)

            logger.info(f"Original image shape: {image.shape[:2]}")
            logger.info(f"Mask shape: {mask.shape}")

            # SCALE FIX: If mask is smaller than image, scale it up
            if mask.shape != image.shape[:2]:
                logger.info(f"Scaling mask from {mask.shape} to {image.shape[:2]}")
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                logger.info(f"Mask scaled to: {mask.shape}")

            # Extract corners from mask
            corners = self._extract_corners_from_mask(mask)

            # VALIDATION: Check if corners make sense relative to piece positions
            if corners:
                logger.info(f"Detected board corners: {corners}")

                # If we have piece detection results, validate against them
                sample_pieces = self._detect_pieces(image)
                if sample_pieces:
                    # Check if pieces are roughly within the detected board area
                    piece_coords = [(p['x_center'], p['y_center']) for p in sample_pieces[:5]]
                    logger.info(f"Sample piece positions: {piece_coords}")

                    # Simple validation: check if pieces are within board bounds
                    board_area_valid = self._validate_corners_against_pieces(corners, piece_coords)
                    if not board_area_valid:
                        logger.warning("Detected board corners don't align with piece positions")
                        # Fall back to using piece positions to estimate board area
                        estimated_corners = self._estimate_corners_from_pieces(sample_pieces, image.shape)
                        if estimated_corners:
                            logger.info(f"Using estimated corners from pieces: {estimated_corners}")
                            corners = estimated_corners

            return corners, confidence, mask

        except Exception as e:
            logger.error(f"Board segmentation failed: {e}")
            return None, 0.0, None

    def _validate_corners_against_pieces(self, corners: Dict, piece_coords: List[Tuple[float, float]]) -> bool:
        """Validate that board corners roughly contain the detected pieces"""
        try:
            if not corners or not piece_coords:
                return False

            # Get corner coordinates
            tl = corners['top_left']
            tr = corners['top_right']
            br = corners['bottom_right']
            bl = corners['bottom_left']

            # Calculate rough board bounds
            min_x = min(tl[0], tr[0], br[0], bl[0])
            max_x = max(tl[0], tr[0], br[0], bl[0])
            min_y = min(tl[1], tr[1], br[1], bl[1])
            max_y = max(tl[1], tr[1], br[1], bl[1])

            # Check how many pieces fall within these bounds (with margin)
            margin = 100  # Allow pieces slightly outside detected board
            pieces_inside = 0

            for x, y in piece_coords:
                if (min_x - margin <= x <= max_x + margin and
                        min_y - margin <= y <= max_y + margin):
                    pieces_inside += 1

            coverage_ratio = pieces_inside / len(piece_coords)
            logger.info(
                f"Board coverage: {pieces_inside}/{len(piece_coords)} pieces inside board ({coverage_ratio:.2%})")

            # Consider valid if at least 60% of pieces are within board bounds
            return coverage_ratio >= 0.6

        except Exception as e:
            logger.error(f"Corner validation failed: {e}")
            return False

    def _estimate_corners_from_pieces(self, pieces: List[Dict], image_shape: Tuple[int, int]) -> Optional[Dict]:
        """Estimate board corners based on piece positions as fallback"""
        try:
            if len(pieces) < 4:
                return None

            # Get all piece coordinates
            x_coords = [p['x_center'] for p in pieces]
            y_coords = [p['y_center'] for p in pieces]

            # Find bounds with some margin
            margin_x = (max(x_coords) - min(x_coords)) * 0.1  # 10% margin
            margin_y = (max(y_coords) - min(y_coords)) * 0.1

            min_x = max(0, min(x_coords) - margin_x)
            max_x = min(image_shape[1], max(x_coords) + margin_x)
            min_y = max(0, min(y_coords) - margin_y)
            max_y = min(image_shape[0], max(y_coords) + margin_y)

            # Create rectangular corners
            estimated_corners = {
                'top_left': (int(min_x), int(min_y)),
                'top_right': (int(max_x), int(min_y)),
                'bottom_right': (int(max_x), int(max_y)),
                'bottom_left': (int(min_x), int(max_y))
            }

            logger.info(f"Estimated corners from {len(pieces)} pieces: {estimated_corners}")
            return estimated_corners

        except Exception as e:
            logger.error(f"Failed to estimate corners from pieces: {e}")
            return None

    def _detect_pieces(self, image: np.ndarray) -> List[Dict]:
        """Detect chess pieces using YOLO object detection model"""
        try:
            # Get piece detection confidence threshold from config
            piece_confidence = self.config.get('piece_confidence', 0.25)
            iou_threshold = self.config.get('iou_threshold', 0.5)

            # Run piece detection
            results = self.pieces_model(image, conf=piece_confidence, iou=iou_threshold)

            detected_pieces = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        piece_info = {
                            'x_center': float(box.xywh[0][0]),
                            'y_center': float(box.xywh[0][1]),
                            'width': float(box.xywh[0][2]),
                            'height': float(box.xywh[0][3]),
                            'category_id': int(box.cls[0]),
                            'confidence': float(box.conf[0])
                        }

                        # Add piece class name if available
                        if hasattr(result, 'names') and piece_info['category_id'] in result.names:
                            piece_info['class_name'] = result.names[piece_info['category_id']]

                        detected_pieces.append(piece_info)

            logger.info(f"Detected {len(detected_pieces)} pieces")
            return detected_pieces

        except Exception as e:
            logger.error(f"Piece detection failed: {e}")
            return []

    def _extract_corners_from_mask(self, mask: np.ndarray) -> Optional[Dict]:
        """Extract board corners from segmentation mask - FIXED VERSION"""
        try:
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.warning("No contours found in mask")
                return None

            # Get the largest contour (should be the chess board)
            largest_contour = max(contours, key=cv2.contourArea)

            # Log contour area for debugging
            contour_area = cv2.contourArea(largest_contour)
            logger.info(f"Largest contour area: {contour_area}")

            # Try different epsilon values for polygon approximation
            for epsilon_factor in [0.02, 0.03, 0.01, 0.04]:
                epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                logger.info(f"Epsilon factor {epsilon_factor}: got {len(approx)} points")

                if len(approx) == 4:
                    # We have 4 corners, verify they are distinct
                    corners_array = approx.reshape(4, 2)

                    # Check if all corners are distinct (minimum distance between any two points)
                    min_distance = float('inf')
                    for i in range(4):
                        for j in range(i + 1, 4):
                            dist = np.linalg.norm(corners_array[i] - corners_array[j])
                            min_distance = min(min_distance, dist)

                    logger.info(f"Minimum distance between corners: {min_distance}")

                    # Only accept if corners are sufficiently far apart
                    if min_distance > 20:  # At least 20 pixels apart
                        corners = self._sort_corners_fixed(corners_array)

                        # Final validation - ensure corners make sense
                        if self._validate_corners(corners):
                            return corners
                        else:
                            logger.warning("Corner validation failed")
                    else:
                        logger.warning(f"Corners too close together (min distance: {min_distance})")

            # If we couldn't find 4 distinct corners, use a more robust method
            logger.info("Could not find 4 distinct corners, using alternative method")
            return self._extract_corners_alternative(largest_contour)

        except Exception as e:
            logger.error(f"Corner extraction failed: {e}")
            return None

    def _sort_corners_fixed(self, corners: np.ndarray) -> Dict:
        """Sort corners to top_left, top_right, bottom_right, bottom_left - FIXED VERSION"""

        # Convert to float for calculations
        corners = corners.astype(np.float32)

        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        logger.info(f"Corners centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")

        # Sort by angle from centroid
        def angle_from_centroid(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

        # Sort corners by angle (starting from top-right, going clockwise)
        sorted_corners = sorted(corners, key=angle_from_centroid)

        # The first corner (smallest angle) should be roughly top-right
        # We'll identify corners by their relative position to centroid
        labeled_corners = []
        for corner in sorted_corners:
            x, y = corner
            cx, cy = centroid

            # Classify corner based on position relative to centroid
            if x >= cx and y <= cy:  # Right and above
                labeled_corners.append(('top_right', corner))
            elif x >= cx and y >= cy:  # Right and below
                labeled_corners.append(('bottom_right', corner))
            elif x <= cx and y >= cy:  # Left and below
                labeled_corners.append(('bottom_left', corner))
            elif x <= cx and y <= cy:  # Left and above
                labeled_corners.append(('top_left', corner))
            else:
                # Fallback - shouldn't happen
                labeled_corners.append(('unknown', corner))

        # Create result dictionary
        result = {}
        for label, corner in labeled_corners:
            if label != 'unknown':
                result[label] = tuple(map(int, corner))

        # Ensure we have all 4 corners
        required_corners = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        if not all(corner in result for corner in required_corners):
            logger.warning("Missing some corner labels, falling back to simple sort")
            # Fallback to simple sorting
            sorted_by_y = sorted(corners, key=lambda p: p[1])
            top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])  # Sort top two by x
            bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])  # Sort bottom two by x

            result = {
                'top_left': tuple(map(int, top_two[0])),
                'top_right': tuple(map(int, top_two[1])),
                'bottom_left': tuple(map(int, bottom_two[0])),
                'bottom_right': tuple(map(int, bottom_two[1]))
            }

        logger.info(f"Final corners: {result}")
        return result

    def _validate_corners(self, corners: Dict) -> bool:
        """Validate that detected corners make geometric sense"""
        try:
            tl = np.array(corners['top_left'])
            tr = np.array(corners['top_right'])
            br = np.array(corners['bottom_right'])
            bl = np.array(corners['bottom_left'])

            # Check that corners form a reasonable quadrilateral
            # 1. No corner should be identical to another
            points = [tl, tr, br, bl]
            for i, p1 in enumerate(points):
                for j, p2 in enumerate(points[i + 1:], i + 1):
                    if np.linalg.norm(p1 - p2) < 10:  # Too close
                        logger.warning(f"Corners {i} and {j} too close: {np.linalg.norm(p1 - p2):.1f}")
                        return False

            # 2. Check ordering makes sense (top corners should be above bottom corners)
            if tl[1] >= bl[1] or tr[1] >= br[1]:
                logger.warning("Top corners not above bottom corners")
                return False

            # 3. Check left corners are to the left of right corners
            if tl[0] >= tr[0] or bl[0] >= br[0]:
                logger.warning("Left corners not to the left of right corners")
                return False

            # 4. Check the quadrilateral isn't too skewed
            # Calculate the area to ensure it's reasonable
            def quad_area(p1, p2, p3, p4):
                # Shoelace formula
                return 0.5 * abs((p1[0] * (p2[1] - p4[1]) + p2[0] * (p3[1] - p1[1]) +
                                  p3[0] * (p4[1] - p2[1]) + p4[0] * (p1[1] - p3[1])))

            area = quad_area(tl, tr, br, bl)
            if area < 1000:  # Too small
                logger.warning(f"Quadrilateral area too small: {area}")
                return False

            logger.info(f"Corner validation passed. Quadrilateral area: {area}")
            return True

        except Exception as e:
            logger.error(f"Corner validation error: {e}")
            return False

    def _extract_corners_alternative(self, contour: np.ndarray) -> Optional[Dict]:
        """Alternative corner extraction method using convex hull and extreme points"""
        try:
            # Get convex hull
            hull = cv2.convexHull(contour)

            # Find extreme points
            leftmost = tuple(hull[hull[:, :, 0].argmin()][0])
            rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
            topmost = tuple(hull[hull[:, :, 1].argmin()][0])
            bottommost = tuple(hull[hull[:, :, 1].argmax()][0])

            # Find corners by combining extreme points
            # This is a simple heuristic - might need refinement
            corners = {
                'top_left': leftmost if leftmost[1] < (topmost[1] + bottommost[1]) / 2 else topmost,
                'top_right': rightmost if rightmost[1] < (topmost[1] + bottommost[1]) / 2 else topmost,
                'bottom_left': leftmost if leftmost[1] >= (topmost[1] + bottommost[1]) / 2 else bottommost,
                'bottom_right': rightmost if rightmost[1] >= (topmost[1] + bottommost[1]) / 2 else bottommost
            }

            logger.info(f"Alternative corner detection result: {corners}")

            # Validate alternative corners
            if self._validate_corners(corners):
                return corners
            else:
                logger.warning("Alternative corner detection also failed validation")

                # Final fallback: use bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                fallback_corners = {
                    'top_left': (x, y),
                    'top_right': (x + w, y),
                    'bottom_right': (x + w, y + h),
                    'bottom_left': (x, y + h)
                }
                logger.info(f"Using bounding rectangle fallback: {fallback_corners}")
                return fallback_corners

        except Exception as e:
            logger.error(f"Alternative corner extraction failed: {e}")
            return None

    def _sort_corners(self, corners: np.ndarray) -> Dict:
        """Sort corners to top_left, top_right, bottom_right, bottom_left - DEPRECATED, use _sort_corners_fixed"""
        logger.warning("Using deprecated _sort_corners method, consider using _sort_corners_fixed")
        return self._sort_corners_fixed(corners)

    def _warp_board_from_mask(self, image: np.ndarray, mask: np.ndarray, corners: Dict,
                              output_size: int = 800) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """Warp the chess board to a square format using detected corners - IMPROVED VERSION"""
        try:
            # Define source points from corners
            src_points = np.array([
                corners['top_left'],
                corners['top_right'],
                corners['bottom_right'],
                corners['bottom_left']
            ], dtype=np.float32)

            # Define destination points (square board)
            dst_points = np.array([
                [0, 0],
                [output_size, 0],
                [output_size, output_size],
                [0, output_size]
            ], dtype=np.float32)

            # VALIDATION: Test the perspective transformation with a few test points
            logger.info(f"Source corners: {src_points.tolist()}")
            logger.info(f"Destination corners: {dst_points.tolist()}")

            # Calculate perspective transform matrix
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            logger.info(f"Transform matrix computed successfully")

            # TEST the transformation with the source corners themselves
            test_points = src_points.reshape(-1, 1, 2)
            transformed_test = cv2.perspectiveTransform(test_points, transform_matrix)

            logger.info("Testing perspective transformation:")
            for i, (orig, trans) in enumerate(zip(src_points, transformed_test.reshape(-1, 2))):
                logger.info(f"  Corner {i}: ({orig[0]:.1f},{orig[1]:.1f}) → ({trans[0]:.1f},{trans[1]:.1f})")

            # Check if transformation is reasonable
            transformed_test_flat = transformed_test.reshape(-1, 2)
            if (np.any(transformed_test_flat < -50) or
                    np.any(transformed_test_flat > output_size + 50)):
                logger.error("Perspective transformation appears invalid - corners transform outside reasonable bounds")

                # Try to fix corner ordering
                logger.info("Attempting to fix corner ordering...")
                fixed_corners = self._fix_corner_ordering(corners, image.shape)
                if fixed_corners:
                    logger.info(f"Trying with reordered corners: {fixed_corners}")
                    src_points_fixed = np.array([
                        fixed_corners['top_left'],
                        fixed_corners['top_right'],
                        fixed_corners['bottom_right'],
                        fixed_corners['bottom_left']
                    ], dtype=np.float32)

                    transform_matrix = cv2.getPerspectiveTransform(src_points_fixed, dst_points)
                    src_points = src_points_fixed  # Use the fixed points
                    corners = fixed_corners  # Update corners

                    # Test again
                    test_points = src_points.reshape(-1, 1, 2)
                    transformed_test = cv2.perspectiveTransform(test_points, transform_matrix)
                    logger.info("Retest after corner reordering:")
                    for i, (orig, trans) in enumerate(zip(src_points, transformed_test.reshape(-1, 2))):
                        logger.info(f"  Corner {i}: ({orig[0]:.1f},{orig[1]:.1f}) → ({trans[0]:.1f},{trans[1]:.1f})")

            # Apply perspective warp
            warped_board = cv2.warpPerspective(image, transform_matrix, (output_size, output_size))

            return warped_board, transform_matrix, corners

        except Exception as e:
            logger.error(f"Board warping failed: {e}")
            return None, None, corners

    def _transform_pieces_to_warped_space(self, pieces: List[Dict], transform_matrix: np.ndarray,
                                          warped_shape: Tuple[int, int]) -> List[Dict]:
        """Transform piece coordinates from original image to warped board space - IMPROVED VERSION"""
        if transform_matrix is None:
            return []

        warped_pieces = []
        out_of_bounds_count = 0

        # First, let's test the transformation matrix with a known point
        center_x, center_y = warped_shape[1] // 2, warped_shape[0] // 2
        test_points = np.array([[[100, 100], [200, 200], [center_x, center_y]]], dtype=np.float32)

        try:
            transformed_test = cv2.perspectiveTransform(test_points, transform_matrix)
            logger.info("Transform matrix test:")
            for i, (orig, trans) in enumerate(zip(test_points[0], transformed_test[0])):
                logger.info(f"  Test point {i}: ({orig[0]:.1f},{orig[1]:.1f}) → ({trans[0]:.1f},{trans[1]:.1f})")
        except Exception as e:
            logger.error(f"Transform matrix appears invalid: {e}")
            return []

        for piece in pieces:
            try:
                # Get original coordinates
                x, y = piece['x_center'], piece['y_center']

                # Transform coordinates
                point = np.array([[[x, y]]], dtype=np.float32)
                transformed_point = cv2.perspectiveTransform(point, transform_matrix)

                new_x, new_y = transformed_point[0][0]

                # More generous margin for boundary pieces
                margin = 10
                if -margin <= new_x < warped_shape[1] + margin and -margin <= new_y < warped_shape[0] + margin:
                    # Clamp to valid bounds
                    new_x = max(0, min(warped_shape[1] - 1, new_x))
                    new_y = max(0, min(warped_shape[0] - 1, new_y))

                    warped_piece = piece.copy()
                    warped_piece['x_center'] = float(new_x)
                    warped_piece['y_center'] = float(new_y)
                    warped_pieces.append(warped_piece)
                else:
                    out_of_bounds_count += 1
                    if out_of_bounds_count <= 5:  # Log first 5 out-of-bounds pieces
                        logger.warning(
                            f"Piece {piece.get('class_name', 'unknown')} at ({x:.1f}, {y:.1f}) → ({new_x:.1f}, {new_y:.1f}) [OUT OF BOUNDS]")

            except Exception as e:
                logger.warning(f"Failed to transform piece coordinates: {e}")
                continue

        logger.info(
            f"Transformed {len(pieces)} pieces: {len(warped_pieces)} in bounds, {out_of_bounds_count} out of bounds")
        return warped_pieces

    def _assign_pieces_to_squares(self, pieces: List[Dict], board_size: int = 800) -> List[List[str]]:
        """Assign detected pieces to chess board squares"""
        # Initialize 8x8 board
        board_matrix = [[''] * 8 for _ in range(8)]

        # Calculate square size
        square_size = board_size / 8

        # Sort pieces by confidence (highest first) to handle conflicts better
        pieces_sorted = sorted(pieces, key=lambda p: p.get('confidence', 0), reverse=True)

        assigned_count = 0
        conflict_count = 0

        for piece in pieces_sorted:
            try:
                # Determine which square this piece belongs to
                col = int(piece['x_center'] // square_size)
                row = int(piece['y_center'] // square_size)

                # Ensure coordinates are within bounds
                if 0 <= row < 8 and 0 <= col < 8:
                    # Convert piece category to chess notation
                    piece_symbol = self._convert_piece_category_to_symbol(piece)
                    if piece_symbol:
                        # Check if square is already occupied
                        if board_matrix[row][col] == '':
                            board_matrix[row][col] = piece_symbol
                            assigned_count += 1
                        else:
                            # Handle conflict - keep piece with higher confidence
                            conflict_count += 1
                            logger.debug(
                                f"Square conflict at ({row},{col}): existing '{board_matrix[row][col]}' vs new '{piece_symbol}' (conf: {piece.get('confidence', 0):.3f})")
                    else:
                        logger.debug(f"Failed to convert piece to symbol: {piece.get('class_name', 'unknown')}")
                else:
                    logger.debug(f"Piece coordinates out of bounds: row={row}, col={col}")

            except Exception as e:
                logger.warning(f"Failed to assign piece to square: {e}")
                continue

        logger.info(f"Assigned {assigned_count} pieces to board, {conflict_count} conflicts")
        return board_matrix

    def _fix_corner_ordering(self, corners: Dict, image_shape: Tuple[int, int]) -> Optional[Dict]:
        """Try to fix corner ordering by analyzing the actual positions"""
        try:
            # Extract all corner points
            points = [
                (corners['top_left'], 'top_left'),
                (corners['top_right'], 'top_right'),
                (corners['bottom_right'], 'bottom_right'),
                (corners['bottom_left'], 'bottom_left')
            ]

            # Sort by Y coordinate first (top to bottom)
            points_by_y = sorted(points, key=lambda p: p[0][1])

            # Get top 2 and bottom 2 points
            top_two = points_by_y[:2]
            bottom_two = points_by_y[2:]

            # Sort top two by X coordinate (left to right)
            top_sorted = sorted(top_two, key=lambda p: p[0][0])
            # Sort bottom two by X coordinate (left to right)
            bottom_sorted = sorted(bottom_two, key=lambda p: p[0][0])

            # Assign corrected labels
            new_corners = {
                'top_left': top_sorted[0][0],  # leftmost of top two
                'top_right': top_sorted[1][0],  # rightmost of top two
                'bottom_left': bottom_sorted[0][0],  # leftmost of bottom two
                'bottom_right': bottom_sorted[1][0]  # rightmost of bottom two
            }

            logger.info(f"Reordered corners: {new_corners}")
            return new_corners

        except Exception as e:
            logger.error(f"Failed to fix corner ordering: {e}")
            return None

    def _convert_piece_category_to_symbol(self, piece: Dict) -> str:
        """Convert detected piece category to FEN notation symbol"""
        try:
            # Get class name from piece info
            class_name = piece.get('class_name', '')

            if not class_name and 'category_id' in piece:
                # Fallback: use category_id to index into piece_classes
                category_id = piece['category_id']
                if 0 <= category_id < len(self.piece_classes):
                    class_name = self.piece_classes[category_id]

            # Map class name to FEN symbol
            name_to_symbol = {
                # Standard naming
                'white-king': 'K', 'white-queen': 'Q', 'white-rook': 'R',
                'white-bishop': 'B', 'white-knight': 'N', 'white-pawn': 'P',
                'black-king': 'k', 'black-queen': 'q', 'black-rook': 'r',
                'black-bishop': 'b', 'black-knight': 'n', 'black-pawn': 'p',

                # Alternative naming patterns
                'white_king': 'K', 'white_queen': 'Q', 'white_rook': 'R',
                'white_bishop': 'B', 'white_knight': 'N', 'white_pawn': 'P',
                'black_king': 'k', 'black_queen': 'q', 'black_rook': 'r',
                'black_bishop': 'b', 'black_knight': 'n', 'black_pawn': 'p',
            }

            symbol = name_to_symbol.get(class_name.lower(), '')

            if not symbol:
                logger.debug(f"Unknown piece class: '{class_name}'")

            return symbol

        except Exception as e:
            logger.warning(f"Failed to convert piece category: {e}")
            return ''