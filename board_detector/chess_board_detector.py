# chess_board_detector.py - Robust chess board detection for raw images

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from pathlib import Path
import math
from skimage import measure, morphology
from scipy import ndimage


@dataclass
class BoardDetection:
    """Result of board detection"""
    corners: np.ndarray  # 4x2 array of corner coordinates
    confidence: float  # Detection confidence [0, 1]
    method: str  # Detection method used
    angle: float  # Board rotation angle in degrees
    center: Tuple[int, int]  # Board center coordinates


class ChessBoardDetector:
    """
    Multi-method chess board detector for raw images

    Implements multiple detection strategies:
    1. Contour-based detection
    2. Checkerboard pattern matching
    3. Line intersection detection
    4. Color clustering analysis
    """

    def __init__(self,
                 min_board_area_ratio: float = 0.1,
                 max_board_area_ratio: float = 0.9,
                 debug: bool = False):
        """
        Initialize chess board detector

        Args:
            min_board_area_ratio: Minimum board area as fraction of image
            max_board_area_ratio: Maximum board area as fraction of image
            debug: Enable debug visualizations
        """
        self.min_board_area_ratio = min_board_area_ratio
        self.max_board_area_ratio = max_board_area_ratio
        self.debug = debug

        # Detection method priorities (tried in order)
        self.detection_methods = [
            self._detect_by_checkerboard_pattern,
            self._detect_by_contours,
            self._detect_by_line_intersection,
            self._detect_by_color_clustering,
            self._detect_fallback
        ]

    def detect_board(self, image: np.ndarray) -> Optional[BoardDetection]:
        """
        Main detection method - tries multiple approaches

        Args:
            image: Input RGB image

        Returns:
            BoardDetection object if successful, None otherwise
        """
        if self.debug:
            print("🔍 Starting chess board detection...")

        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Try each detection method in priority order
        for method in self.detection_methods:
            try:
                result = method(image, gray)
                if result is not None:
                    if self.debug:
                        print(f"✅ Board detected using: {result.method}")
                        self._visualize_detection(image, result)
                    return result
            except Exception as e:
                if self.debug:
                    print(f"❌ Method {method.__name__} failed: {e}")
                continue

        if self.debug:
            print("❌ All detection methods failed")
        return None

    def _detect_by_checkerboard_pattern(self,
                                        image: np.ndarray,
                                        gray: np.ndarray) -> Optional[BoardDetection]:
        """
        Detect board using OpenCV's checkerboard pattern detection
        """
        # Try different checkerboard sizes
        board_sizes = [(7, 7), (6, 6), (8, 8), (5, 5)]

        for board_size in board_sizes:
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret and corners is not None:
                # Refine corner coordinates
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
                )

                # Extract board corners from internal corners
                board_corners = self._extract_board_corners_from_pattern(corners, board_size)

                if board_corners is not None:
                    confidence = self._calculate_board_confidence(board_corners, image.shape)
                    angle = self._calculate_board_angle(board_corners)
                    center = self._calculate_board_center(board_corners)

                    return BoardDetection(
                        corners=board_corners,
                        confidence=confidence,
                        method="checkerboard_pattern",
                        angle=angle,
                        center=center
                    )

        return None

    def _detect_by_contours(self,
                            image: np.ndarray,
                            gray: np.ndarray) -> Optional[BoardDetection]:
        """
        Detect board by finding rectangular contours
        """
        # Preprocessing for better contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multiple edge detection approaches
        edges_methods = [
            cv2.Canny(blurred, 50, 150, apertureSize=3),
            cv2.Canny(blurred, 30, 100, apertureSize=3),
            cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
        ]

        for edges in edges_methods:
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            image_area = image.shape[0] * image.shape[1]
            min_area = image_area * self.min_board_area_ratio
            max_area = image_area * self.max_board_area_ratio

            for contour in contours:
                area = cv2.contourArea(contour)

                if min_area < area < max_area:
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # Check if it's roughly rectangular (4 corners)
                    if len(approx) >= 4:
                        # Take the 4 most extreme points
                        corners = self._get_rectangle_corners(approx.reshape(-1, 2))

                        if self._validate_board_shape(corners, image.shape):
                            confidence = self._calculate_board_confidence(corners, image.shape)
                            angle = self._calculate_board_angle(corners)
                            center = self._calculate_board_center(corners)

                            return BoardDetection(
                                corners=corners,
                                confidence=confidence * 0.8,  # Lower confidence for contour method
                                method="contour_detection",
                                angle=angle,
                                center=center
                            )

        return None

    def _detect_by_line_intersection(self,
                                     image: np.ndarray,
                                     gray: np.ndarray) -> Optional[BoardDetection]:
        """
        Detect board by finding grid lines using Hough transforms
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100, minLineLength=50, maxLineGap=10
        )

        if lines is None or len(lines) < 8:  # Need at least 8 lines for a grid
            return None

        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(angle) < 20 or abs(angle) > 160:  # Horizontal-ish
                horizontal_lines.append(line[0])
            elif 70 < abs(angle) < 110:  # Vertical-ish
                vertical_lines.append(line[0])

        # Need multiple lines in each direction
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None

        # Find line intersections to determine board corners
        corners = self._find_grid_corners(horizontal_lines, vertical_lines, image.shape)

        if corners is not None and self._validate_board_shape(corners, image.shape):
            confidence = self._calculate_board_confidence(corners, image.shape)
            angle = self._calculate_board_angle(corners)
            center = self._calculate_board_center(corners)

            return BoardDetection(
                corners=corners,
                confidence=confidence * 0.7,  # Lower confidence for line method
                method="line_intersection",
                angle=angle,
                center=center
            )

        return None

    def _detect_by_color_clustering(self,
                                    image: np.ndarray,
                                    gray: np.ndarray) -> Optional[BoardDetection]:
        """
        Detect board by analyzing alternating light/dark square patterns
        """
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Use adaptive thresholding to find checkerboard pattern
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 2
        )

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        # Filter components by size (should be roughly square-sized)
        image_area = image.shape[0] * image.shape[1]
        min_square_area = image_area * 0.001  # Very small squares
        max_square_area = image_area * 0.05  # Very large squares

        valid_components = []
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if min_square_area < area < max_square_area:
                valid_components.append(i)

        if len(valid_components) < 16:  # Need at least 16 squares visible
            return None

        # Try to find rectangular arrangement of components
        component_centers = [centroids[i] for i in valid_components]
        corners = self._find_board_from_components(component_centers, image.shape)

        if corners is not None and self._validate_board_shape(corners, image.shape):
            confidence = self._calculate_board_confidence(corners, image.shape)
            angle = self._calculate_board_angle(corners)
            center = self._calculate_board_center(corners)

            return BoardDetection(
                corners=corners,
                confidence=confidence * 0.6,  # Lower confidence for clustering method
                method="color_clustering",
                angle=angle,
                center=center
            )

        return None

    def _detect_fallback(self,
                         image: np.ndarray,
                         gray: np.ndarray) -> Optional[BoardDetection]:
        """
        Fallback detection - assume board occupies central region
        """
        h, w = image.shape[:2]

        # Assume board takes up central 60% of image
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)

        corners = np.array([
            [margin_x, margin_y],  # Top-left
            [w - margin_x, margin_y],  # Top-right
            [w - margin_x, h - margin_y],  # Bottom-right
            [margin_x, h - margin_y]  # Bottom-left
        ], dtype=np.float32)

        return BoardDetection(
            corners=corners,
            confidence=0.3,  # Very low confidence
            method="fallback_center",
            angle=0.0,
            center=(w // 2, h // 2)
        )

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _extract_board_corners_from_pattern(self,
                                            corners: np.ndarray,
                                            board_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract outer board corners from internal checkerboard corners"""
        if corners is None:
            return None

        corners = corners.reshape(-1, 2)

        # Find extreme points
        top_left = corners[np.argmin(corners[:, 0] + corners[:, 1])]
        top_right = corners[np.argmax(corners[:, 0] - corners[:, 1])]
        bottom_right = corners[np.argmax(corners[:, 0] + corners[:, 1])]
        bottom_left = corners[np.argmin(corners[:, 0] - corners[:, 1])]

        # Estimate square size and expand to board edges
        square_size = np.mean([
            np.linalg.norm(top_right - top_left) / board_size[0],
            np.linalg.norm(bottom_left - top_left) / board_size[1]
        ])

        # Expand by half square size in each direction
        expansion = square_size * 0.5

        board_corners = np.array([
            top_left - [expansion, expansion],
            top_right + [expansion, -expansion],
            bottom_right + [expansion, expansion],
            bottom_left + [-expansion, expansion]
        ], dtype=np.float32)

        return board_corners

    def _get_rectangle_corners(self, points: np.ndarray) -> np.ndarray:
        """Get the 4 corner points of a rectangle from a set of points"""
        # Find the 4 extreme points
        top_left = points[np.argmin(points[:, 0] + points[:, 1])]
        top_right = points[np.argmax(points[:, 0] - points[:, 1])]
        bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
        bottom_left = points[np.argmin(points[:, 0] - points[:, 1])]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def _find_grid_corners(self,
                           horizontal_lines: List,
                           vertical_lines: List,
                           image_shape: Tuple) -> Optional[np.ndarray]:
        """Find board corners from grid lines"""

        def line_intersection(line1, line2):
            """Find intersection point of two lines"""
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return None

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            return (x, y)

        # Find all intersections
        intersections = []
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = line_intersection(h_line, v_line)
                if intersection is not None:
                    x, y = intersection
                    if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                        intersections.append([x, y])

        if len(intersections) < 4:
            return None

        intersections = np.array(intersections)

        # Find the 4 extreme intersection points
        corners = self._get_rectangle_corners(intersections)
        return corners

    def _find_board_from_components(self,
                                    centers: List,
                                    image_shape: Tuple) -> Optional[np.ndarray]:
        """Find board corners from component centers (checkerboard squares)"""
        if len(centers) < 4:
            return None

        centers = np.array(centers)

        # Find bounding box of all component centers
        min_x, min_y = np.min(centers, axis=0)
        max_x, max_y = np.max(centers, axis=0)

        # Expand slightly to include full board
        width = max_x - min_x
        height = max_y - min_y
        expansion_x = width * 0.1
        expansion_y = height * 0.1

        corners = np.array([
            [min_x - expansion_x, min_y - expansion_y],
            [max_x + expansion_x, min_y - expansion_y],
            [max_x + expansion_x, max_y + expansion_y],
            [min_x - expansion_x, max_y + expansion_y]
        ], dtype=np.float32)

        return corners

    def _validate_board_shape(self, corners: np.ndarray, image_shape: Tuple) -> bool:
        """Validate that detected corners form a reasonable board shape"""
        if corners is None or len(corners) != 4:
            return False

        # Check if all corners are within image bounds
        h, w = image_shape[:2]
        for corner in corners:
            if corner[0] < 0 or corner[0] >= w or corner[1] < 0 or corner[1] >= h:
                return False

        # Calculate area
        area = cv2.contourArea(corners)
        image_area = h * w

        if not (self.min_board_area_ratio * image_area < area < self.max_board_area_ratio * image_area):
            return False

        # Check aspect ratio (should be roughly square)
        rect = cv2.minAreaRect(corners)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height)

        if aspect_ratio > 2.0:  # Too rectangular
            return False

        return True

    def _calculate_board_confidence(self, corners: np.ndarray, image_shape: Tuple) -> float:
        """Calculate confidence score for detected board"""
        if corners is None:
            return 0.0

        confidence = 1.0

        # Area-based confidence
        area = cv2.contourArea(corners)
        image_area = image_shape[0] * image_shape[1]
        area_ratio = area / image_area

        # Prefer boards that take up reasonable portion of image
        if 0.2 <= area_ratio <= 0.7:
            confidence *= 1.0
        elif 0.1 <= area_ratio < 0.2 or 0.7 < area_ratio <= 0.9:
            confidence *= 0.8
        else:
            confidence *= 0.5

        # Shape-based confidence
        rect = cv2.minAreaRect(corners)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height)

        if aspect_ratio <= 1.2:  # Very square
            confidence *= 1.0
        elif aspect_ratio <= 1.5:
            confidence *= 0.9
        else:
            confidence *= 0.7

        return min(confidence, 1.0)

    def _calculate_board_angle(self, corners: np.ndarray) -> float:
        """Calculate board rotation angle in degrees"""
        if corners is None or len(corners) < 4:
            return 0.0

        # Use top edge to calculate angle
        top_left, top_right = corners[0], corners[1]
        dx = top_right[0] - top_left[0]
        dy = top_right[1] - top_left[1]

        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle

    def _calculate_board_center(self, corners: np.ndarray) -> Tuple[int, int]:
        """Calculate board center coordinates"""
        if corners is None:
            return (0, 0)

        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])

        return (int(center_x), int(center_y))

    def _visualize_detection(self, image: np.ndarray, detection: BoardDetection):
        """Visualize detection results for debugging"""
        img_vis = image.copy()

        # Draw detected corners
        corners = detection.corners.astype(int)

        # Draw board outline
        cv2.polylines(img_vis, [corners], True, (0, 255, 0), 3)

        # Draw corner points
        for i, corner in enumerate(corners):
            cv2.circle(img_vis, tuple(corner), 8, (255, 0, 0), -1)
            cv2.putText(img_vis, str(i), (corner[0] + 10, corner[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw center point
        cv2.circle(img_vis, detection.center, 5, (0, 0, 255), -1)

        # Add text information
        info_text = [
            f"Method: {detection.method}",
            f"Confidence: {detection.confidence:.3f}",
            f"Angle: {detection.angle:.1f}°"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(img_vis, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Board Detection: {detection.method} (conf: {detection.confidence:.3f})")
        plt.axis('off')
        plt.show()


def test_board_detector():
    """Test the board detector with sample images"""

    # Initialize detector
    detector = ChessBoardDetector(debug=True)

    print("🧪 Testing Chess Board Detector")
    print("=" * 50)

    # Test with sample image (you'll need to provide actual image path)
    test_image_path = "sample_chess_image.jpg"  # Replace with actual path

    if Path(test_image_path).exists():
        # Load test image
        image = cv2.imread(test_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"📸 Testing with image: {test_image_path}")
        print(f"Image size: {image.shape}")

        # Detect board
        detection = detector.detect_board(image)

        if detection:
            print("✅ Board detection successful!")
            print(f"Method: {detection.method}")
            print(f"Confidence: {detection.confidence:.3f}")
            print(f"Corners: \n{detection.corners}")
            print(f"Center: {detection.center}")
            print(f"Angle: {detection.angle:.1f}°")
        else:
            print("❌ Board detection failed!")

    else:
        print(f"❌ Test image not found: {test_image_path}")
        print("Please provide a chess board image to test")


if __name__ == "__main__":
    test_board_detector()