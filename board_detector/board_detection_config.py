
from dataclasses import dataclass
from typing import Tuple, List
import json
from pathlib import Path


@dataclass
class BoardDetectionConfig:
    """Configuration settings for chess board detection"""

    # Board size constraints
    min_board_area_ratio: float = 0.1  # Minimum board area as fraction of image
    max_board_area_ratio: float = 0.9  # Maximum board area as fraction of image
    min_aspect_ratio: float = 0.5  # Minimum width/height ratio
    max_aspect_ratio: float = 2.0  # Maximum width/height ratio

    # Checkerboard detection settings
    checkerboard_sizes: List[Tuple[int, int]] = None
    checkerboard_flags: int = None

    # Contour detection settings
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    canny_aperture: int = 3
    gaussian_blur_kernel: int = 5
    contour_epsilon_factor: float = 0.02

    # Line detection settings (Hough)
    hough_rho: int = 1
    hough_theta: float = None  # Will be set to np.pi/180
    hough_threshold: int = 100
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10
    horizontal_angle_tolerance: int = 20  # degrees
    vertical_angle_tolerance: int = 20  # degrees

    # Color clustering settings
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75
    adaptive_thresh_block_size: int = 15
    adaptive_thresh_c: int = 2
    min_component_area_ratio: float = 0.001
    max_component_area_ratio: float = 0.05
    min_components_required: int = 16

    # Fallback settings
    fallback_margin_ratio: float = 0.2  # Central area margin

    # Detection confidence settings
    confidence_area_weight: float = 0.4
    confidence_shape_weight: float = 0.6
    method_confidence_multipliers: dict = None

    # Visualization settings
    debug_enabled: bool = False
    visualization_line_thickness: int = 3
    visualization_point_radius: int = 8
    visualization_font_scale: float = 0.7

    def __post_init__(self):
        """Set default values that depend on imports"""
        import numpy as np

        if self.checkerboard_sizes is None:
            self.checkerboard_sizes = [(7, 7), (6, 6), (8, 8), (5, 5)]

        if self.checkerboard_flags is None:
            import cv2
            self.checkerboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

        if self.hough_theta is None:
            self.hough_theta = np.pi / 180

        if self.method_confidence_multipliers is None:
            self.method_confidence_multipliers = {
                'checkerboard_pattern': 1.0,
                'contour_detection': 0.8,
                'line_intersection': 0.7,
                'color_clustering': 0.6,
                'fallback_center': 0.3
            }

    @classmethod
    def from_json(cls, config_file: str) -> 'BoardDetectionConfig':
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, config_file: str):
        """Save configuration to JSON file"""
        # Convert to dict, excluding functions and complex objects
        config_dict = {}
        for key, value in self.__dict__.items():
            if not callable(value) and key != 'checkerboard_flags':
                if hasattr(value, 'tolist'):  # numpy arrays
                    config_dict[key] = value.tolist()
                else:
                    config_dict[key] = value

        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"📝 Configuration saved to: {config_file}")

    def validate(self) -> List[str]:
        """Validate configuration parameters"""
        issues = []

        if self.min_board_area_ratio >= self.max_board_area_ratio:
            issues.append("min_board_area_ratio must be less than max_board_area_ratio")

        if self.min_aspect_ratio >= self.max_aspect_ratio:
            issues.append("min_aspect_ratio must be less than max_aspect_ratio")

        if self.canny_low_threshold >= self.canny_high_threshold:
            issues.append("canny_low_threshold must be less than canny_high_threshold")

        if self.fallback_margin_ratio < 0 or self.fallback_margin_ratio > 0.5:
            issues.append("fallback_margin_ratio must be between 0 and 0.5")

        return issues


def create_default_config(config_file: str = "board_detection_config.json"):
    """Create a default configuration file"""

    config = BoardDetectionConfig()
    config.to_json(config_file)

    print(f"📝 Default configuration created: {config_file}")
    print("\nKey settings to customize:")
    print(f"  - min_board_area_ratio: {config.min_board_area_ratio}")
    print(f"  - max_board_area_ratio: {config.max_board_area_ratio}")
    print(f"  - debug_enabled: {config.debug_enabled}")


def create_tuning_configs():
    """Create preset configurations for different scenarios"""

    configs = {
        "high_precision": BoardDetectionConfig(
            min_board_area_ratio=0.2,
            max_board_area_ratio=0.8,
            canny_low_threshold=30,
            canny_high_threshold=100,
            method_confidence_multipliers={
                'checkerboard_pattern': 1.0,
                'contour_detection': 0.9,
                'line_intersection': 0.8,
                'color_clustering': 0.7,
                'fallback_center': 0.1  # Avoid fallback
            }
        ),

        "high_recall": BoardDetectionConfig(
            min_board_area_ratio=0.05,
            max_board_area_ratio=0.95,
            canny_low_threshold=20,
            canny_high_threshold=80,
            method_confidence_multipliers={
                'checkerboard_pattern': 1.0,
                'contour_detection': 0.8,
                'line_intersection': 0.7,
                'color_clustering': 0.6,
                'fallback_center': 0.5  # Allow fallback
            }
        ),

        "fast_detection": BoardDetectionConfig(
            checkerboard_sizes=[(7, 7), (6, 6)],  # Fewer sizes to try
            gaussian_blur_kernel=3,  # Less blur processing
            hough_threshold=150,  # Higher threshold (fewer lines)
            min_components_required=12,  # Fewer components needed
        )
    }

    for name, config in configs.items():
        config.to_json(f"board_detection_{name}.json")

    print("📝 Preset configurations created:")
    for name in configs.keys():
        print(f"  - board_detection_{name}.json")


# Usage example integrating config with detector
def create_detector_from_config(config_file: str) -> 'ChessBoardDetector':
    """Create a detector instance from configuration file"""

    if Path(config_file).exists():
        config = BoardDetectionConfig.from_json(config_file)

        # Validate configuration
        issues = config.validate()
        if issues:
            print("⚠️ Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")

        # Import here to avoid circular imports
        from chess_board_detector import ChessBoardDetector

        # Create detector with config parameters
        detector = ChessBoardDetector(
            min_board_area_ratio=config.min_board_area_ratio,
            max_board_area_ratio=config.max_board_area_ratio,
            debug=config.debug_enabled
        )

        # Apply additional config settings to detector
        detector.config = config

        return detector

    else:
        print(f"❌ Configuration file not found: {config_file}")
        print("Creating default configuration...")
        create_default_config(config_file)
        return create_detector_from_config(config_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Board Detection Configuration")
    parser.add_argument("--create-default", action="store_true",
                        help="Create default configuration file")
    parser.add_argument("--create-presets", action="store_true",
                        help="Create preset configurations")
    parser.add_argument("--validate", type=str,
                        help="Validate configuration file")

    args = parser.parse_args()

    if args.create_default:
        create_default_config()
    elif args.create_presets:
        create_tuning_configs()
    elif args.validate:
        if Path(args.validate).exists():
            config = BoardDetectionConfig.from_json(args.validate)
            issues = config.validate()
            if issues:
                print("❌ Configuration issues:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("✅ Configuration is valid")
        else:
            print(f"❌ Configuration file not found: {args.validate}")
    else:
        print("Creating default configuration and presets...")
        create_default_config()
        create_tuning_configs()