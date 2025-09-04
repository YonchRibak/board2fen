#!/usr/bin/env python3
"""
Test script for ChessReD preprocessing pipeline.

This script tests the functions in preprocess.py by:
1. Downloading real ChessReD data
2. Testing chess position conversion
3. Creating board matrices using preprocessing functions
4. Validating results against expected values
5. Visualizing sample boards

Run this to verify your preprocessing pipeline works correctly.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import functions from preprocess.py
try:
    from end2end_board_classifier_new_data.preprocess import (
        chess_position_to_index,
        _create_label_from_pieces,
        _download_file_with_retry,
        GCS_ANNOTATIONS_URL,
        LOCAL_CACHE_DIR,
        CLASS_NAMES,
        EMPTY_CLASS_ID
    )
except ImportError as e:
    print(f"[ERROR] Cannot import from preprocess.py: {e}")
    print("Make sure preprocess.py is in the same directory as this test script.")
    sys.exit(1)


def visualize_board(board: List[int], categories: List[Dict]) -> None:
    """Visualize the chess board with piece symbols using preprocessing results."""
    # Create category lookup
    cat_lookup = {cat['id']: cat['name'] for cat in categories}

    # Piece symbols mapping
    piece_symbols = {
        'white-pawn': '♙', 'white-rook': '♖', 'white-knight': '♘',
        'white-bishop': '♗', 'white-queen': '♕', 'white-king': '♔',
        'black-pawn': '♟', 'black-rook': '♜', 'black-knight': '♞',
        'black-bishop': '♝', 'black-queen': '♛', 'black-king': '♚',
        'empty': '·'
    }

    print("    a b c d e f g h")
    print("    ----------------")

    for row in range(8):
        print(f"{8 - row} | ", end="")
        for col in range(8):
            square_idx = row * 8 + col
            piece_id = board[square_idx]
            piece_name = cat_lookup.get(piece_id, 'empty')
            symbol = piece_symbols.get(piece_name, '?')
            print(f"{symbol} ", end="")
        print(f"| {8 - row}")

    print("    ----------------")
    print("    a b c d e f g h")


def test_chess_position_conversion() -> bool:
    """Test the chess_position_to_index function from preprocess.py."""
    print("=" * 60)
    print("TESTING CHESS POSITION CONVERSION")
    print("=" * 60)

    # Test cases: (position, expected_index)
    test_cases = [
        ('a8', 0), ('b8', 1), ('c8', 2), ('d8', 3),
        ('e8', 4), ('f8', 5), ('g8', 6), ('h8', 7),
        ('a7', 8), ('h7', 15),
        ('a1', 56), ('h1', 63),
        ('e4', 36), ('d5', 27),  # Fixed: e4 should be 36, not 28
        # Invalid cases
        ('i8', None), ('a9', None), ('z1', None), ('a', None)
    ]

    all_passed = True

    for position, expected in test_cases:
        result = chess_position_to_index(position)
        status = "PASS" if result == expected else "FAIL"

        if status == "FAIL":
            all_passed = False

        # Handle None values in formatting
        result_str = "None" if result is None else str(result)
        expected_str = "None" if expected is None else str(expected)
        print(f"  {position:>3} -> {result_str:>4} (expected {expected_str:>4}) [{status}]")

    print(f"\nPosition conversion test: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_board_creation_pipeline(chessred_data: Dict[str, Any]) -> bool:
    """Test the board creation using preprocess.py functions."""
    print("\n" + "=" * 60)
    print("TESTING BOARD CREATION PIPELINE")
    print("=" * 60)

    images = chessred_data['images']
    piece_annotations = chessred_data['annotations']['pieces']
    categories = chessred_data['categories']

    print(f"Data: {len(images)} images, {len(piece_annotations)} pieces, {len(categories)} categories")
    print()

    # Show categories using the same structure as preprocessing
    print("CATEGORIES (from preprocessing):")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {i}: {class_name}")
    print()

    # Group pieces by image using same logic as preprocessing
    pieces_by_image = {}
    for piece in piece_annotations:
        img_id = piece['image_id']
        if img_id not in pieces_by_image:
            pieces_by_image[img_id] = []
        pieces_by_image[img_id].append(piece)

    # Find test samples - prefer full boards
    test_samples = []

    # Get some full boards (32 pieces)
    for img_id, pieces in pieces_by_image.items():
        if len(pieces) == 32:
            image_info = next((img for img in images if img['id'] == img_id), None)
            if image_info:
                test_samples.append((image_info, pieces))
        if len(test_samples) >= 2:
            break

    # Get some partial boards
    for img_id, pieces in pieces_by_image.items():
        if 10 <= len(pieces) <= 25:  # Partial boards
            image_info = next((img for img in images if img['id'] == img_id), None)
            if image_info:
                test_samples.append((image_info, pieces))
        if len(test_samples) >= 4:
            break

    print(f"Selected {len(test_samples)} test samples")

    all_tests_passed = True

    for i, (image_info, pieces) in enumerate(test_samples):
        print(f"\n{'=' * 40}")
        print(f"TEST SAMPLE {i + 1}: {image_info['file_name']}")
        print(f"{'=' * 40}")
        print(f"Image ID: {image_info['id']}")
        print(f"Pieces: {len(pieces)}")

        # Show piece positions for debugging
        piece_positions = [(p['chessboard_position'], p['category_id']) for p in pieces]
        piece_positions.sort()

        print(f"\nPiece positions (first 10):")
        for pos, cat_id in piece_positions[:10]:
            cat_name = categories[cat_id]['name'] if cat_id < len(categories) else f'Unknown-{cat_id}'
            print(f"  {pos}: {cat_name}")
        if len(piece_positions) > 10:
            print(f"  ... and {len(piece_positions) - 10} more")

        # Test the preprocessing function
        print(f"\nTesting _create_label_from_pieces()...")
        try:
            board = _create_label_from_pieces(pieces)

            # Validation
            non_empty_count = sum(1 for x in board if x != EMPTY_CLASS_ID)
            empty_count = sum(1 for x in board if x == EMPTY_CLASS_ID)

            print(f"Results:")
            print(f"  Board length: {len(board)} (expected 64)")
            print(f"  Non-empty squares: {non_empty_count}")
            print(f"  Empty squares: {empty_count}")
            print(f"  Input pieces: {len(pieces)}")
            print(f"  Pieces match: {'YES' if non_empty_count == len(pieces) else 'NO'}")

            # Test that all values are valid
            invalid_values = [x for x in board if x < 0 or x > 12]
            print(f"  Invalid values: {len(invalid_values)}")

            # Visualize if it's a reasonable size board
            if len(pieces) >= 10:
                print(f"\nBoard visualization:")
                visualize_board(board, categories)

            # Check if test passed
            test_passed = (
                    len(board) == 64 and
                    non_empty_count == len(pieces) and
                    len(invalid_values) == 0 and
                    empty_count + non_empty_count == 64
            )

            print(f"\nSample {i + 1} test: {'PASSED' if test_passed else 'FAILED'}")
            if not test_passed:
                all_tests_passed = False

        except Exception as e:
            print(f"ERROR processing sample {i + 1}: {e}")
            all_tests_passed = False

        print()

    print(f"Board creation pipeline test: {'PASSED' if all_tests_passed else 'FAILED'}")
    return all_tests_passed


def test_download_function() -> Dict[str, Any]:
    """Test the download function from preprocess.py."""
    print("=" * 60)
    print("TESTING DOWNLOAD FUNCTION")
    print("=" * 60)

    # Create cache directory
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    annotations_path = LOCAL_CACHE_DIR / "test_annotations.json"

    print(f"Testing download to: {annotations_path}")

    try:
        # Test the download function from preprocess.py
        success = _download_file_with_retry(GCS_ANNOTATIONS_URL, annotations_path)

        if success and annotations_path.exists():
            file_size = annotations_path.stat().st_size / (1024 * 1024)  # MB
            print(f"Download successful! File size: {file_size:.1f} MB")

            # Load and return the data
            with open(annotations_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"JSON loaded successfully with keys: {list(data.keys())}")
            return data

        else:
            print("Download failed!")
            return None

    except Exception as e:
        print(f"Download test failed: {e}")
        return None


def run_all_tests() -> bool:
    """Run all preprocessing pipeline tests."""
    print("*" * 80)
    print("CHESSRED PREPROCESSING PIPELINE TEST SUITE")
    print("*" * 80)
    print()

    # Test 1: Download function
    print("Test 1: Download Function")
    chessred_data = test_download_function()
    if chessred_data is None:
        print("[FATAL] Cannot proceed without data")
        return False

    # Test 2: Position conversion
    print(f"\nTest 2: Chess Position Conversion")
    pos_test_passed = test_chess_position_conversion()

    # Test 3: Board creation pipeline
    print(f"\nTest 3: Board Creation Pipeline")
    board_test_passed = test_board_creation_pipeline(chessred_data)

    # Summary
    print("\n" + "*" * 80)
    print("TEST SUMMARY")
    print("*" * 80)

    tests = [
        ("Download Function", chessred_data is not None),
        ("Position Conversion", pos_test_passed),
        ("Board Creation Pipeline", board_test_passed)
    ]

    all_passed = True
    for test_name, passed in tests:
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name:<25}: {status}")
        if not passed:
            all_passed = False

    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    if all_passed:
        print("\n✓ Your preprocessing pipeline is working correctly!")
        print("✓ Ready to run preprocess.py for full data preprocessing")
    else:
        print("\n✗ Some tests failed. Check the preprocessing logic.")

    return all_passed


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Test interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\n[FATAL] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)