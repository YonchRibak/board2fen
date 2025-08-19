#!/usr/bin/env python3
"""
json_sampler.py

Create a compact sample of a large JSON file by taking up to N evenly-spaced
items from every list found anywhere in the structure (recursively).
All dictionary keys are preserved.

Usage:
    python json_sampler.py --input "c://datasets/ChessReD/annotations.json"
    # optional:
    # python json_sampler.py --input ... --output annotations_sample.json --n 3
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

def evenly_spaced_indices(length: int, n: int) -> list[int]:
    """
    Return up to n indices evenly spaced over range(length), always including
    the first and last when length >= 2. Deterministic (no randomness).
    """
    if length <= n:
        return list(range(length))
    # Create n positions between [0, length-1], inclusive
    return sorted({int(round(i * (length - 1) / (n - 1))) for i in range(n)})

def sample_structure(obj: Any, n: int = 3) -> Any:
    """
    Recursively sample lists to at most n elements; keep all dict keys.
    Non-list/dict values pass through unchanged.
    """
    if isinstance(obj, list):
        idxs = evenly_spaced_indices(len(obj), n)
        return [sample_structure(obj[i], n) for i in idxs]
    elif isinstance(obj, dict):
        # Preserve all keys; recurse into values
        return {k: sample_structure(v, n) for k, v in obj.items()}
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description="Sample every list in a JSON file to N items.")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON file.")
    parser.add_argument("--output", "-o", help="Path to output JSON file (default: <input>_sample.json).")
    parser.add_argument("--n", type=int, default=3, help="Max items per list (default: 3).")
    parser.add_argument("--indent", type=int, default=2, help="Pretty-print indent (default: 2).")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_sample.json")

    # Load, sample, save
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sampled = sample_structure(data, n=args.n)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=args.indent)

    print(f"✅ Wrote sampled JSON to: {out_path} (max {args.n} items per list)")

if __name__ == "__main__":
    main()
