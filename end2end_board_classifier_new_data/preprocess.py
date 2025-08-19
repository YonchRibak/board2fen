# board2fen/end2end_board_classifier/preprocess.py
"""
Preprocess ChessReD annotations into streamable JSONL indexes.

Inputs:
  C:/datasets/ChessReD/annotations.json
  C:/datasets/ChessReD/chessred/images/...    (or)
  C:/datasets/ChessReD/images/...             (or)
  C:/datasets/ChessRed/chessred/images/...

Outputs (next to annotations.json):
  C:/datasets/ChessReD/chessred_preprocessed/
    - class_order.json
    - index_train.jsonl
    - index_val.jsonl
    - index_test.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ----- CONFIG: point this at the folder that holds annotations.json -----
ANNOT_DIR = Path(r"C:/datasets/ChessReD")
ANNOT_PATH = ANNOT_DIR / "annotations.json"
OUT_DIR = ANNOT_DIR / "chessred_preprocessed"

# Candidate bases that may contain the 'images' folder referenced in annotations
IMAGE_BASE_CANDIDATES = [
    ANNOT_DIR,                                   # C:/datasets/ChessReD/images/...
    ANNOT_DIR / "chessred",                      # C:/datasets/ChessReD/chessred/images/...
    Path(r"C:/datasets/ChessRed/chessred"),      # C:/datasets/ChessRed/chessred/images/...
]

FILE_TO_COL = {c: i for i, c in enumerate("abcdefgh")}

def algebraic_to_index(square: str) -> int:
    # a1=0..h1=7, a8=56..h8=63 (python-chess style indexing)
    file_c = square[0].lower()
    rank_c = square[1]
    col = FILE_TO_COL[file_c]
    row = int(rank_c) - 1
    return row * 8 + col

def build_class_order(categories: List[Dict[str, Any]]) -> List[str]:
    """Build class order from categories (sorted by id)."""
    categories_sorted = sorted(categories, key=lambda c: c["id"])
    return [c["name"].strip().lower() for c in categories_sorted]

def build_category_maps(categories: List[Dict[str, Any]], class_order: List[str]) -> Tuple[Dict[int, int], int]:
    """Map dataset category_id -> model class index [0..N-1]; also return the dataset 'empty' id."""
    name_to_idx = {n: i for i, n in enumerate(class_order)}
    catid_to_idx: Dict[int, int] = {}
    empty_id = None
    for c in categories:
        nm = c["name"].strip().lower()
        if nm not in name_to_idx:
            raise ValueError(f"Category '{nm}' not present in class_order.")
        catid_to_idx[c["id"]] = name_to_idx[nm]
        if nm == "empty":
            empty_id = c["id"]
    if empty_id is None:
        raise ValueError("No 'empty' category found.")
    return catid_to_idx, empty_id

def collect_pieces_by_image(anno: Dict[str, Any], class_order: List[str]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Returns mapping: image_id -> list of (square_index, class_index) for all non-empty pieces.
    Any squares not listed will be considered 'empty' during training.
    """
    catid_to_idx, _empty_id = build_category_maps(anno["categories"], class_order)
    per_image: Dict[int, List[Tuple[int, int]]] = {}
    for p in anno["annotations"]["pieces"]:
        img_id = int(p["image_id"])
        sq = p["chessboard_position"]  # e.g. "a8"
        ds_cat = int(p["category_id"])
        sq_idx = algebraic_to_index(sq)
        cls_idx = catid_to_idx[ds_cat]
        if class_order[cls_idx] == "empty":
            continue
        per_image.setdefault(img_id, []).append((sq_idx, cls_idx))
    return per_image

def images_index_by_id(anno: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(im["id"]): im for im in anno["images"]}

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def resolve_split_ids(anno: Dict[str, Any], split_name: str) -> List[int]:
    return [int(x) for x in anno["splits"][split_name]["image_ids"]]

def resolve_image_abs_path(rel_path: str) -> Path | None:
    """
    Try multiple base directories until an existing image is found.
    rel_path is like 'images/0/G000_IMG000.jpg' coming from annotations.json.
    """
    rel_path = rel_path.replace("\\", "/")
    for base in IMAGE_BASE_CANDIDATES:
        cand = (base / rel_path).resolve()
        if cand.exists():
            return cand
    return None

def main():
    print(f"📥 Loading annotations: {ANNOT_PATH}")
    if not ANNOT_PATH.exists():
        raise FileNotFoundError(f"{ANNOT_PATH} not found")

    anno = json.loads(ANNOT_PATH.read_text(encoding="utf-8"))

    # Build class order from file
    class_order = build_class_order(anno["categories"])

    # Build maps
    per_image_sparse = collect_pieces_by_image(anno, class_order)
    img_by_id = images_index_by_id(anno)

    # Save class order
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "class_order.json").write_text(json.dumps(class_order, indent=2), encoding="utf-8")
    print(f"📝 Wrote {OUT_DIR/'class_order.json'}")

    # Build rows for a split
    missing = 0
    def build_rows(ids: List[int]) -> List[Dict[str, Any]]:
        nonlocal missing
        rows = []
        for iid in ids:
            meta = img_by_id.get(iid)
            if meta is None:
                continue
            rel_path = meta["path"].replace("\\", "/")  # e.g. 'images/47/G047_IMG013.jpg'
            abs_path = resolve_image_abs_path(rel_path)
            if abs_path is None:
                missing += 1
                continue
            sparse = per_image_sparse.get(iid, [])  # list[(sq, cls)]
            rows.append({
                "image_id": iid,
                "file_path": abs_path.as_posix(),
                "labels_sparse": sparse,
            })
        return rows

    train_ids = resolve_split_ids(anno, "train")
    val_ids   = resolve_split_ids(anno, "val")
    test_ids  = resolve_split_ids(anno, "test")

    print("🔧 Building split indexes...")
    train_rows = build_rows(train_ids)
    val_rows   = build_rows(val_ids)
    test_rows  = build_rows(test_ids)

    write_jsonl(OUT_DIR / "index_train.jsonl", train_rows)
    write_jsonl(OUT_DIR / "index_val.jsonl",   val_rows)
    write_jsonl(OUT_DIR / "index_test.jsonl",  test_rows)

    print(f"✅ Wrote {len(train_rows)} train, {len(val_rows)} val, {len(test_rows)} test entries.")
    if missing:
        print(f"⚠️  Skipped {missing} images because the files were not found. "
              f"Check your images base directories and case (ChessReD vs ChessRed).")
    print(f"📂 Output dir: {OUT_DIR}")

if __name__ == "__main__":
    main()
