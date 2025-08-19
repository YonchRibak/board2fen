from pathlib import Path

# 🗂️ Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path("C:/chess_pieces_individual_classifier")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "piece_classifier"

# 🧠 Model
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 12  # 6 white pieces + 6 black pieces
CLASS_NAMES = [
    "black_bishop", "black_king", "black_knight", "black_pawn", "black_queen", "black_rook",
    "white_bishop", "white_king", "white_knight", "white_pawn", "white_queen", "white_rook"
]

# 🎲 Reproducibility
SEED = 42
