"""
Central configuration for the ECE 176 Final Project.

Edit these constants to adjust the experiment scale.
All notebooks import from here so you only need to change settings once.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(PROJECT_ROOT, "data", "synthetic")
REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real")
CKPT_DIR      = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR      = os.path.join(RESULTS_DIR, "logs")

# ── Stage 1 — synthetic dataset generation ───────────────────────────────────

N_PER_CLASS   = 1500    # samples per class (train + val combined)
N_CANDLES     = 60      # candlestick bars per chart window
IMG_SIZE      = 224     # square image size in pixels
VAL_SPLIT     = 0.15    # fraction held out for validation
DATA_SEED     = 42      # reproducibility seed

# ── Stage 2 — model training ──────────────────────────────────────────────────

BATCH_SIZE    = 32      # mini-batch size (Lecture 6)
LEARNING_RATE = 1e-3    # Adam initial LR (Lecture 6)
WEIGHT_DECAY  = 1e-4    # L2 regularisation (Lecture 7)
N_EPOCHS      = 40      # max training epochs
PATIENCE      = 10      # early-stopping patience
STEP_SIZE     = 10      # StepLR epoch interval
GAMMA         = 0.50    # StepLR decay factor
DROPOUT_P     = 0.50    # dropout probability (Lecture 7)
NUM_WORKERS   = 2       # DataLoader worker processes

# ── Stage 3 — real-data verification ─────────────────────────────────────────

VERIFY_START          = "2015-01-01"
VERIFY_END            = "2022-12-31"   # leaves 2023-24 for out-of-sample backtest
BACKTEST_START        = "2023-01-01"
BACKTEST_END          = "2024-12-31"
WINDOW_SIZE           = 60            # bars per detection window (matches training)
STEP_SIZE_SLIDE       = 5             # bars to advance between windows
CONFIDENCE_THRESHOLD  = 0.70          # minimum confidence to record detection
FORWARD_DAYS          = 5             # forward-return horizon in trading days

# ── Grad-CAM ──────────────────────────────────────────────────────────────────

GRADCAM_N_SAMPLES = 5   # number of sample images per class in visualisation grid

# ── Class metadata ────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "head_and_shoulders",     # 0 — bearish
    "double_top",             # 1 — bearish
    "descending_triangle",    # 2 — bearish
    "inv_head_and_shoulders", # 3 — bullish
    "double_bottom",          # 4 — bullish
    "ascending_triangle",     # 5 — bullish
    "no_pattern",             # 6 — neutral
]

SHORT_NAMES = ["H&S", "DblTop", "DescTri", "InvH&S", "DblBot", "AscTri", "NoPat"]

CLASS_DIRECTION = {
    "head_and_shoulders":     "bearish",
    "double_top":             "bearish",
    "descending_triangle":    "bearish",
    "inv_head_and_shoulders": "bullish",
    "double_bottom":          "bullish",
    "ascending_triangle":     "bullish",
    "no_pattern":             "neutral",
}
