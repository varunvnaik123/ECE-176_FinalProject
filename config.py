from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / 'data'
SYNTHETIC_DIR = DATA_DIR / 'synthetic'
REAL_DIR      = DATA_DIR / 'real'
CKPT_DIR      = BASE_DIR / 'checkpoints'
RESULTS_DIR   = BASE_DIR / 'results'

# ── Data generation ────────────────────────────────────────────────────────────
N_PER_CLASS = 1500   # images per class
N_CANDLES   = 60     # candles per chart
IMG_SIZE    = 224
VAL_SPLIT   = 0.15
DATA_SEED   = 42

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LR         = 1e-3
N_EPOCHS   = 40
PATIENCE   = 10      # early stopping
DROPOUT_P  = 0.5

# ── Detection (sliding window on real data) ────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55
WINDOW_SIZE          = 60
SLIDE_STEP           = 5
FORWARD_DAYS         = 5

# ── Classes ────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'head_and_shoulders',
    'double_top',
    'descending_triangle',
    'inv_head_and_shoulders',
    'double_bottom',
    'ascending_triangle',
    'no_pattern',
]

CLASS_DIRECTION = {
    'head_and_shoulders':     'bearish',
    'double_top':             'bearish',
    'descending_triangle':    'bearish',
    'inv_head_and_shoulders': 'bullish',
    'double_bottom':          'bullish',
    'ascending_triangle':     'bullish',
    'no_pattern':             'neutral',
}
