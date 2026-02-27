"""
Synthetic chart pattern generation for ECE 176 Final Project.

Generates OHLC price time series for 7 pattern classes:
  Bearish  (0) Head and Shoulders
  Bearish  (1) Double Top
  Bearish  (2) Descending Triangle
  Bullish  (3) Inverse Head and Shoulders
  Bullish  (4) Double Bottom
  Bullish  (5) Ascending Triangle
  Neutral  (6) No Pattern  (geometric Brownian motion)

Each generator returns a close-price array; generate_ohlc() wraps it into
open / high / low / close columns that a candlestick renderer can consume.
"""

import numpy as np
from scipy.interpolate import CubicSpline

# ── Label metadata ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "head_and_shoulders",     # 0
    "double_top",             # 1
    "descending_triangle",    # 2
    "inv_head_and_shoulders", # 3
    "double_bottom",          # 4
    "ascending_triangle",     # 5
    "no_pattern",             # 6
]

CLASS_DIRECTION = {
    0: "bearish", 1: "bearish", 2: "bearish",
    3: "bullish", 4: "bullish", 5: "bullish",
    6: "neutral",
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _smooth(x_key, y_key, n_points, noise_level, base=100.0):
    """Cubic-spline interpolation through key points, then add multiplicative noise."""
    cs  = CubicSpline(x_key, y_key)
    x   = np.linspace(x_key[0], x_key[-1], n_points)
    out = cs(x)
    out *= (1.0 + np.random.normal(0.0, noise_level, n_points))
    out  = np.maximum(out, base * 0.30)   # keep prices positive
    return out


# ── Individual pattern generators ─────────────────────────────────────────────

def generate_head_and_shoulders(n_points=60, noise_level=0.012):
    """Three-peak bearish reversal: left shoulder < head > right shoulder."""
    rng  = np.random
    base = 100.0

    sh  = rng.uniform(0.06, 0.10) * base   # shoulder height
    hh  = rng.uniform(0.13, 0.18) * base   # head height
    nl  = base - rng.uniform(0.02, 0.04) * base   # neckline level
    rsh = sh * rng.uniform(0.70, 1.00)             # right shoulder (slightly asymmetric)
    post = nl - rng.uniform(0.04, 0.08) * base     # post-break drop

    x_key = np.array([0.00, 0.12, 0.22, 0.33, 0.43, 0.50, 0.58, 0.68, 0.78, 0.88, 1.00])
    y_key = np.array([
        base,                        # 0.00 start
        base + sh * 0.40,            # 0.12 rising
        base + sh,                   # 0.22 LEFT SHOULDER peak
        nl,                          # 0.33 drop to neckline
        nl + (base + sh - nl)*0.08,  # 0.43 slight recovery toward head
        base + hh,                   # 0.50 HEAD peak (highest)
        nl,                          # 0.58 drop to neckline
        nl + (base + rsh - nl)*0.05, # 0.68 slight recovery toward right shoulder
        base + rsh,                  # 0.78 RIGHT SHOULDER peak
        nl - (base - nl)*0.30,       # 0.88 breaking neckline
        post,                        # 1.00 continued drop
    ])
    return _smooth(x_key, y_key, n_points, noise_level, base)


def generate_double_top(n_points=60, noise_level=0.012):
    """Two roughly equal peaks (M-shape) → bearish reversal."""
    rng  = np.random
    base = 100.0

    peak  = base + rng.uniform(0.08, 0.15) * base
    valley= base - rng.uniform(0.02, 0.05) * base
    peak2 = peak  * rng.uniform(0.93, 1.04)        # roughly equal heights
    post  = valley - rng.uniform(0.04, 0.08) * base

    x_key = np.array([0.00, 0.15, 0.28, 0.43, 0.58, 0.72, 0.85, 1.00])
    y_key = np.array([
        base,
        base + (peak - base) * 0.50,
        peak,                   # first peak
        valley,                 # valley
        peak2,                  # second peak
        valley,                 # breaking support
        valley - (valley - post)*0.50,
        post,
    ])
    return _smooth(x_key, y_key, n_points, noise_level, base)


def generate_descending_triangle(n_points=60, noise_level=0.012):
    """Flat support + declining resistance → bearish breakout below support."""
    rng  = np.random
    base = 100.0

    support   = base - rng.uniform(0.03, 0.07) * base
    top_start = base + rng.uniform(0.05, 0.10) * base
    top_end   = support + (top_start - support) * rng.uniform(0.10, 0.30)
    n_osc     = rng.randint(3, 6)

    t          = np.linspace(0, 1, n_points)
    resistance = top_start - (top_start - top_end) * t  # declining

    oscillation = 0.5 + 0.5 * np.sin(2 * np.pi * n_osc * t)
    amplitude   = (resistance - support) * (1.0 - 0.60 * t)  # amplitude shrinks
    prices      = support + amplitude * oscillation

    # Break below support in final 15 %
    break_start = int(0.82 * n_points)
    drop_end    = support - rng.uniform(0.04, 0.08) * base
    prices[break_start:] = np.linspace(support, drop_end, n_points - break_start)

    prices += np.random.normal(0, noise_level * base, n_points)
    return np.maximum(prices, base * 0.30)


def generate_inv_head_and_shoulders(n_points=60, noise_level=0.012):
    """Three-trough bullish reversal: left shoulder > head < right shoulder."""
    rng  = np.random
    base = 100.0

    st   = rng.uniform(0.06, 0.10) * base    # shoulder trough depth
    ht   = rng.uniform(0.13, 0.18) * base    # head trough depth
    nl   = base + rng.uniform(0.02, 0.04) * base   # neckline level
    rst  = st * rng.uniform(0.70, 1.00)             # right shoulder (asymmetric)
    post = nl + rng.uniform(0.04, 0.08) * base      # post-break rally

    x_key = np.array([0.00, 0.12, 0.22, 0.33, 0.43, 0.50, 0.58, 0.68, 0.78, 0.88, 1.00])
    y_key = np.array([
        base,
        base - st * 0.40,
        base - st,                   # LEFT SHOULDER trough
        nl,                          # rise to neckline
        nl - (nl - (base - ht))*0.08,
        base - ht,                   # HEAD trough (deepest)
        nl,                          # rise to neckline
        nl - (nl - (base - rst))*0.05,
        base - rst,                  # RIGHT SHOULDER trough
        nl + (post - nl) * 0.30,
        post,                        # continued rally
    ])
    return _smooth(x_key, y_key, n_points, noise_level, base)


def generate_double_bottom(n_points=60, noise_level=0.012):
    """Two roughly equal troughs (W-shape) → bullish reversal."""
    rng  = np.random
    base = 100.0

    trough = base - rng.uniform(0.08, 0.15) * base
    peak   = base + rng.uniform(0.02, 0.05) * base
    trough2= trough * rng.uniform(0.97, 1.07)
    post   = peak + rng.uniform(0.04, 0.08) * base

    x_key = np.array([0.00, 0.15, 0.28, 0.43, 0.58, 0.72, 0.85, 1.00])
    y_key = np.array([
        base,
        base - (base - trough) * 0.50,
        trough,           # first trough
        peak,             # peak between troughs
        trough2,          # second trough
        peak,             # breaking above resistance
        peak + (post - peak) * 0.50,
        post,
    ])
    return _smooth(x_key, y_key, n_points, noise_level, base)


def generate_ascending_triangle(n_points=60, noise_level=0.012):
    """Flat resistance + rising support → bullish breakout above resistance."""
    rng  = np.random
    base = 100.0

    resistance  = base + rng.uniform(0.05, 0.10) * base
    bot_start   = base - rng.uniform(0.03, 0.07) * base
    bot_end     = resistance - (resistance - bot_start) * rng.uniform(0.10, 0.30)
    n_osc       = rng.randint(3, 6)

    t          = np.linspace(0, 1, n_points)
    support    = bot_start + (bot_end - bot_start) * t  # rising

    oscillation = 0.5 + 0.5 * np.sin(2 * np.pi * n_osc * t + np.pi)
    amplitude   = (resistance - support) * (1.0 - 0.60 * t)
    prices      = support + amplitude * oscillation

    # Break above resistance in final 15 %
    break_start = int(0.82 * n_points)
    rally_end   = resistance + rng.uniform(0.04, 0.08) * base
    prices[break_start:] = np.linspace(resistance, rally_end, n_points - break_start)

    prices += np.random.normal(0, noise_level * base, n_points)
    return np.maximum(prices, base * 0.30)


def generate_no_pattern(n_points=60, noise_level=0.012):
    """Geometric Brownian Motion — no recognisable pattern."""
    dt    = 1.0 / n_points
    mu    = np.random.uniform(-0.10, 0.10)
    sigma = np.random.uniform(0.15, 0.40)
    S0    = 100.0

    log_ret = np.random.normal(
        (mu - 0.5 * sigma**2) * dt,
        sigma * np.sqrt(dt),
        n_points
    )
    prices = S0 * np.exp(np.cumsum(log_ret))
    return np.maximum(prices, S0 * 0.30)


# ── OHLC wrapper ───────────────────────────────────────────────────────────────

PATTERN_GENERATORS = {
    0: generate_head_and_shoulders,
    1: generate_double_top,
    2: generate_descending_triangle,
    3: generate_inv_head_and_shoulders,
    4: generate_double_bottom,
    5: generate_ascending_triangle,
    6: generate_no_pattern,
}


def generate_ohlc(close_prices, wick_scale=0.40):
    """
    Convert a close-price array into realistic OHLC candlestick data.

    Args:
        close_prices: 1-D np.array of closing prices
        wick_scale:   controls wick length relative to body size

    Returns:
        np.array of shape (N, 4) — columns [open, high, low, close]
    """
    n      = len(close_prices)
    opens  = np.empty(n)
    opens[0] = close_prices[0] * np.random.uniform(0.996, 1.004)
    opens[1:] = close_prices[:-1] * (1 + np.random.normal(0, 0.003, n - 1))

    body   = np.abs(close_prices - opens)
    avg_b  = np.mean(body) + 1e-8

    upper_wick = body * np.random.uniform(wick_scale*0.5, wick_scale*1.5, n) \
               + np.random.exponential(avg_b * 0.25, n)
    lower_wick = body * np.random.uniform(wick_scale*0.5, wick_scale*1.5, n) \
               + np.random.exponential(avg_b * 0.25, n)

    highs  = np.maximum(opens, close_prices) + upper_wick
    lows   = np.minimum(opens, close_prices) - lower_wick

    return np.stack([opens, highs, lows, close_prices], axis=1)  # (N, 4)


def generate_pattern_sample(class_id, n_points=60, noise_level=None):
    """
    Generate a single OHLC sample for the given class.

    Args:
        class_id:    int in [0, 6]
        n_points:    number of candlesticks in the window
        noise_level: if None, sampled uniformly in [0.008, 0.022]

    Returns:
        ohlc: np.array (n_points, 4)
    """
    if noise_level is None:
        noise_level = np.random.uniform(0.008, 0.022)
    close = PATTERN_GENERATORS[class_id](n_points=n_points, noise_level=noise_level)
    return generate_ohlc(close)
