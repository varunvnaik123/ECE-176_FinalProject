"""
Real stock data fetching and sliding-window pattern detection.

Stage 2 of the project pipeline:
  1. Download S&P 500 OHLC data with yfinance.
  2. Slide a 60-bar window over each stock's history.
  3. Classify each window with the trained CNN.
  4. For high-confidence detections (≥ 70 %), record the 5-day forward return.

The goal is to answer: when the model detects a pattern, does price move in
the predicted direction?

Usage
─────
    from src.verification.stock_data import run_verification

    detections = run_verification(
        model, device,
        start="2015-01-01", end="2022-12-31",   # leaves 2023-24 for backtesting
    )
"""

import io
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import yfinance as yf

from src.data_generation.patterns import CLASS_NAMES, CLASS_DIRECTION


# ── 50 liquid S&P 500 tickers (diversified across sectors) ───────────────────

SP500_TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "ADBE",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "BMY", "AMGN",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C",  "AXP", "USB", "PNC",
    # Consumer
    "WMT", "PG",  "KO",  "PEP", "MCD", "NKE", "SBUX","TGT", "COST","LOW",
    # Industrials
    "BA",  "CAT", "HON", "GE",  "MMM", "RTX", "LMT", "NOC", "DE",  "UPS",
]


# ── Preprocessing (must match training) ───────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download OHLC data for one ticker.  Returns None on failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
        except Exception as e:
            print(f"  [WARN] {ticker}: download failed — {e}")
            return None

    if df is None or len(df) < 120:
        return None

    df = df[["Open", "High", "Low", "Close"]].copy()
    df.columns = ["open", "high", "low", "close"]
    df = df.dropna()
    return df


def _ohlc_window_to_image(window: pd.DataFrame, img_size: int = 224) -> Image.Image:
    """
    Render a slice of a OHLC DataFrame as a candlestick chart image.
    Matches the synthetic rendering style used during training.
    """
    n      = len(window)
    opens  = window["open"].values
    highs  = window["high"].values
    lows   = window["low"].values
    closes = window["close"].values

    inches = img_size / 100.0
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    width = 0.65
    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        colour = "#26a69a" if c >= o else "#ef5350"

        ax.plot([i, i], [l, h], color=colour, linewidth=0.80, zorder=1)
        body_lo = min(o, c)
        body_h  = max(abs(c - o), (h - l) * 0.01)
        rect    = mpatches.FancyBboxPatch(
            (i - width / 2, body_lo), width, body_h,
            boxstyle="square,pad=0",
            facecolor=colour, edgecolor=colour, linewidth=0.40, zorder=2,
        )
        ax.add_patch(rect)

    prange = max(highs.max() - lows.min(), 1e-6)
    ax.set_xlim(-0.60, n - 0.40)
    ax.set_ylim(lows.min() - prange * 0.04, highs.max() + prange * 0.04)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100,
                bbox_inches="tight", pad_inches=0, facecolor="white")
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((img_size, img_size), Image.LANCZOS)
    plt.close(fig)
    return img


# ── Per-stock detection ────────────────────────────────────────────────────────

def detect_in_stock(
    model,
    device,
    ohlc_df:              pd.DataFrame,
    ticker:               str,
    window_size:          int   = 60,
    step_size:            int   = 5,
    confidence_threshold: float = 0.70,
    forward_days:         int   = 5,
    img_size:             int   = 224,
) -> list[dict]:
    """
    Slide a window over one stock's OHLC history and return pattern detections.

    Args:
        model:                trained PyTorch model
        device:               torch.device
        ohlc_df:              DataFrame with columns [open, high, low, close]
        ticker:               stock symbol (for output records)
        window_size:          number of candlesticks per chart window
        step_size:            bars to advance between windows
        confidence_threshold: minimum softmax probability to record a detection
        forward_days:         how many bars ahead to measure the return
        img_size:             chart image size (must match training)

    Returns:
        list of dicts:
            ticker, date, class_id, class_name, direction,
            confidence, forward_return_{forward_days}d
    """
    model.eval()
    detections = []
    dates      = ohlc_df.index.tolist()
    n          = len(ohlc_df)

    for i in range(0, n - window_size - forward_days, step_size):
        window   = ohlc_df.iloc[i : i + window_size]
        det_date = dates[i + window_size - 1]

        img        = _ohlc_window_to_image(window, img_size=img_size)
        img_tensor = _TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            conf  = conf.item()
            pred  = pred.item()

        if conf < confidence_threshold:
            continue

        # 5-day forward return
        close_now    = ohlc_df.iloc[i + window_size - 1]["close"]
        close_future = ohlc_df.iloc[i + window_size + forward_days - 1]["close"]
        fwd_return   = float((close_future - close_now) / close_now)

        detections.append({
            "ticker":                      ticker,
            "date":                        str(det_date)[:10],
            "class_id":                    pred,
            "class_name":                  CLASS_NAMES[pred],
            "direction":                   CLASS_DIRECTION[pred],
            "confidence":                  round(conf, 4),
            f"forward_return_{forward_days}d": round(fwd_return, 6),
        })

    return detections


# ── Full verification run ─────────────────────────────────────────────────────

def run_verification(
    model,
    device,
    tickers:              list[str]     = None,
    start:                str           = "2015-01-01",
    end:                  str           = "2022-12-31",
    window_size:          int           = 60,
    step_size:            int           = 5,
    confidence_threshold: float         = 0.70,
    forward_days:         int           = 5,
    img_size:             int           = 224,
    verbose:              bool          = True,
) -> pd.DataFrame:
    """
    Run pattern detection across all tickers and collect detections.

    Args:
        model:                trained PyTorch classification model
        device:               torch.device
        tickers:              list of ticker symbols (defaults to SP500_TICKERS)
        start / end:          date range for historical data
        window_size:          candlesticks per window
        step_size:            bars to advance between windows
        confidence_threshold: min softmax confidence to record detection
        forward_days:         forward return horizon in trading days
        img_size:             chart render size (px, square)
        verbose:              print progress

    Returns:
        pd.DataFrame of all detections
    """
    tickers = tickers or SP500_TICKERS
    all_detections = []

    for ticker in tickers:
        if verbose:
            print(f"Processing {ticker} ...", end=" ", flush=True)

        ohlc = _download(ticker, start, end)
        if ohlc is None:
            if verbose:
                print("skipped (no data)")
            continue

        dets = detect_in_stock(
            model, device, ohlc, ticker,
            window_size=window_size,
            step_size=step_size,
            confidence_threshold=confidence_threshold,
            forward_days=forward_days,
            img_size=img_size,
        )
        all_detections.extend(dets)

        if verbose:
            print(f"{len(dets)} detections")

    df = pd.DataFrame(all_detections)
    if verbose:
        print(f"\nTotal detections: {len(df)}")
        if not df.empty:
            print(df["class_name"].value_counts())

    return df
