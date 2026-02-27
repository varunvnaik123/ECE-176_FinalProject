"""
Backtesting and trading-performance metrics for chart-pattern strategies.

Answers the key question: when our CNN detects a pattern at ≥ 70 % confidence,
does price subsequently move in the predicted direction?

Metrics reported
────────────────
1. Directional accuracy — fraction of detections where price moved correctly.
   Baseline: 50 % (coin flip).  We test against this with a one-sample t-test.

2. Sharpe ratio — annualised risk-adjusted return of a simple long/short strategy
   that acts on every high-confidence detection.

3. Per-class breakdown — accuracy and mean return for each pattern type.

4. Cumulative P&L curve — portfolio growth over time.

All definitions follow standard quantitative-finance conventions.

Usage
─────
    from src.verification.backtest import (
        directional_accuracy, sharpe_ratio,
        per_class_stats, trading_simulation, print_summary,
    )
    results = trading_simulation(detections_df)
    print_summary(results)
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.data_generation.patterns import CLASS_DIRECTION, CLASS_NAMES


# ── Core metrics ──────────────────────────────────────────────────────────────

def directional_accuracy(df: pd.DataFrame, forward_col: str = "forward_return_5d"):
    """
    Compute directional accuracy: did price move in the predicted direction?

    Args:
        df:          DataFrame of detections (output of run_verification)
        forward_col: column name for the forward return

    Returns:
        accuracy (float), t_stat (float), p_value (float), n_trades (int)
    """
    # Exclude 'no_pattern' class and rows with missing returns
    mask = (df["direction"] != "neutral") & df[forward_col].notna()
    sub  = df[mask].copy()

    if len(sub) == 0:
        return None, None, None, 0

    # Binary correct / incorrect per trade
    correct = np.where(
        (sub["direction"] == "bearish") & (sub[forward_col] < 0), 1,
        np.where(
            (sub["direction"] == "bullish") & (sub[forward_col] > 0), 1, 0
        ),
    )

    accuracy = correct.mean()

    # One-sample t-test: is accuracy significantly different from 50 %?
    t_stat, p_value = stats.ttest_1samp(correct, 0.50)

    return accuracy, t_stat, p_value, len(sub)


def sharpe_ratio(
    returns:         np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised Sharpe ratio (risk-free rate assumed 0).

    Args:
        returns:          array of per-trade returns
        periods_per_year: used only when trades are roughly daily frequency

    Returns:
        float Sharpe ratio  (NaN if insufficient data)
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2 or np.std(returns, ddof=1) == 0:
        return float("nan")

    # Annualise assuming each trade ≈ 5 trading days
    # → 252 / 5 ≈ 50 trades per year
    ann_factor = np.sqrt(periods_per_year / 5)
    return float(np.mean(returns) / np.std(returns, ddof=1) * ann_factor)


# ── Per-class analysis ────────────────────────────────────────────────────────

def per_class_stats(df: pd.DataFrame, forward_col: str = "forward_return_5d") -> pd.DataFrame:
    """
    Break down directional accuracy and mean return for each pattern class.

    Returns:
        pd.DataFrame indexed by class_name with columns:
            n, directional_accuracy, mean_return, std_return,
            t_stat, p_value, direction
    """
    rows = []
    for cls_name in CLASS_NAMES:
        if cls_name == "no_pattern":
            continue

        sub = df[(df["class_name"] == cls_name) & df[forward_col].notna()]
        if len(sub) == 0:
            rows.append({"class_name": cls_name, "n": 0})
            continue

        direction = sub["direction"].iloc[0]
        rets      = sub[forward_col].values
        correct   = np.where(
            (direction == "bearish") & (rets < 0), 1,
            np.where((direction == "bullish") & (rets > 0), 1, 0),
        )

        acc = float(correct.mean())
        t_stat, p_val = stats.ttest_1samp(correct, 0.50) if len(correct) > 1 else (np.nan, np.nan)

        rows.append({
            "class_name":           cls_name,
            "direction":            direction,
            "n":                    len(sub),
            "directional_accuracy": round(acc, 4),
            "mean_return":          round(float(rets.mean()), 6),
            "std_return":           round(float(rets.std()), 6),
            "t_stat":               round(float(t_stat), 3),
            "p_value":              round(float(p_val), 4),
        })

    return pd.DataFrame(rows).set_index("class_name")


# ── Trading simulation ────────────────────────────────────────────────────────

def trading_simulation(
    df:           pd.DataFrame,
    forward_col:  str   = "forward_return_5d",
    initial_cap:  float = 10_000.0,
) -> dict:
    """
    Simulate a simple long/short strategy:
        • Bearish detection → short the stock for 5 days
        • Bullish detection → long  the stock for 5 days
        • No-pattern class  → do nothing

    One unit of capital per trade; no compounding between simultaneous trades
    (simplification for clarity).

    Returns:
        dict with keys:
            trade_returns, cumulative_return, sharpe, directional_accuracy,
            t_stat, p_value, n_trades, equity_curve (pd.Series)
    """
    mask = (df["direction"] != "neutral") & df[forward_col].notna()
    sub  = df[mask].sort_values("date").reset_index(drop=True)

    if len(sub) == 0:
        return {"n_trades": 0}

    # Trade return: positive when direction is correct
    trade_rets = np.where(
        sub["direction"] == "bullish",
        sub[forward_col].values,
        -sub[forward_col].values,    # short → negate the price return
    )

    # Build equity curve
    equity   = initial_cap * np.cumprod(1 + trade_rets)
    eq_series = pd.Series(equity, index=sub["date"].values, name="portfolio_value")

    cum_ret = float(equity[-1] / initial_cap - 1)
    sr      = sharpe_ratio(trade_rets)
    acc, t_stat, p_val, n = directional_accuracy(sub, forward_col)

    return {
        "trade_returns":        trade_rets,
        "cumulative_return":    cum_ret,
        "sharpe":               sr,
        "directional_accuracy": acc,
        "t_stat":               t_stat,
        "p_value":              p_val,
        "n_trades":             n,
        "equity_curve":         eq_series,
    }


# ── Pretty-print summary ──────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    """Print a formatted summary of backtesting results."""
    if results.get("n_trades", 0) == 0:
        print("No trades detected.")
        return

    print("=" * 55)
    print("  PATTERN TRADING STRATEGY — BACKTEST SUMMARY")
    print("=" * 55)
    print(f"  Trades executed      : {results['n_trades']:,}")
    print(f"  Directional accuracy : {results['directional_accuracy']:.2%}  "
          f"(baseline 50.00 %)")
    print(f"  t-stat vs 50 %       : {results['t_stat']:+.3f}  "
          f"p = {results['p_value']:.4f}")
    sig = "YES  ★" if results["p_value"] < 0.05 else "no"
    print(f"  Statistically sig.   : {sig}")
    print(f"  Cumulative return    : {results['cumulative_return']:+.2%}")
    print(f"  Annualised Sharpe    : {results['sharpe']:.3f}")
    print("=" * 55)
