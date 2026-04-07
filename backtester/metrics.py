"""
backtester/metrics.py

Performance metrics computed from a 1m-resolution equity curve and trade list.

Annualization factor: 365 * 24 * 60 = 525,600 (1m bars per calendar year).
All returns are bar-by-bar differences on the equity curve (unrealized + realized).
"""
import math
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import config

BARS_PER_YEAR = 365 * 24 * 60  # 525,600 — 1m bars per calendar year


def compute_metrics(
    equity_curve: pd.Series,
    trades: list,
    killed_early: bool = False,
) -> dict:
    """
    Compute full performance metrics.

    equity_curve: pd.Series at 1m resolution (includes unrealized + realized PnL,
                  funding payments, and fee drag). DatetimeIndex recommended but not required.
    trades:       list of Trade objects (closed round-trips only).
    killed_early: True if the account DD kill switch triggered during the simulation.
    """
    if len(equity_curve) < 2:
        return _empty_metrics(killed_early)

    equity = equity_curve.dropna().values.astype(np.float64)
    if len(equity) < 2 or equity[0] <= 0:
        return _empty_metrics(killed_early)

    # ── Bar-by-bar returns ────────────────────────────────────────────────────
    prev   = np.maximum(equity[:-1], 1e-8)
    returns = np.diff(equity) / prev
    n_bars  = len(returns)

    # ── CAGR ──────────────────────────────────────────────────────────────────
    years        = n_bars / BARS_PER_YEAR
    total_return = (equity[-1] / equity[0]) - 1.0
    cagr = (equity[-1] / equity[0]) ** (1.0 / max(years, 1e-10)) - 1.0

    # ── Sharpe (annualized, risk-free = 0) ────────────────────────────────────
    mean_ret = returns.mean()
    std_ret  = max(returns.std(ddof=1), 1e-10)
    sharpe   = mean_ret / std_ret * math.sqrt(BARS_PER_YEAR)

    # ── Sortino (downside deviation only) ─────────────────────────────────────
    neg_ret  = returns[returns < 0.0]
    down_std = max(neg_ret.std(ddof=1) if len(neg_ret) > 1 else std_ret, 1e-10)
    sortino  = mean_ret / down_std * math.sqrt(BARS_PER_YEAR)

    # ── Max drawdown ──────────────────────────────────────────────────────────
    peak         = np.maximum.accumulate(equity)
    drawdowns    = (peak - equity) / np.maximum(peak, 1e-10)
    max_drawdown = float(drawdowns.max())

    # ── Calmar & return/DD ratio ──────────────────────────────────────────────
    dd_denom        = max(max_drawdown, 1e-10)
    calmar          = cagr / dd_denom
    return_dd_ratio = total_return / dd_denom

    # ── Trade-based metrics ───────────────────────────────────────────────────
    n_trades = len(trades)
    if n_trades > 0:
        pnls         = np.array([t.pnl for t in trades], dtype=np.float64)
        gross_profit = float(pnls[pnls > 0].sum())
        gross_loss   = float(abs(pnls[pnls < 0].sum()))
        profit_factor   = gross_profit / max(gross_loss, 1e-10)
        win_rate        = float((pnls > 0).sum()) / n_trades
        avg_trade_pct   = float(pnls.mean()) / equity[0] * 100.0
        hold_bars_arr   = np.array([t.hold_bars for t in trades], dtype=np.float64)
        avg_hold_bars   = float(hold_bars_arr.mean())
        max_consec      = _max_consec_losses(trades)
    else:
        profit_factor = 0.0
        win_rate      = 0.0
        avg_trade_pct = 0.0
        avg_hold_bars = 0.0
        max_consec    = 0

    dd_recovery_bars = _avg_dd_recovery(equity)

    return {
        "sharpe":            float(sharpe),
        "sortino":           float(sortino),
        "max_drawdown":      float(max_drawdown),
        "calmar":            float(calmar),
        "return_dd_ratio":   float(return_dd_ratio),
        "profit_factor":     float(profit_factor),
        "cagr":              float(cagr),
        "win_rate":          float(win_rate),
        "avg_trade_pct":     float(avg_trade_pct),
        "n_trades":          int(n_trades),
        "avg_hold_bars":     float(avg_hold_bars),
        "max_consec_losses": int(max_consec),
        "dd_recovery_bars":  float(dd_recovery_bars),
        "killed_early":      bool(killed_early),
    }


def compute_deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_bars: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """
    Deflated Sharpe Ratio (López de Prado, 2018).

    Penalizes Sharpe by the number of strategies tested before finding this one.
    Prevents lucky survivors from being mistaken for genuine edge.

    sharpe_star = sharpe * sqrt(1 - skew*sharpe + (kurt-1)/4 * sharpe^2)
    E_max       = expected maximum SR from n_trials IID trials
    DSR         = Phi(sharpe_star / E_max * sqrt(n_bars))

    A strategy passes if DSR >= config.min_sharpe.
    """
    if n_bars <= 1 or n_trials <= 0:
        return 0.0

    euler_gamma = 0.5772156649  # Euler–Mascheroni constant

    # Correct sharpe for non-normal return distribution
    correction = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2
    if correction <= 0.0:
        return 0.0
    sharpe_star = sharpe * math.sqrt(correction)

    # Expected maximum Sharpe over n_trials IID tests
    p1 = max(min(1.0 - 1.0 / n_trials, 1.0 - 1e-9), 1e-9)
    p2 = max(min(1.0 - 1.0 / (n_trials * math.e), 1.0 - 1e-9), 1e-9)
    z1 = float(norm.ppf(p1))
    z2 = float(norm.ppf(p2))
    e_max = (1.0 - euler_gamma) * z1 + euler_gamma * z2

    if e_max <= 0.0:
        return float(sharpe_star > 0)

    t_stat = (sharpe_star - e_max) * math.sqrt(n_bars)
    return float(norm.cdf(t_stat))


# ── Private helpers ────────────────────────────────────────────────────────────

def _empty_metrics(killed_early: bool = False) -> dict:
    return {
        "sharpe":            0.0,
        "sortino":           0.0,
        "max_drawdown":      0.0,
        "calmar":            0.0,
        "return_dd_ratio":   0.0,
        "profit_factor":     0.0,
        "cagr":              0.0,
        "win_rate":          0.0,
        "avg_trade_pct":     0.0,
        "n_trades":          0,
        "avg_hold_bars":     0.0,
        "max_consec_losses": 0,
        "dd_recovery_bars":  0.0,
        "killed_early":      killed_early,
    }


def _max_consec_losses(trades: list) -> int:
    """Longest consecutive sequence of losing trades."""
    max_streak = 0
    streak     = 0
    for t in trades:
        if t.pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _avg_dd_recovery(equity: np.ndarray) -> float:
    """
    Average number of 1m bars to recover from each drawdown trough.
    A "recovery" is when equity returns to (or exceeds) the prior peak.
    Returns 0.0 if there are no completed recoveries.
    """
    if len(equity) < 3:
        return 0.0

    peak = np.maximum.accumulate(equity)
    in_dd          = False
    trough_start   = 0
    trough_peak    = 0.0
    recoveries:    list = []

    for idx in range(len(equity)):
        if not in_dd:
            if equity[idx] < peak[idx]:
                in_dd        = True
                trough_start = idx
                trough_peak  = float(peak[idx])
        else:
            if equity[idx] >= trough_peak:
                recoveries.append(idx - trough_start)
                in_dd = False
            # Update start if drawdown deepens (for cleaner recovery measurement)
            elif equity[idx] < equity[trough_start]:
                trough_start = idx

    return float(np.mean(recoveries)) if recoveries else 0.0
