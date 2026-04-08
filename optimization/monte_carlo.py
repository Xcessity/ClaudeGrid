"""
optimization/monte_carlo.py

Permutation significance test for grid strategy edge.

Algorithm:
    1. Run a real backtest on the optimization set → real_score
    2. Generate n_shuffles synthetic price series by shuffling LOG RETURNS
       (preserves volatility distribution, destroys autocorrelation / edge)
    3. Run a backtest on each synthetic series → synth_score
    4. p_value = fraction of shuffled runs that beat or match real_score

Pass threshold: p_value <= config.mc_significance (0.05)

Performance:
    - 200 backtests × 1m resolution is expensive.
    - Shuffled runs use a "lite" version: no equity curve storage — just final score.
    - Parallelized via joblib (n_jobs from config.n_workers).

Integration in main.py:
    p = monte_carlo_significance(params, opt_data, ref_atr)
    if p > config.mc_significance:
        continue   # strategy not statistically significant
"""
from __future__ import annotations

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from config import config
from data.cache import DataPair
from backtester.engine import Backtester
from backtester.metrics import compute_metrics
from strategy.grid_engine import GridParams

# ── Shared mmap paths for MC DataFrames ────────────────────────────────────────
_TEMP_DIR        = tempfile.gettempdir()
MMAP_MC_4H_PATH  = os.path.join(_TEMP_DIR, "claudegrid_mc_4h.mmap")
MMAP_MC_1M_PATH  = os.path.join(_TEMP_DIR, "claudegrid_mc_1m.mmap")


# ── Composite score ────────────────────────────────────────────────────────────

def _composite_score(m: dict) -> float:
    """
    Single scalar summary used to rank real vs shuffled backtests.
    Sharpe is the primary signal; penalize heavy drawdown and low trade count.
    Returns -inf for degenerate results (no trades, killed early).
    """
    if m["n_trades"] < config.min_trades_per_backtest or m.get("killed_early", False):
        return -np.inf
    sharpe   = m["sharpe"]
    sortino  = m["sortino"]
    max_dd   = m["max_drawdown"]
    # Geometric combination: rewards both risk-adjusted returns; penalizes high drawdown.
    return float(sharpe * 0.5 + sortino * 0.3) * (1.0 - max_dd)


# ── OHLCV reconstruction from shuffled close prices ───────────────────────────

def _rebuild_ohlcv_from_close(
    df_1m: pd.DataFrame,
    shuffled_close: np.ndarray,
) -> pd.DataFrame:
    """
    Rebuild a synthetic 1m OHLCV DataFrame by scaling the original OHLC by
    the ratio new_close / original_close.  Volume is kept as-is.

    This preserves:
        - OHLC self-consistency (high ≥ open, close; low ≤ open, close)
        - Relative intrabar spread (high-low / close) at every bar
    while randomising the price path (autocorrelation destroyed).
    """
    orig_close = df_1m["close"].values.copy()
    # Avoid division by zero on flat / ghost bars
    scale = np.where(np.abs(orig_close) > 1e-10, shuffled_close / orig_close, 1.0)

    synth = df_1m.copy()
    synth["open"]  = df_1m["open"].values  * scale
    synth["high"]  = df_1m["high"].values  * scale
    synth["low"]   = df_1m["low"].values   * scale
    synth["close"] = shuffled_close
    return synth


# ── Single shuffle worker ──────────────────────────────────────────────────────

def _run_one_shuffle(
    params:        GridParams,
    log_returns:   np.ndarray,
    initial_close: float,
    funding:       pd.Series,
    ref_atr:       float,
    seed:          int,
) -> float:
    """
    Build one synthetic price series, run a backtest, return composite_score.
    Each call is independent and safe to run in a loky worker.

    DataFrames are NOT received as arguments — they are loaded from mmap files
    written by monte_carlo_significance before the parallel loop starts.
    This eliminates the per-worker pickle overhead for large DataFrames.
    """
    # Load DataFrames via mmap — read-only, zero-copy from disk.
    df_4h = joblib.load(MMAP_MC_4H_PATH, mmap_mode="r")
    df_1m = joblib.load(MMAP_MC_1M_PATH, mmap_mode="r")

    rng = np.random.default_rng(seed)
    shuffled_returns = rng.permutation(log_returns)
    shuffled_prices  = initial_close * np.exp(
        np.concatenate([[0.0], shuffled_returns.cumsum()])
    )
    # shuffled_prices has len = len(log_returns) + 1 = len(df_1m)
    synth_1m   = _rebuild_ohlcv_from_close(df_1m, shuffled_prices)
    synth_data = DataPair(df_4h=df_4h, df_1m=synth_1m, funding=funding)

    try:
        bt     = Backtester()
        result = bt.run(params, synth_data, ref_atr=ref_atr)
        m      = compute_metrics(result.equity_curve, result.trades, result.killed_early)
        return _composite_score(m)
    except Exception:
        return -np.inf


# ── Public entry point ─────────────────────────────────────────────────────────

def monte_carlo_significance(
    params:     GridParams,
    data:       DataPair,
    ref_atr:    float,
    n_shuffles: int = None,
) -> float:
    """
    Returns p-value: fraction of shuffled runs that beat or match the real result.

    Pass threshold: p_value <= config.mc_significance (default 0.05)

    Parameters
    ----------
    params     : grid strategy parameters
    data       : DataPair with the optimization set (df_4h + df_1m + funding)
    ref_atr    : median ATR from the full optimization period
    n_shuffles : number of synthetic runs (default: config.mc_n_shuffles = 200)
    """
    if n_shuffles is None:
        n_shuffles = config.mc_n_shuffles

    # ── Real backtest ─────────────────────────────────────────────────────────
    bt          = Backtester()
    real_result = bt.run(params, data, ref_atr=ref_atr)
    real_m      = compute_metrics(real_result.equity_curve, real_result.trades,
                                   real_result.killed_early)
    real_score  = _composite_score(real_m)

    logger.info(
        f"Monte Carlo: real sharpe={real_m['sharpe']:.2f} "
        f"score={real_score:.4f} — running {n_shuffles} shuffles …"
    )

    # ── Prepare shared data ───────────────────────────────────────────────────
    df_1m         = data.df_1m
    close_vals    = df_1m["close"].values
    log_returns   = np.diff(np.log(np.where(close_vals > 1e-10, close_vals, 1e-10)))
    initial_close = float(close_vals[0])

    funding   = data.funding if data.funding is not None else pd.Series(dtype=float)
    n_workers = config.n_workers

    # Dump DataFrames to mmap files once; workers load with mmap_mode='r'.
    # Only log_returns (tiny 1-D array) + small scalars are pickled per task.
    joblib.dump(data.df_4h, MMAP_MC_4H_PATH)
    joblib.dump(df_1m,      MMAP_MC_1M_PATH)
    logger.debug(f"MC: dumped mmap frames → {MMAP_MC_4H_PATH}, {MMAP_MC_1M_PATH}")

    # ── Parallel shuffles — shuffling happens entirely inside each worker ─────
    scores: list[float] = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_run_one_shuffle)(
            params, log_returns, initial_close, funding, ref_atr,
            seed=i,
        )
        for i in range(n_shuffles)
    )

    # ── p-value ───────────────────────────────────────────────────────────────
    beat_count = sum(1 for s in scores if s >= real_score)
    p_value    = beat_count / n_shuffles

    logger.info(
        f"Monte Carlo: {beat_count}/{n_shuffles} shuffles beat real "
        f"(p={p_value:.3f}, threshold={config.mc_significance})"
    )
    return p_value
