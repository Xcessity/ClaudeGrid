"""
debug_trial.py — Timing diagnostic for ClaudeGrid optimization pipeline.

Run from the project root:
    python debug_trial.py

Reports timing for each phase so you know exactly where time is spent
before running the full infinite search loop.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import config
from data.cache import CACHE_DIR
from strategy.indicators import (
    compute_atr, compute_adx, compute_hurst_dfa,
    compute_bb_width, compute_bb_width_percentile,
)
from optimization.parameter_space import PARAM_SPACE
from optimization.optimizer import NumpyOHLCV, prepare_numpy_arrays
from backtester.engine import Backtester
from data.cache import DataPair
from strategy.grid_engine import GridParams


def _hms(s: float) -> str:
    if s < 60:
        return f"{s:.2f}s"
    m, s2 = divmod(s, 60)
    return f"{int(m)}m {s2:.1f}s"


def main() -> None:
    # ── Pick first cached symbol ───────────────────────────────────────────────
    parquet_files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith("_4h.parquet"))
    if not parquet_files:
        print("No cached parquet files found — run main.py once to populate cache.")
        sys.exit(1)

    sym_file = parquet_files[0]
    print(f"Using cached symbol: {sym_file.replace('_4h.parquet', '')}")

    df_4h = pd.read_parquet(CACHE_DIR / sym_file)
    df_1m = pd.read_parquet(CACHE_DIR / sym_file.replace("_4h", "_1m"))

    split_idx = int(len(df_4h) * (1 - config.holdout_fraction))
    df_4h_opt = df_4h.iloc[:split_idx]
    df_1m_opt = df_1m[df_1m.index < df_4h.index[split_idx]]
    print(f"Optimization data — 4h bars: {len(df_4h_opt):,}  |  1m bars: {len(df_1m_opt):,}\n")

    atr_lo, atr_hi = int(PARAM_SPACE["atr_period"][1]),   int(PARAM_SPACE["atr_period"][2])
    adx_lo, adx_hi = int(PARAM_SPACE["adx_period"][1]),   int(PARAM_SPACE["adx_period"][2])
    hst_lo, hst_hi = int(PARAM_SPACE["hurst_window"][1]), int(PARAM_SPACE["hurst_window"][2])
    n_hurst_windows = hst_hi - hst_lo + 1

    # ── Phase 1: ATR ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for p in range(atr_lo, atr_hi + 1):
        compute_atr(df_4h_opt, p)
    atr_t = time.perf_counter() - t0
    print(f"ATR  ({atr_hi - atr_lo + 1:>3} periods):  {_hms(atr_t)}")

    # ── Phase 2: ADX ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for p in range(adx_lo, adx_hi + 1):
        compute_adx(df_4h_opt, p)
    adx_t = time.perf_counter() - t0
    print(f"ADX  ({adx_hi - adx_lo + 1:>3} periods):  {_hms(adx_t)}")

    # ── Phase 3: Hurst (one window, then extrapolate) ─────────────────────────
    t0 = time.perf_counter()
    compute_hurst_dfa(df_4h_opt["close"], 100)
    one_hurst = time.perf_counter() - t0
    est_all_hurst = one_hurst * n_hurst_windows
    print(f"Hurst (1 window, w=100):  {_hms(one_hurst)}  "
          f"-> est. {n_hurst_windows} windows: {_hms(est_all_hurst)}")

    # ── Phase 4: Full prepare_4h_arrays ───────────────────────────────────────
    print("\nRunning full prepare_4h_arrays (this is the slow part) …")
    t0 = time.perf_counter()
    from optimization.optimizer import prepare_4h_arrays
    np_4h = prepare_4h_arrays(df_4h_opt)
    prep_t = time.perf_counter() - t0
    print(f"prepare_4h_arrays total: {_hms(prep_t)}")

    # ── Phase 5: Single trial timing ──────────────────────────────────────────
    np_1m = prepare_numpy_arrays(df_1m_opt)
    params = GridParams()   # default params
    data = DataPair(df_4h=df_4h_opt, df_1m=df_1m_opt, funding=pd.Series(dtype=float))
    bt = Backtester()

    print("\nRunning 3 timed trials …")
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        result = bt.run(params, data, initial_capital=10_000.0, ref_atr=100.0, np_4h=np_4h)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Trial {i+1}: {_hms(elapsed)}  (trades={len(result.trades)})")

    avg_trial = sum(times) / len(times)
    print(f"\nAverage per-trial: {_hms(avg_trial)}")
    print(f"Projected 500 trials (n_jobs=1): {_hms(avg_trial * 500)}")
    print(f"\nSummary:")
    print(f"  prepare_4h_arrays:  {_hms(prep_t)}  (runs once per symbol per cycle)")
    print(f"  per trial:          {_hms(avg_trial)}  × 500 = {_hms(avg_trial * 500)}")
    print(f"  total per cycle:    {_hms(prep_t + avg_trial * 500)}")


if __name__ == "__main__":
    main()
