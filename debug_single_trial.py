"""
debug_single_trial.py — Directly simulate what study.optimize() does for one trial,
with per-step timing to isolate where time is spent.

Run:  python debug_single_trial.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import config
from data.cache import CACHE_DIR, DataPair
from optimization.optimizer import (
    NumpyOHLCV, prepare_4h_arrays, prepare_numpy_arrays,
    _numpy_to_dataframe, _run_trial,
)
import optimization.optimizer as _opt_mod
from optimization.parameter_space import sample_optuna_params
from backtester.engine import Backtester
from strategy.grid_engine import GridEngine, GridParams
from backtester.metrics import compute_metrics


def hms(s):
    return f"{s:.3f}s" if s < 60 else f"{s/60:.1f}min"


def main():
    files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith("_4h.parquet"))
    df_4h = pd.read_parquet(CACHE_DIR / files[0])
    df_1m = pd.read_parquet(CACHE_DIR / files[0].replace("_4h", "_1m"))
    split_idx = int(len(df_4h) * (1 - config.holdout_fraction))
    df_4h_opt = df_4h.iloc[:split_idx]
    df_1m_opt = df_1m[df_1m.index < df_4h.index[split_idx]]
    print(f"4h rows: {len(df_4h_opt):,}  1m rows: {len(df_1m_opt):,}")

    # ── Set up module globals as optimizer.__init__ would ──────────────────────
    if _opt_mod._NP_4H is None:
        print("Building _NP_4H / _NP_1M ...")
        t0 = time.perf_counter()
        _opt_mod._NP_4H = prepare_4h_arrays(df_4h_opt)
        _opt_mod._NP_1M = prepare_numpy_arrays(df_1m_opt)
        print(f"  prepare: {hms(time.perf_counter()-t0)}")
    else:
        print("Using existing _NP_4H / _NP_1M")

    np_4h = _opt_mod._NP_4H
    np_1m = _opt_mod._NP_1M

    # ── Simulate Optuna sampling ───────────────────────────────────────────────
    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(population_size=50, seed=42),
    )
    trial = study.ask()
    params = sample_optuna_params(trial)
    print(f"\nSampled params:")
    print(f"  atr_period={params.atr_period}  adx_period={params.adx_period}  hurst_window={params.hurst_window}")
    print(f"  n_levels={params.n_levels}  vpvr_window={params.vpvr_window}  use_vpvr_anchor={params.use_vpvr_anchor}")
    print(f"  position_size_pct={params.position_size_pct:.2f}  max_open_levels={params.max_open_levels}")

    # ── Step 1: DataFrame reconstruction ──────────────────────────────────────
    print("\n[Step 1] _numpy_to_dataframe ...")
    t0 = time.perf_counter()
    df_4h_w = _numpy_to_dataframe(np_4h)
    print(f"  4h: {hms(time.perf_counter()-t0)}  rows={len(df_4h_w):,}  tz={df_4h_w.index.tz}")
    t0 = time.perf_counter()
    df_1m_w = _numpy_to_dataframe(np_1m)
    print(f"  1m: {hms(time.perf_counter()-t0)}  rows={len(df_1m_w):,}  tz={df_1m_w.index.tz}")

    # ── Step 2: Test _period_1m alignment ─────────────────────────────────────
    print("\n[Step 2] _period_1m for first 5 bars ...")
    engine = GridEngine()
    for bar in df_4h_w.head(5).itertuples():
        period = engine._period_1m(df_1m_w, bar.Index)
        print(f"  bar_4h={bar.Index}  1m_rows={len(period)}")

    # ── Step 3: Time the warm-up period (first hurst_window bars) ─────────────
    print(f"\n[Step 3] Warm-up period: {params.hurst_window} bars × inner loop ...")
    t0 = time.perf_counter()
    count = 0
    for i, bar_4h in enumerate(df_4h_w.itertuples()):
        if i >= params.hurst_window:
            break
        for bar_1m in engine._period_1m(df_1m_w, bar_4h.Index).itertuples():
            count += 1
    print(f"  {hms(time.perf_counter()-t0)} for {params.hurst_window} outer × {count} inner bars")

    # ── Step 4: Time compute_vpvr specifically ─────────────────────────────────
    from strategy.indicators import compute_vpvr
    print(f"\n[Step 4] compute_vpvr (vpvr_window={params.vpvr_window}) x 10 times ...")
    t0 = time.perf_counter()
    for i in range(10, 20):
        vpvr_slice = df_4h_w.iloc[max(0, i - params.vpvr_window): i]
        compute_vpvr(vpvr_slice, lookback=params.vpvr_window)
    elapsed = time.perf_counter() - t0
    print(f"  10 calls: {hms(elapsed)}  ({elapsed/10:.4f}s each)")
    print(f"  Estimated if called every bar: {hms(len(df_4h_w) * elapsed/10)}")

    # ── Step 5: Full trial with step-by-step timing ────────────────────────────
    print(f"\n[Step 5] Full backtest via Backtester.run() ...")
    data = DataPair(df_4h=df_4h_w, df_1m=df_1m_w, funding=pd.Series(dtype=float))
    bt = Backtester()

    import tracemalloc
    tracemalloc.start()
    t0 = time.perf_counter()
    result = bt.run(params, data, initial_capital=10_000.0, ref_atr=100.0, np_4h=np_4h)
    elapsed = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  elapsed: {hms(elapsed)}")
    print(f"  trades={len(result.trades)}  equity_pts={len(result.equity_curve):,}")
    print(f"  mem peak={peak/1e6:.1f}MB  cur={cur/1e6:.1f}MB")

    # ── Step 6: compute_metrics timing ────────────────────────────────────────
    print(f"\n[Step 6] compute_metrics ...")
    t0 = time.perf_counter()
    metrics = compute_metrics(result.equity_curve, result.trades, result.killed_early)
    print(f"  {hms(time.perf_counter()-t0)}  sharpe={metrics['sharpe']:.3f}  n_trades={metrics['n_trades']}")

    # ── Step 7: Full _run_trial with these exact params ────────────────────────
    print(f"\n[Step 7] Full _run_trial (includes DataFrame reconstruction) ...")
    t0 = time.perf_counter()
    m2 = _run_trial(params, np_4h, np_1m, pd.Series(dtype=float), 100.0)
    print(f"  {hms(time.perf_counter()-t0)}  sharpe={m2['sharpe']:.3f}")


if __name__ == "__main__":
    main()
