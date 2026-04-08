"""
debug_worker_profile.py — Profile a single trial as it runs inside a worker process.

This script ITSELF runs as if it were a worker (it calls _worker_init then _worker_eval),
so it must be run as a proper file (not -c) to be importable by ProcessPoolExecutor.

Run:  python debug_worker_profile.py
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import config
from data.cache import CACHE_DIR, DataPair
from optimization.optimizer import (
    _MMAP_4H_PATH, _MMAP_1M_PATH,
    _worker_init, _worker_eval, _NP_4H, _NP_1M,
    prepare_4h_arrays, prepare_numpy_arrays,
    _numpy_to_dataframe, _run_trial,
)
import optimization.optimizer as _opt_mod
from optimization.parameter_space import PARAM_SPACE
from strategy.grid_engine import GridParams, GridEngine
from backtester.engine import Backtester
import joblib, numpy as np


def main() -> None:
    files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith("_4h.parquet"))
    df_4h = pd.read_parquet(CACHE_DIR / files[0])
    df_1m = pd.read_parquet(CACHE_DIR / files[0].replace("_4h", "_1m"))
    split_idx = int(len(df_4h) * (1 - config.holdout_fraction))
    df_4h_opt = df_4h.iloc[:split_idx]
    df_1m_opt = df_1m[df_1m.index < df_4h.index[split_idx]]

    # ── Step 1: Write mmap files (normally done in MultiObjectiveOptimizer.__init__)
    if not os.path.exists(_MMAP_4H_PATH) or not os.path.exists(_MMAP_1M_PATH):
        print("Building mmap files (one-time) ...")
        np_4h = prepare_4h_arrays(df_4h_opt)
        np_1m = prepare_numpy_arrays(df_1m_opt)
        joblib.dump(np_4h, _MMAP_4H_PATH)
        joblib.dump(np_1m, _MMAP_1M_PATH)
        print("Done.")
    else:
        print("Using existing mmap files.")

    # ── Step 2: Simulate _worker_init (load mmap files)
    print("\n[STEP 2] _worker_init: loading mmap files ...")
    t0 = time.perf_counter()
    _worker_init(_MMAP_4H_PATH, _MMAP_1M_PATH)
    print(f"  _worker_init: {time.perf_counter()-t0:.3f}s")

    np_4h = _opt_mod._NP_4H
    np_1m = _opt_mod._NP_1M
    print(f"  np_4h type: {type(np_4h.close).__name__}  np_1m type: {type(np_1m.close).__name__}")
    print(f"  np_1m.close is mmap: {hasattr(np_1m.close, 'filename')}")

    # ── Step 3: Time each sub-step of _run_trial
    params = GridParams()  # default params

    print("\n[STEP 3] Sub-step timing ...")

    t0 = time.perf_counter()
    df_4h_worker = _numpy_to_dataframe(np_4h)
    print(f"  _numpy_to_dataframe(np_4h): {time.perf_counter()-t0:.3f}s  rows={len(df_4h_worker):,}")

    t0 = time.perf_counter()
    df_1m_worker = _numpy_to_dataframe(np_1m)
    print(f"  _numpy_to_dataframe(np_1m): {time.perf_counter()-t0:.3f}s  rows={len(df_1m_worker):,}")

    # Check index types
    print(f"  df_4h index dtype: {df_4h_worker.index.dtype}  tz={df_4h_worker.index.tz}")
    print(f"  df_1m index dtype: {df_1m_worker.index.dtype}  tz={df_1m_worker.index.tz}")

    # Test _period_1m for first 3 bars
    engine = GridEngine()
    t0 = time.perf_counter()
    total_1m_rows = 0
    for i, bar in enumerate(df_4h_worker.head(5).itertuples()):
        period = engine._period_1m(df_1m_worker, bar.Index)
        total_1m_rows += len(period)
    print(f"  _period_1m x5 bars: {time.perf_counter()-t0:.3f}s  total_1m_rows={total_1m_rows}")

    # ── Step 4: Profile full trial
    print("\n[STEP 4] Full trial profile ...")
    data = DataPair(df_4h=df_4h_worker, df_1m=df_1m_worker, funding=pd.Series(dtype=float))
    bt = Backtester()

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    result = bt.run(params, data, initial_capital=10_000.0, ref_atr=100.0, np_4h=np_4h)
    elapsed = time.perf_counter() - t0
    pr.disable()

    print(f"  Full trial: {elapsed:.2f}s  trades={len(result.trades)}")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print(s.getvalue())

    # ── Step 5: Measure _worker_eval end-to-end
    print("[STEP 5] _worker_eval end-to-end ...")
    t0 = time.perf_counter()
    result2 = _worker_eval(params, pd.Series(dtype=float), 100.0)
    print(f"  _worker_eval: {time.perf_counter()-t0:.2f}s  result={result2}")


if __name__ == "__main__":
    main()
