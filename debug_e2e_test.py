"""
debug_e2e_test.py  —  End-to-end timing test for the optimization pipeline.

Run:  python debug_e2e_test.py [n_trials] [n_workers]
      python debug_e2e_test.py 10 4
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import config
from data.cache import CACHE_DIR, DataPair
from optimization.optimizer import MultiObjectiveOptimizer


def main() -> None:
    n_trials  = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith("_4h.parquet"))
    if not files:
        print("No cached data — run main.py once first.")
        sys.exit(1)

    sym = files[0].replace("_4h.parquet", "")
    print(f"Symbol: {sym}  |  n_trials={n_trials}  n_workers={n_workers}")

    df_4h = pd.read_parquet(CACHE_DIR / files[0])
    df_1m = pd.read_parquet(CACHE_DIR / files[0].replace("_4h", "_1m"))
    split_idx = int(len(df_4h) * (1 - config.holdout_fraction))
    opt_data = DataPair(
        df_4h=df_4h.iloc[:split_idx],
        df_1m=df_1m[df_1m.index < df_4h.index[split_idx]],
        funding=pd.Series(dtype=float),
    )
    print(f"4h bars: {len(opt_data.df_4h):,}  |  1m bars: {len(opt_data.df_1m):,}")

    t0 = time.perf_counter()
    optimizer = MultiObjectiveOptimizer(opt_data=opt_data, n_trials=n_trials, n_workers=n_workers)
    init_t = time.perf_counter() - t0
    print(f"prepare_4h_arrays + dump: {init_t:.1f}s  |  effective workers: {optimizer.n_workers}")

    t0 = time.perf_counter()
    study = optimizer.run()
    run_t = time.perf_counter() - t0

    n_done = len(study.trials)
    print(f"\nCompleted {n_done} trials in {run_t:.1f}s  ({run_t/max(n_done,1):.2f}s/trial)")
    print(f"Pareto-front trials: {len(study.best_trials)}")
    print(f"Projected 500 trials: {run_t/max(n_done,1)*500/60:.1f} min")


if __name__ == "__main__":
    main()
