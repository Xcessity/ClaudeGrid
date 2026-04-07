"""
test_phase3_optuna.py

Miniature smoke test for Phase 3 — verifies that:
  1. prepare_4h_arrays() completes (pre-computes all indicator lookups).
  2. NSGAIISampler study runs N_TRIALS trials via loky parallel workers.
  3. Workers successfully receive the memory-mapped NumPy arrays without crashing.
  4. get_pareto_front() and get_stable_params() return results without error.

Uses synthetic price data — no real Binance cache needed.

Run:
    python test_phase3_optuna.py
"""
import sys
import time

import numpy as np
import pandas as pd

# ── Synthetic data generator ───────────────────────────────────────────────────

def _make_ohlcv(
    n_bars:        int,
    start_price:   float = 30_000.0,
    daily_vol:     float = 0.02,
    freq:          str   = "4h",
    seed:          int   = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame using geometric Brownian motion.
    OHLC is self-consistent: high ≥ max(open, close), low ≤ min(open, close).
    """
    rng    = np.random.default_rng(seed)
    bars_per_day = {"4h": 6, "1min": 1440, "1m": 1440}[freq]
    bar_vol = daily_vol / np.sqrt(bars_per_day)

    # Log-normal close prices
    log_ret = rng.normal(0, bar_vol, n_bars)
    closes  = start_price * np.exp(np.cumsum(log_ret))

    # Realistic OHLC from close path
    opens   = np.concatenate([[start_price], closes[:-1]])
    intra   = rng.uniform(0.0005, bar_vol * 1.5, n_bars)   # intrabar spread
    highs   = np.maximum(opens, closes) * (1.0 + intra)
    lows    = np.minimum(opens, closes) * (1.0 - intra)
    volumes = rng.uniform(1_000, 50_000, n_bars)

    end   = pd.Timestamp("2023-01-01", tz="UTC")
    freq_td = {"4h": "4h", "1min": "1min", "1m": "1min"}[freq]
    index = pd.date_range(end=end, periods=n_bars, freq=freq_td)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": volumes},
        index=index,
    )


def _make_funding(index_4h: pd.DatetimeIndex) -> pd.Series:
    """Tiny 8h funding-rate series covering the same date range."""
    rng     = np.random.default_rng(0)
    n_fund  = max(len(index_4h) // 2, 1)   # ~2 4h bars per 8h period
    idx     = pd.date_range(
        start=index_4h[0], periods=n_fund, freq="8h", tz="UTC"
    )
    return pd.Series(rng.uniform(-0.0001, 0.0003, n_fund), index=idx)


# ── Main test ──────────────────────────────────────────────────────────────────

def main():
    N_TRIALS  = 5
    N_4H_BARS = 400    # warmup needs ≤200 bars; leaves ~200 bars for trading
    N_1M_BARS = N_4H_BARS * 240   # 96,000 bars → ~3.8 MB per array (above mmap threshold)
    N_JOBS    = 2      # force parallel execution to exercise loky mmap path

    print("=" * 60)
    print("Phase 3 smoke test — Optuna NSGAIISampler + loky mmap")
    print("=" * 60)
    print(f"  4h bars : {N_4H_BARS}")
    print(f"  1m bars : {N_1M_BARS:,}  ({N_1M_BARS * 5 * 8 / 1e6:.1f} MB of float64 arrays)")
    print(f"  trials  : {N_TRIALS}")
    print(f"  n_jobs  : {N_JOBS}")
    print()

    # ── Build synthetic DataPair ───────────────────────────────────────────────
    print("[1/4] Generating synthetic OHLCV data …")
    t0 = time.perf_counter()

    df_4h     = _make_ohlcv(N_4H_BARS, freq="4h",  seed=42)
    df_1m     = _make_ohlcv(N_1M_BARS, freq="1m",  seed=99, start_price=df_4h["close"].iloc[0])
    funding   = _make_funding(df_4h.index)

    from data.cache import DataPair
    opt_data  = DataPair(df_4h=df_4h, df_1m=df_1m, funding=funding)
    print(f"   done in {time.perf_counter() - t0:.1f}s")
    print(f"   4h shape: {df_4h.shape}  |  1m shape: {df_1m.shape}")
    print()

    # ── prepare_4h_arrays ─────────────────────────────────────────────────────
    print("[2/4] Calling prepare_4h_arrays() — pre-computes all indicator lookups …")
    t0 = time.perf_counter()

    from optimization.optimizer import prepare_4h_arrays, prepare_numpy_arrays
    np_4h = prepare_4h_arrays(df_4h)
    np_1m = prepare_numpy_arrays(df_1m)

    elapsed = time.perf_counter() - t0
    print(f"   done in {elapsed:.1f}s")
    print(f"   ATR arrays  : {len(np_4h.atr_arrays)}")
    print(f"   ADX arrays  : {len(np_4h.adx_arrays)}")
    print(f"   Hurst arrays: {len(np_4h.hurst_arrays)}")
    print(f"   np_1m close : shape={np_1m.close.shape}, "
          f"dtype={np_1m.close.dtype}, "
          f"size={np_1m.close.nbytes / 1e6:.2f} MB")
    assert len(np_4h.atr_arrays) == 41,   f"Expected 41 ATR arrays, got {len(np_4h.atr_arrays)}"
    assert len(np_4h.adx_arrays) == 21,   f"Expected 21 ADX arrays, got {len(np_4h.adx_arrays)}"
    assert len(np_4h.hurst_arrays) == 151, f"Expected 151 Hurst arrays, got {len(np_4h.hurst_arrays)}"
    print("   [OK] indicator array counts match PARAM_SPACE bounds")
    print()

    # ── NSGAIISampler study ───────────────────────────────────────────────────
    print(f"[3/4] Running Optuna NSGAIISampler study ({N_TRIALS} trials, n_jobs={N_JOBS}) …")
    t0 = time.perf_counter()

    from optimization.optimizer import MultiObjectiveOptimizer
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    opt = MultiObjectiveOptimizer(
        opt_data=opt_data,
        n_trials=N_TRIALS,
        n_workers=N_JOBS,
    )
    study = opt.run()

    elapsed = time.perf_counter() - t0
    n_complete = len([t for t in study.trials if t.state.is_finished()])
    print(f"   done in {elapsed:.1f}s  |  completed trials: {n_complete}/{N_TRIALS}")

    # Print trial results
    print()
    print("   Trial values (sharpe, sortino, max_dd):")
    for t in study.trials:
        state = "OK" if t.state.is_finished() else t.state.name
        vals  = t.values if t.values else ["—", "—", "—"]
        print(f"     trial {t.number:2d}: {vals}  [{state}]")

    assert n_complete == N_TRIALS, \
        f"Only {n_complete}/{N_TRIALS} trials completed — loky workers may have crashed"
    print()
    print("   [OK] all trials completed without crash")
    print()

    # ── Pareto front & stable params ──────────────────────────────────────────
    print("[4/4] Extracting Pareto front and stable params …")
    pareto  = opt.get_pareto_front(study)
    plateau = opt.score_plateau_width(study)
    stable  = opt.get_stable_params(study)

    print(f"   Pareto-front trials (quality-filtered): {len(pareto)}")
    print(f"   avg plateau width score: {sum(plateau.values()) / max(len(plateau), 1):.3f}")
    print(f"   Stable params returned : {len(stable)}")
    # These may be 0 with synthetic random data (strategy likely underperforms
    # quality thresholds).  What matters is no exception was raised.
    print("   [OK] get_pareto_front / get_stable_params completed without error")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("PASS — Phase 3 smoke test completed successfully.")
    print("  - prepare_4h_arrays: indicator lookups built for entire PARAM_SPACE")
    print("  - loky workers read memory-mapped NumPy arrays without crashing")
    print(f"  - {N_TRIALS} Optuna NSGAIISampler trials completed")
    print("=" * 60)


if __name__ == "__main__":
    # Windows requires the __main__ guard when spawning loky worker processes.
    main()
