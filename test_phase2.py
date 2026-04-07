"""
test_phase2.py — Phase 2 simulation smoke test.

Loads cached BTC/USDT data written by Phase 1 (no network calls).
Runs a single backtest with dummy GridParams on the optimization slice.
Prints the resulting metrics dictionary.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from data.cache import CACHE_DIR, DataPair
from strategy.grid_engine import GridParams
from backtester.engine import Backtester
from backtester.metrics import compute_metrics

TEST_SYMBOL = "BTC/USDT:USDT"


def load_cached_data() -> tuple:
    """
    Load parquet files written by Phase 1 and apply the stored split dates.
    Returns (opt_pair, holdout_pair, viz_pair) as DataPair namedtuples.
    No network calls — uses on-disk cache only.
    """
    safe = TEST_SYMBOL.replace("/", "").replace(":", "")

    path_4h   = CACHE_DIR / f"{safe}_4h.parquet"
    path_1m   = CACHE_DIR / f"{safe}_1m.parquet"
    path_fund = CACHE_DIR / f"{safe}_funding.parquet"
    path_meta = CACHE_DIR / f"{safe}_meta.json"

    for p in (path_4h, path_1m, path_meta):
        if not p.exists():
            logger.error(f"Missing cache file: {p}")
            logger.error("Run test_phase1.py first to populate the cache.")
            sys.exit(1)

    logger.info("Loading 4h data …")
    df_4h = pd.read_parquet(path_4h)
    logger.info(f"  4h: {len(df_4h):,} bars  {df_4h.index[0].date()} → {df_4h.index[-1].date()}")

    logger.info("Loading 1m data …")
    df_1m = pd.read_parquet(path_1m)
    logger.info(f"  1m: {len(df_1m):,} bars  {df_1m.index[0].date()} → {df_1m.index[-1].date()}")

    if path_fund.exists():
        funding = pd.read_parquet(path_fund).squeeze()
        if isinstance(funding, pd.DataFrame):
            funding = funding.iloc[:, 0]
        logger.info(f"  funding: {len(funding):,} entries")
    else:
        funding = pd.Series(dtype=float)
        logger.warning("No funding cache found — funding payments disabled for this test.")

    with open(path_meta) as f:
        meta = json.load(f)
    split_date = pd.Timestamp(meta["split_date"], tz="UTC")
    viz_date   = pd.Timestamp(meta["viz_date"],   tz="UTC")
    logger.info(f"  split_date={split_date.date()}  viz_date={viz_date.date()}")

    def _slice(df, start, end):
        return df.loc[start:end]

    def _fund_slice(start, end):
        return funding.loc[start:end] if not funding.empty else funding

    start = df_4h.index[0]
    end   = df_4h.index[-1]

    opt_pair = DataPair(
        df_4h=_slice(df_4h, start,      split_date - pd.Timedelta("4h")),
        df_1m=_slice(df_1m, start,      split_date - pd.Timedelta("1min")),
        funding=_fund_slice(start,      split_date),
    )
    holdout_pair = DataPair(
        df_4h=_slice(df_4h, split_date, viz_date - pd.Timedelta("4h")),
        df_1m=_slice(df_1m, split_date, viz_date - pd.Timedelta("1min")),
        funding=_fund_slice(split_date, viz_date),
    )
    viz_pair = DataPair(
        df_4h=_slice(df_4h, viz_date,   end),
        df_1m=_slice(df_1m, viz_date,   end),
        funding=_fund_slice(viz_date,   end),
    )

    return opt_pair, holdout_pair, viz_pair


def run():
    logger.info("=== Phase 2 Simulation Test ===\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    opt, holdout, viz = load_cached_data()
    logger.info(
        f"\nData splits:"
        f"\n  Opt     : 4h={len(opt.df_4h):>5}  1m={len(opt.df_1m):>8,}"
        f"  {opt.df_4h.index[0].date()} → {opt.df_4h.index[-1].date()}"
        f"\n  Holdout : 4h={len(holdout.df_4h):>5}  1m={len(holdout.df_1m):>8,}"
        f"  {holdout.df_4h.index[0].date()} → {holdout.df_4h.index[-1].date()}"
        f"\n  Viz     : 4h={len(viz.df_4h):>5}  1m={len(viz.df_1m):>8,}"
        f"  {viz.df_4h.index[0].date()} → {viz.df_4h.index[-1].date()}\n"
    )

    # ── 2. Dummy GridParams ───────────────────────────────────────────────────
    params = GridParams(
        atr_period       = 14,
        atr_multiplier   = 0.5,
        geometric_ratio  = 1.2,
        n_levels         = 3,
        pullback_pct     = 1.5,
        hurst_window     = 100,
        hurst_threshold  = 0.55,
        adx_period       = 14,
        adx_threshold    = 25.0,
        vpvr_window      = 100,
        use_vpvr_anchor  = False,   # False = no VPVR anchor (faster for smoke test)
        position_size_pct= 2.0,
        max_open_levels  = 4,
    )
    logger.info(f"GridParams: {params}\n")

    # ── 3. Run backtest on optimization set ───────────────────────────────────
    backtester = Backtester(ref_data=opt)
    logger.info("Running backtest on optimization set …")
    result = backtester.run(params, data=opt, initial_capital=10_000.0)

    logger.info(
        f"\nSimulation complete:"
        f"\n  Equity points : {len(result.equity_curve):,}"
        f"\n  Closed trades : {len(result.trades)}"
        f"\n  Grid resets   : {result.n_grid_resets}"
        f"\n  Regime pauses : {result.n_regime_pauses}"
        f"\n  Killed early  : {result.killed_early}"
    )

    if len(result.equity_curve) > 0:
        start_eq = result.equity_curve.iloc[0]
        end_eq   = result.equity_curve.iloc[-1]
        logger.info(
            f"\n  Start equity  : ${start_eq:,.2f}"
            f"\n  End equity    : ${end_eq:,.2f}"
            f"\n  Return        : {(end_eq / start_eq - 1) * 100:.2f}%"
        )

    # ── 4. Compute and print metrics ──────────────────────────────────────────
    metrics = compute_metrics(result.equity_curve, result.trades, result.killed_early)

    logger.info("\n── Metrics ──────────────────────────────────────")
    for key, val in metrics.items():
        if isinstance(val, float):
            logger.info(f"  {key:<25s} {val:>10.4f}")
        else:
            logger.info(f"  {key:<25s} {val!r:>10}")

    # ── 5. Assertions ─────────────────────────────────────────────────────────
    assert isinstance(metrics, dict), "metrics must be a dict"
    assert "sharpe" in metrics, "sharpe missing from metrics"
    assert "max_drawdown" in metrics, "max_drawdown missing from metrics"
    assert 0.0 <= metrics["max_drawdown"] <= 1.0, f"max_drawdown out of range: {metrics['max_drawdown']}"
    assert len(result.equity_curve) > 0, "equity_curve is empty"

    logger.success("\n=== Phase 2 test PASSED ===")


if __name__ == "__main__":
    run()
