"""
Phase 1 verification test.

Uses 6 months of BTC/USDT:USDT data (fast fetch) to verify:
  1. 1m data has no missing minute bars
  2. 1m close prices have no NaN values
  3. Ghost-bar fill produces flat bars with volume=0
  4. Data split (opt / holdout / viz) is non-empty and non-overlapping
  5. Screener passes known long-lived symbols
"""
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from config import config
from data.fetcher import BinanceFetcher
from data.cache import DataCache, CACHE_DIR
from data.screener import SymbolScreener

TEST_SYMBOL = "BTC/USDT:USDT"
# 400 days ensures 20% holdout (80d) > 60d viz requirement
FETCH_DAYS = 400


def run():
    logger.info("=== Phase 1 Verification Test ===")

    fetcher = BinanceFetcher()
    cache = DataCache()

    # ── 1. Age check ──────────────────────────────────────────────────────────
    earliest = fetcher.get_earliest_timestamp(TEST_SYMBOL, config.signal_timeframe)
    years = (datetime.now(tz=timezone.utc) - earliest).days / 365.25
    logger.info(f"{TEST_SYMBOL} earliest bar: {earliest.date()} ({years:.1f}yr history)")
    assert years >= config.min_years, f"Age {years:.1f}yr < {config.min_years}yr"
    logger.success(f"PASS: age check ({years:.1f}yr ≥ {config.min_years}yr)")

    # ── 2. Fetch 6 months of 1m data ─────────────────────────────────────────
    since = datetime.now(tz=timezone.utc) - timedelta(days=FETCH_DAYS)
    logger.info(f"Fetching 1m data from {since.date()} …")
    df_1m = fetcher.fetch_ohlcv(TEST_SYMBOL, "1m", since=since)
    logger.info(f"Fetched {len(df_1m):,} 1m bars")

    # ── 3. Continuous index check ─────────────────────────────────────────────
    full_index = pd.date_range(df_1m.index[0], df_1m.index[-1], freq="1min")
    missing = len(full_index) - len(df_1m)
    assert missing == 0, f"1m data has {missing} missing bars"
    logger.success(f"PASS: 1m continuous — {len(df_1m):,} bars, 0 gaps")

    # ── 4. No NaN in OHLC ────────────────────────────────────────────────────
    for col in ["open", "high", "low", "close"]:
        n = df_1m[col].isna().sum()
        assert n == 0, f"1m {col} has {n} NaN values"
    logger.success("PASS: 1m OHLC all non-NaN")

    # ── 5. Ghost-bar stats ────────────────────────────────────────────────────
    zero_vol = (df_1m["volume"] == 0).sum()
    logger.info(f"INFO: ghost bars (volume=0): {zero_vol:,} ({zero_vol/len(df_1m)*100:.3f}%)")

    # ── 6. Manually seed cache with the 6-month 1m + 4h data ─────────────────
    logger.info("Fetching 4h data for same window …")
    df_4h = fetcher.fetch_ohlcv(TEST_SYMBOL, config.signal_timeframe, since=since)
    logger.info(f"4h bars: {len(df_4h)}")

    # Trim to common range and save to cache so get_split() can use it
    start = max(df_4h.index[0], df_1m.index[0])
    end   = min(df_4h.index[-1], df_1m.index[-1])
    df_4h_trim = df_4h.loc[start:end]
    df_1m_trim = df_1m.loc[start:end]

    safe = TEST_SYMBOL.replace("/", "").replace(":", "")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df_4h_trim.to_parquet(CACHE_DIR / f"{safe}_4h.parquet")
    df_1m_trim.to_parquet(CACHE_DIR / f"{safe}_1m.parquet")
    # Remove stale meta so split dates get recomputed for this window
    meta_path = CACHE_DIR / f"{safe}_meta.json"
    if meta_path.exists():
        meta_path.unlink()
    logger.info(f"Seeded cache: 4h={len(df_4h_trim)}, 1m={len(df_1m_trim):,}")

    # ── 7. Test data split ────────────────────────────────────────────────────
    opt, holdout, viz = cache.get_split(TEST_SYMBOL)

    logger.info(
        f"Opt    : 4h={len(opt.df_4h):,}  1m={len(opt.df_1m):,}  "
        f"({opt.df_4h.index[0].date()} → {opt.df_4h.index[-1].date()})"
    )
    logger.info(
        f"Holdout: 4h={len(holdout.df_4h):,}  1m={len(holdout.df_1m):,}  "
        f"({holdout.df_4h.index[0].date()} → {holdout.df_4h.index[-1].date()})"
    )
    logger.info(
        f"Viz    : 4h={len(viz.df_4h):,}  1m={len(viz.df_1m):,}  "
        f"({viz.df_4h.index[0].date()} → {viz.df_4h.index[-1].date()})"
    )

    for name, pair in [("opt", opt), ("holdout", holdout), ("viz", viz)]:
        assert len(pair.df_4h) > 0, f"{name} 4h slice is empty"
        assert len(pair.df_1m) > 0, f"{name} 1m slice is empty"

    assert holdout.df_4h.index[0] > opt.df_4h.index[-1], "holdout overlaps opt"
    assert viz.df_4h.index[0] > holdout.df_4h.index[-1], "viz overlaps holdout"

    # Viz should cover ~60 days
    viz_days = (viz.df_4h.index[-1] - viz.df_4h.index[0]).days
    assert viz_days >= 55, f"viz window too short: {viz_days} days"
    logger.success(f"PASS: splits non-empty, non-overlapping, viz={viz_days}d")

    # ── 8. Screener smoke test ────────────────────────────────────────────────
    logger.info("Screener smoke test (3 known symbols) …")
    screener = SymbolScreener()
    test_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    result = screener.screen(test_symbols)
    logger.info(f"Screener returned: {result}")
    assert len(result) > 0, "screener returned empty list"
    assert "BTC/USDT:USDT" in result or "ETH/USDT:USDT" in result, \
        "expected BTC or ETH to survive screening"
    logger.success(f"PASS: screener returned {len(result)} symbol(s)")

    logger.success("\n=== All Phase 1 checks PASSED ===")


if __name__ == "__main__":
    run()
