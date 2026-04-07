from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

from config import config
from data.fetcher import BinanceFetcher


class SymbolScreener:
    """
    Screens USDT-M perpetual symbols by age, liquidity, and correlation.

    Filtering pipeline:
        1. Age filter      — symbol must have ≥ min_years of 4h history
        2. Liquidity filter — avg 30-day volume ≥ min_volume_usdt
        3. Correlation filter — cluster by Pearson correlation (threshold=0.85),
                                keep one representative per cluster (highest volume)

    Returns list sorted by volume descending, max ~30 symbols.
    """

    CORRELATION_THRESHOLD = 0.85
    MAX_SYMBOLS = 30
    VOLUME_LOOKBACK_DAYS = 30
    CORRELATION_LOOKBACK_DAYS = 730  # 2 years

    def __init__(self):
        self.fetcher = BinanceFetcher()

    def screen(self, symbols: list[str], timeframe: str = None) -> list[str]:
        if timeframe is None:
            timeframe = config.signal_timeframe

        logger.info(f"Screening {len(symbols)} symbols (timeframe={timeframe})")

        # ── Step 1: Age filter ────────────────────────────────────────────────
        age_passed: list[tuple[str, datetime]] = []
        for sym in symbols:
            try:
                earliest = self.fetcher.get_earliest_timestamp(sym, timeframe)
                years = (datetime.now(tz=timezone.utc) - earliest).days / 365.25
                if years >= config.min_years:
                    age_passed.append((sym, earliest))
                else:
                    logger.debug(f"  {sym}: age={years:.1f}y < {config.min_years}y — skipped")
            except Exception as e:
                logger.warning(f"  {sym}: age check failed ({e}) — skipped")

        logger.info(f"Age filter: {len(age_passed)}/{len(symbols)} passed (≥{config.min_years}yr)")

        if not age_passed:
            return []

        # ── Step 2: Liquidity filter ──────────────────────────────────────────
        liquidity_passed: list[tuple[str, float]] = []
        for sym, _ in age_passed:
            try:
                avg_vol = self._avg_daily_volume(sym, timeframe)
                if avg_vol >= config.min_volume_usdt:
                    liquidity_passed.append((sym, avg_vol))
                    logger.debug(f"  {sym}: avg_vol=${avg_vol:,.0f} — passed")
                else:
                    logger.debug(f"  {sym}: avg_vol=${avg_vol:,.0f} < ${config.min_volume_usdt:,.0f} — skipped")
            except Exception as e:
                logger.warning(f"  {sym}: volume check failed ({e}) — skipped")

        # Sort by volume descending
        liquidity_passed.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Liquidity filter: {len(liquidity_passed)} passed (≥${config.min_volume_usdt:,.0f})")

        if not liquidity_passed:
            return []

        if len(liquidity_passed) == 1:
            return [liquidity_passed[0][0]]

        # ── Step 3: Correlation filter ────────────────────────────────────────
        syms_sorted = [s for s, _ in liquidity_passed]
        vol_map = {s: v for s, v in liquidity_passed}

        daily_returns = self._fetch_daily_returns(syms_sorted)
        survivors = self._cluster_and_select(daily_returns, vol_map)

        # Enforce MAX_SYMBOLS cap (already sorted by volume)
        survivors = survivors[: self.MAX_SYMBOLS]

        logger.info(
            f"Correlation filter: {len(survivors)} symbols remaining "
            f"(threshold={self.CORRELATION_THRESHOLD})"
        )
        return survivors

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _avg_daily_volume(self, symbol: str, timeframe: str) -> float:
        """Compute average daily USDT volume over the last 30 days."""
        # Fetch recent 4h bars (30 days × 6 bars/day = 180 bars)
        bars_needed = self.VOLUME_LOOKBACK_DAYS * 6
        candles = self.fetcher._fetch_with_retry(
            symbol, timeframe, since_ms=None, limit=bars_needed
        )
        if not candles:
            return 0.0
        df = self.fetcher._candles_to_df(candles)
        # Volume in quote (USDT): volume × close price
        daily_vol = (df["volume"] * df["close"]).resample("1D").sum()
        return float(daily_vol.tail(self.VOLUME_LOOKBACK_DAYS).mean())

    def _fetch_daily_returns(self, symbols: list[str]) -> pd.DataFrame:
        """
        Fetch ~2 years of daily close prices and compute log-returns.
        Returns DataFrame: index=dates, columns=symbols.
        Missing symbols are dropped silently.
        """
        closes: dict[str, pd.Series] = {}

        for sym in symbols:
            try:
                candles = self.fetcher._fetch_with_retry(
                    sym, "1d", since_ms=None, limit=self.CORRELATION_LOOKBACK_DAYS
                )
                if candles:
                    df = self.fetcher._candles_to_df(candles)
                    closes[sym] = df["close"]
            except Exception as e:
                logger.warning(f"  {sym}: daily fetch for correlation failed ({e})")

        if not closes:
            return pd.DataFrame()

        close_df = pd.DataFrame(closes).dropna(how="all")
        returns = np.log(close_df / close_df.shift(1)).dropna(how="all")
        return returns

    def _cluster_and_select(
        self,
        returns: pd.DataFrame,
        vol_map: dict[str, float],
    ) -> list[str]:
        """
        Single-linkage correlation clustering: remove correlated duplicates.
        Keeps one representative per cluster (highest volume).
        Input symbols are assumed sorted by volume descending.
        """
        if returns.empty or len(returns.columns) == 1:
            return list(returns.columns)

        # Pearson correlation matrix on overlapping periods
        corr = returns.corr(method="pearson").fillna(0)

        symbols = list(corr.columns)
        excluded: set[str] = set()
        survivors: list[str] = []

        # Greedy: process highest-volume symbol first (already sorted)
        for sym in symbols:
            if sym in excluded:
                continue
            survivors.append(sym)
            # Exclude all highly correlated peers (except self)
            for other in symbols:
                if other == sym or other in excluded:
                    continue
                if sym in corr.index and other in corr.columns:
                    if corr.loc[sym, other] >= self.CORRELATION_THRESHOLD:
                        excluded.add(other)
                        logger.debug(
                            f"  Excluding {other} (corr={corr.loc[sym, other]:.2f} with {sym})"
                        )

        return survivors
