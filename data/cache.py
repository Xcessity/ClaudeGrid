import json
import os
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from config import config
from data.fetcher import BinanceFetcher

# DataPair carries matching slices of both timeframes + funding for one date range
DataPair = namedtuple("DataPair", ["df_4h", "df_1m", "funding"])

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


class DataCache:
    """
    Parquet-backed cache for 4h, 1m, and funding data.

    Cache files:
        data/cache/{symbol}_4h.parquet
        data/cache/{symbol}_1m.parquet
        data/cache/{symbol}_funding.parquet
        data/cache/{symbol}_meta.json      ← stores split dates
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = BinanceFetcher()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_or_fetch(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Return full history for one timeframe.
        Appends only new bars if cache already exists and is fresh enough.
        """
        path = self._parquet_path(symbol, timeframe)

        if path.exists():
            df = pd.read_parquet(path)
            if self.is_fresh(symbol, timeframe):
                return df
            # Incremental update: fetch only bars after the last cached bar
            last_ts = df.index[-1]
            since = last_ts + pd.Timedelta("1" + ("h" if timeframe == "4h" else "min"))
            logger.info(f"Updating {symbol} {timeframe} from {since.date()}")
            new_df = self.fetcher.fetch_ohlcv(
                symbol, timeframe, since=since.to_pydatetime()
            )
            if not new_df.empty:
                df = pd.concat([df, new_df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.to_parquet(path)
            return df

        logger.info(f"Fetching full {timeframe} history for {symbol}")
        df = self.fetcher.fetch_ohlcv(symbol, timeframe)
        df.to_parquet(path)
        return df

    def get_or_fetch_both(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (df_4h, df_1m). Fetches/updates whichever is missing or stale.
        Both are trimmed to the same common date range.
        """
        df_4h = self.get_or_fetch(symbol, config.signal_timeframe)
        df_1m = self.get_or_fetch(symbol, config.fill_timeframe)

        # Also ensure funding is cached
        self._get_or_fetch_funding(symbol, df_1m.index[0], df_1m.index[-1])

        # Trim to common range
        start = max(df_4h.index[0], df_1m.index[0])
        end   = min(df_4h.index[-1], df_1m.index[-1])
        return df_4h.loc[start:end], df_1m.loc[start:end]

    def get_split(self, symbol: str) -> tuple[DataPair, DataPair, DataPair]:
        """
        Return (opt_pair, holdout_pair, viz_pair) — the three non-overlapping data slices.

        Split boundaries (computed once, stored in meta.json):
            split_date = 80th percentile date  (optimization / holdout boundary)
            viz_date   = most_recent - 60 days (holdout / visualization boundary)

        Returns
        -------
        opt_pair      DataPair for [start, split_date)
        holdout_pair  DataPair for [split_date, viz_date)
        viz_pair      DataPair for [viz_date, end]
        """
        df_4h, df_1m = self.get_or_fetch_both(symbol)
        funding = self._get_or_fetch_funding(symbol, df_1m.index[0], df_1m.index[-1])

        split_date, viz_date = self._get_or_compute_split_dates(symbol, df_4h)

        def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
            return df.loc[start:end]

        def _fund_slice(start, end) -> pd.Series:
            return funding.loc[start:end] if not funding.empty else funding

        start = df_4h.index[0]
        end   = df_4h.index[-1]

        opt_pair = DataPair(
            df_4h=_slice(df_4h, start, split_date - pd.Timedelta("4h")),
            df_1m=_slice(df_1m, start, split_date - pd.Timedelta("1min")),
            funding=_fund_slice(start, split_date),
        )
        holdout_pair = DataPair(
            df_4h=_slice(df_4h, split_date, viz_date - pd.Timedelta("4h")),
            df_1m=_slice(df_1m, split_date, viz_date - pd.Timedelta("1min")),
            funding=_fund_slice(split_date, viz_date),
        )
        viz_pair = DataPair(
            df_4h=_slice(df_4h, viz_date, end),
            df_1m=_slice(df_1m, viz_date, end),
            funding=_fund_slice(viz_date, end),
        )
        return opt_pair, holdout_pair, viz_pair

    def is_fresh(self, symbol: str, timeframe: str, max_age_hours: int = 4) -> bool:
        """Return True if the cached file exists and is younger than max_age_hours."""
        path = self._parquet_path(symbol, timeframe)
        if not path.exists():
            return False
        age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
        return age_hours < max_age_hours

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _parquet_path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace("/", "").replace(":", "")
        return self.cache_dir / f"{safe}_{timeframe}.parquet"

    def _meta_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "").replace(":", "")
        return self.cache_dir / f"{safe}_meta.json"

    def _get_or_fetch_funding(
        self,
        symbol: str,
        since: pd.Timestamp,
        until: pd.Timestamp,
    ) -> pd.Series:
        path = self._parquet_path(symbol, "funding")

        if path.exists():
            series = pd.read_parquet(path).squeeze()
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            # Extend if needed
            if series.index[-1] < until:
                new_series = self.fetcher.fetch_funding_rates(
                    symbol,
                    since=series.index[-1].to_pydatetime(),
                    until=until.to_pydatetime(),
                )
                if not new_series.empty:
                    series = pd.concat([series, new_series])
                    series = series[~series.index.duplicated(keep="last")].sort_index()
                    series.to_frame("funding_rate").to_parquet(path)
            return series

        logger.info(f"Fetching funding rates for {symbol}")
        series = self.fetcher.fetch_funding_rates(
            symbol,
            since=since.to_pydatetime(),
            until=until.to_pydatetime(),
        )
        if not series.empty:
            series.to_frame("funding_rate").to_parquet(path)
        return series

    def _get_or_compute_split_dates(
        self,
        symbol: str,
        df_4h: pd.DataFrame,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Return (split_date, viz_date), computing and persisting them on first call.
        Dates are never recomputed after first call to prevent contamination.
        """
        meta_path = self._meta_path(symbol)

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            split_date = pd.Timestamp(meta["split_date"], tz="UTC")
            viz_date   = pd.Timestamp(meta["viz_date"],   tz="UTC")
            return split_date, viz_date

        # Compute split boundaries
        n_4h_bars = len(df_4h)
        split_idx  = int(n_4h_bars * (1 - config.holdout_fraction))
        split_date = df_4h.index[split_idx]

        most_recent = df_4h.index[-1]
        viz_date    = most_recent - pd.Timedelta(days=config.viz_lookback_days)

        # Sanity check: holdout must be non-empty
        if viz_date <= split_date:
            raise ValueError(
                f"{symbol}: viz_date ({viz_date.date()}) ≤ split_date ({split_date.date()}). "
                "Not enough history for a 60-day viz slice within the holdout period."
            )

        meta = {
            "split_date": split_date.isoformat(),
            "viz_date":   viz_date.isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"{symbol} split dates: opt=[start,{split_date.date()}), "
            f"holdout=[{split_date.date()},{viz_date.date()}), "
            f"viz=[{viz_date.date()},end]"
        )
        return split_date, viz_date
