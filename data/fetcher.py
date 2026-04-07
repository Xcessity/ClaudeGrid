import time
from datetime import datetime, timezone

import ccxt
import pandas as pd
from loguru import logger

from config import config


class BinanceFetcher:
    """Fetches OHLCV and funding rate data from Binance USDT-M perpetual futures."""

    LIMIT = 1000          # candles per request (Binance max)
    RETRY_ATTEMPTS = 5
    RETRY_DELAY = 2.0     # seconds between retries

    def __init__(self, use_futures: bool = True):
        if use_futures:
            self.exchange = ccxt.binanceusdm({
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
        else:
            self.exchange = ccxt.binance({"enableRateLimit": True})
        self.use_futures = use_futures

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch full OHLCV history for a symbol/timeframe.

        Parameters
        ----------
        since : datetime (UTC) or None — start of range; None = earliest available
        until : datetime (UTC) or None — end of range; None = now

        Returns
        -------
        DataFrame with DatetimeIndex (UTC) and columns: open, high, low, close, volume
        """
        until_ms = int(until.timestamp() * 1000) if until else None

        # When no start is given, paginate from the earliest available bar.
        # Without this, ccxt returns only the most-recent LIMIT bars and the
        # next page starts after them (in the future), so the loop ends with
        # just one page of data instead of the full history.
        if since is None:
            earliest = self.get_earliest_timestamp(symbol, timeframe)
            since_ms = int(earliest.timestamp() * 1000)
        else:
            since_ms = int(since.timestamp() * 1000)

        all_candles: list[list] = []

        while True:
            candles = self._fetch_with_retry(symbol, timeframe, since_ms)
            if not candles:
                break

            # Filter out candles beyond `until`
            if until_ms:
                candles = [c for c in candles if c[0] <= until_ms]

            all_candles.extend(candles)

            if len(candles) < self.LIMIT:
                break  # last page

            last_ts = candles[-1][0]
            if until_ms and last_ts >= until_ms:
                break

            since_ms = last_ts + 1  # next page starts after last candle

        if not all_candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = self._candles_to_df(all_candles)

        # Apply ghost-bar fix for 1m data
        if timeframe == "1m":
            df = self._fill_ghost_bars(df)

        return df

    def fetch_both(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch signal (4h) and fill (1m) timeframes for a symbol.

        Returns (df_4h, df_1m) with matching UTC DatetimeIndex coverage.
        The 1m data has ghost bars forward-filled (see _fill_ghost_bars).
        """
        logger.info(f"Fetching 4h data for {symbol}")
        df_4h = self.fetch_ohlcv(symbol, config.signal_timeframe)

        logger.info(f"Fetching 1m data for {symbol} ({len(df_4h)} 4h bars → ~{len(df_4h)*240} 1m bars expected)")
        df_1m = self.fetch_ohlcv(symbol, config.fill_timeframe)

        # Trim both to common date range
        start = max(df_4h.index[0], df_1m.index[0])
        end   = min(df_4h.index[-1], df_1m.index[-1])

        df_4h = df_4h.loc[start:end]
        df_1m = df_1m.loc[start:end]

        logger.info(
            f"{symbol}: 4h bars={len(df_4h)}, 1m bars={len(df_1m)}, "
            f"range={start.date()} → {end.date()}"
        )
        return df_4h, df_1m

    def fetch_funding_rates(
        self,
        symbol: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> pd.Series:
        """
        Fetch historical 8h funding rates from Binance.

        Returns pd.Series with DatetimeIndex (UTC, ~8h intervals),
        values = funding rate as fraction (e.g. 0.0001 = 0.01%).
        """
        since_ms = int(since.timestamp() * 1000) if since else None
        until_ms = int(until.timestamp() * 1000) if until else None

        all_rates: list[dict] = []
        fetch_since = since_ms

        while True:
            try:
                batch = self.exchange.fetch_funding_rate_history(
                    symbol, since=fetch_since, limit=1000
                )
            except Exception as e:
                logger.warning(f"Funding rate fetch error for {symbol}: {e}")
                break

            if not batch:
                break

            if until_ms:
                batch = [r for r in batch if r["timestamp"] <= until_ms]

            all_rates.extend(batch)

            if len(batch) < 1000:
                break

            last_ts = batch[-1]["timestamp"]
            if until_ms and last_ts >= until_ms:
                break
            fetch_since = last_ts + 1

        if not all_rates:
            return pd.Series(dtype=float, name="funding_rate")

        timestamps = pd.to_datetime([r["timestamp"] for r in all_rates], unit="ms", utc=True)
        rates = [r["fundingRate"] for r in all_rates]
        series = pd.Series(rates, index=timestamps, name="funding_rate")
        series = series[~series.index.duplicated(keep="last")].sort_index()
        return series

    def get_all_usdt_symbols(self) -> list[str]:
        """Return all active USDT-M perpetual symbols."""
        markets = self.exchange.load_markets()
        symbols = [
            s for s, m in markets.items()
            if m.get("quote") == "USDT"
            and m.get("active", True)
            and m.get("type") in ("swap", "future")
            and m.get("linear", True)
        ]
        return sorted(symbols)

    def get_earliest_timestamp(self, symbol: str, timeframe: str) -> datetime:
        """
        Find the earliest available candle timestamp for a symbol.
        Uses a binary-search approach via ccxt's since parameter.
        """
        # Request a single candle from the very beginning of time
        candles = self._fetch_with_retry(symbol, timeframe, since_ms=0, limit=1)
        if not candles:
            raise ValueError(f"No data found for {symbol} at {timeframe}")
        ts_ms = candles[0][0]
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int | None,
        limit: int = None,
    ) -> list[list]:
        limit = limit or self.LIMIT
        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                return self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=limit
                )
            except ccxt.RateLimitExceeded:
                wait = self.RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Rate limit hit for {symbol}, waiting {wait:.1f}s")
                time.sleep(wait)
            except ccxt.NetworkError as e:
                wait = self.RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Network error for {symbol}: {e}, retrying in {wait:.1f}s")
                time.sleep(wait)
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error for {symbol}: {e}")
                raise
        raise RuntimeError(f"Failed to fetch {symbol} {timeframe} after {self.RETRY_ATTEMPTS} attempts")

    def _candles_to_df(self, candles: list[list]) -> pd.DataFrame:
        """Convert raw ccxt candles to a clean DataFrame."""
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.astype(float)
        return df

    def _fill_ghost_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reindex 1m data to a perfect continuous DatetimeIndex, filling gaps.

        Binance occasionally omits 1m bars during low activity or API outages.
        These missing bars would cause NaN indicators and misaligned timestamps
        in the inner simulation loop.

        Fix:
        - Reindex to a complete 1-minute range
        - Forward-fill OHLC (carries last known price → flat bar)
        - Fill volume with 0 (no real activity)

        A flat bar (high == low == close) cannot trigger any limit fills,
        and volume=0 means VPVR correctly ignores these synthetic bars.
        """
        if df.empty:
            return df

        full_index = pd.date_range(df.index[0], df.index[-1], freq="1min")
        df = df.reindex(full_index)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
        df["volume"] = df["volume"].fillna(0.0)
        return df
