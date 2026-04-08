"""
strategy/indicators.py

Vectorized indicator functions for the grid trading strategy.
All inputs are DataFrames with OHLCV columns: open, high, low, close, volume.
Relies on pandas-ta for ATR, ADX, and Bollinger Bands.
grid_engine.py receives the pre-computed Series/DataFrames returned here.
"""
import math

import numpy as np
import pandas as pd
import pandas_ta as ta


# ── ATR ───────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR via pandas-ta. Returns Series aligned to df.index."""
    result = df.ta.atr(length=period)
    result.name = f"atr_{period}"
    return result


# ── ADX + DI +/- ──────────────────────────────────────────────────────────────

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    ADX + DI+/DI- via pandas-ta.
    Returns DataFrame with columns: ADX, DI_plus, DI_minus.
    """
    raw = df.ta.adx(length=period)
    # Locate columns by prefix to be version-agnostic
    adx_col = next(c for c in raw.columns if c.startswith("ADX"))
    dmp_col = next(c for c in raw.columns if c.startswith("DMP"))
    dmn_col = next(c for c in raw.columns if c.startswith("DMN"))
    return pd.DataFrame(
        {"ADX": raw[adx_col], "DI_plus": raw[dmp_col], "DI_minus": raw[dmn_col]},
        index=df.index,
    )


# ── Hurst DFA ─────────────────────────────────────────────────────────────────

def compute_hurst_dfa(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling Detrended Fluctuation Analysis (DFA) Hurst exponent.

    For each window-sized rolling period ending at bar i:
      1. Compute integrated series Y = cumsum(series - mean)
      2. For ~10 log-spaced box sizes n:
           - Divide Y into non-overlapping segments of length n
           - Fit linear trend per segment, compute RMS of residuals → F(n)
      3. H = slope of log(F(n)) vs log(n)

    H < 0.45 → mean-reverting (grid-friendly)
    H > 0.55 → trending       (regime filter — pause grid)
    0.45–0.55 → random walk

    Returns NaN for first `window - 1` bars (warm-up period).

    Input: price-level series (converted internally to log-returns so that
    DFA measures return autocorrelation, not price-level trend).
    Passing raw prices gives H ≈ 1.0 always; log-returns give sensible 0–1 values.

    Implementation: vectorised over all rolling windows simultaneously using
    numpy.lib.stride_tricks.sliding_window_view.  The outer Python loop that
    iterates over each bar has been replaced by batched matrix operations,
    reducing 4.5 M per-window function calls (and ~200 s for 151 windows) to
    ~10 numpy calls that complete in <1 s.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    prices   = series.to_numpy(dtype=np.float64)
    log_ret  = np.diff(np.log(np.maximum(prices, 1e-10)))
    values   = np.concatenate([[0.0], log_ret])   # aligned with series
    n_total  = len(values)
    hurst_vals = np.full(n_total, np.nan)

    min_box = 4
    max_box = max(min_box + 1, window // 4)
    n_sizes = min(10, max_box - min_box + 1)
    box_sizes = np.unique(
        np.round(
            np.exp(np.linspace(np.log(min_box), np.log(max_box), n_sizes))
        ).astype(int)
    )
    box_sizes = box_sizes[box_sizes >= min_box]

    if len(box_sizes) < 3 or n_total < window:
        return pd.Series(hurst_vals, index=series.index, name=f"hurst_{window}")

    # ── Build all rolling windows at once (zero-copy stride view) ─────────────
    # Shape: (n_windows, window)  where n_windows = n_total - window + 1
    wins      = sliding_window_view(values, window)        # (W, window)
    n_windows = wins.shape[0]

    # Demean each window then cumsum → integrated series for every window
    # Shape: (W, window)
    y_all = np.cumsum(wins - wins.mean(axis=1, keepdims=True), axis=1)

    # ── For each box size, compute RMS of DFA residuals across ALL windows ────
    collected_log_n: list[float]       = []
    collected_rms:   list[np.ndarray]  = []   # each entry: (W,)

    for n in box_sizes.tolist():
        n_boxes = window // n
        if n_boxes < 1:
            continue
        usable = n_boxes * n

        # (W, n_boxes, n)  — reshape without copy when possible
        blocks = y_all[:, :usable].reshape(n_windows, n_boxes, n)

        # Vectorised OLS detrend inside each block
        # trend(t) = slope*(t - t_m) + block_mean
        t       = np.arange(n, dtype=np.float64)
        t_dev   = t - t.mean()                            # (n,)
        t_var   = (t_dev ** 2).sum()
        if t_var == 0:
            continue

        # slopes: (W, n_boxes)
        slopes = (blocks * t_dev).sum(axis=2) / t_var
        # trends: (W, n_boxes, n)
        trends = slopes[:, :, None] * t_dev + blocks.mean(axis=2)[:, :, None]
        # rms per window: (W,)
        rms = np.sqrt(((blocks - trends) ** 2).mean(axis=(1, 2)))

        collected_log_n.append(math.log(n))
        collected_rms.append(rms)

    if len(collected_log_n) < 3:
        return pd.Series(hurst_vals, index=series.index, name=f"hurst_{window}")

    # ── Vectorised OLS: slope of log(F) vs log(n) for every window at once ───
    log_ns  = np.array(collected_log_n, dtype=np.float64)  # (B,)
    rms_mat = np.stack(collected_rms,   axis=0)             # (B, W)

    # Only trust windows where every box size produced rms > 0
    valid = (rms_mat > 0).all(axis=0)                       # (W,)

    if not valid.any():
        return pd.Series(hurst_vals, index=series.index, name=f"hurst_{window}")

    log_fs = np.where(rms_mat > 0, np.log(np.maximum(rms_mat, 1e-300)), 0.0)  # (B, W)

    B    = float(len(log_ns))
    sx   = log_ns.sum()
    sx2  = (log_ns ** 2).sum()
    denom = B * sx2 - sx ** 2

    if denom == 0:
        return pd.Series(hurst_vals, index=series.index, name=f"hurst_{window}")

    sxy = (log_ns[:, None] * log_fs).sum(axis=0)  # (W,)
    sy  =  log_fs.sum(axis=0)                      # (W,)

    slopes_all = (B * sxy - sx * sy) / denom       # (W,)
    slopes_clipped = np.clip(slopes_all, 0.0, 1.0)

    # Place results: index window-1 in original series = window index 0
    hurst_vals[window - 1:][valid] = slopes_clipped[valid]

    return pd.Series(hurst_vals, index=series.index, name=f"hurst_{window}")


# ── VPVR ──────────────────────────────────────────────────────────────────────

def compute_vpvr(df: pd.DataFrame, lookback: int, n_bins: int = 50) -> dict:
    """
    Volume Profile Visible Range using the last `lookback` bars.

    Each bar's volume is distributed uniformly across its [low, high] price range
    (not just at close — captures intrabar price-volume distribution).

    Returns:
        poc  – price bin with highest accumulated volume
        hvn  – bin centers with volume >= 70th percentile
        lvn  – bin centers with volume <= 30th percentile
    """
    if len(df) == 0:
        return {"poc": 0.0, "hvn": [], "lvn": []}

    subset = df.iloc[-lookback:] if lookback < len(df) else df

    price_min = float(subset["low"].min())
    price_max = float(subset["high"].max())

    if price_min >= price_max:
        mid = (price_min + price_max) / 2.0
        return {"poc": mid, "hvn": [mid], "lvn": []}

    bin_edges   = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    vol_per_bin = np.zeros(n_bins, dtype=np.float64)

    for row in subset.itertuples(index=False):
        lo  = float(row.low)
        hi  = float(row.high)
        vol = float(row.volume)
        if vol == 0.0:
            continue
        if hi <= lo:
            idx = int(np.searchsorted(bin_edges, lo, side="right")) - 1
            vol_per_bin[max(0, min(idx, n_bins - 1))] += vol
            continue
        span = hi - lo
        i_start = max(0, int(np.searchsorted(bin_edges, lo, side="right")) - 1)
        i_end   = min(n_bins, int(np.searchsorted(bin_edges, hi, side="left")))
        for b in range(i_start, i_end):
            overlap = min(bin_edges[b + 1], hi) - max(bin_edges[b], lo)
            if overlap > 0:
                vol_per_bin[b] += vol * overlap / span

    poc_idx = int(np.argmax(vol_per_bin))
    poc     = float(bin_centers[poc_idx])
    p70 = np.percentile(vol_per_bin, 70)
    p30 = np.percentile(vol_per_bin, 30)
    hvn = [float(c) for c, v in zip(bin_centers, vol_per_bin) if v >= p70]
    lvn = [float(c) for c, v in zip(bin_centers, vol_per_bin) if v <= p30]

    return {"poc": poc, "hvn": hvn, "lvn": lvn}


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def compute_bb_width(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Bollinger Band Width = (upper - lower) / middle.
    Computed manually from bbands to ensure fractional output (not %).
    """
    bb = df.ta.bbands(length=period)
    lower_col  = next(c for c in bb.columns if c.startswith("BBL"))
    middle_col = next(c for c in bb.columns if c.startswith("BBM"))
    upper_col  = next(c for c in bb.columns if c.startswith("BBU"))
    width = (bb[upper_col] - bb[lower_col]) / bb[middle_col].replace(0.0, np.nan)
    width.name = "bb_width"
    return width


def compute_bb_width_percentile(bb_width: pd.Series, lookback: int = 120) -> pd.Series:
    """
    Rolling percentile rank of current BB width vs previous `lookback` bars.
    Returns 0.0–1.0. Value < 0.10 → Bollinger Squeeze (bottom 10% of recent history).
    Default lookback = 120 4h bars ≈ 20 trading days.
    """
    def _pct_rank(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return np.nan
        valid = arr[:-1]
        valid = valid[~np.isnan(valid)]
        if len(valid) == 0:
            return np.nan
        return float((valid < arr[-1]).sum()) / len(valid)

    return bb_width.rolling(lookback + 1, min_periods=2).apply(_pct_rank, raw=True)


# ── Volatility-adjusted position sizing ──────────────────────────────────────

def compute_volatility_adjusted_size(
    base_pct: float,
    atr: float,
    ref_atr: float,
    bb_width_pct: float,
    min_size_pct: float = 1.0,
    bb_squeeze_threshold: float = 0.10,
) -> float:
    """
    Position size as % of capital, adjusted for volatility and Bollinger Squeeze.

    1. Squeeze guard: if bb_width_pct < threshold → return min_size_pct
       (Breakout imminent — cap to minimum to limit exposure before violent move)

    2. Normal (inverse volatility scaling):
       size = base_pct * (ref_atr / current_atr)
       ATR floor 1e-8 prevents ZeroDivisionError on ghost bars / exchange outages.
    """
    if pd.isna(bb_width_pct) or bb_width_pct < bb_squeeze_threshold:
        return float(min_size_pct)
    return base_pct * (max(ref_atr, 1e-8) / max(atr, 1e-8))
