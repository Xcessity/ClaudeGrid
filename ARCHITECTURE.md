# ClaudeGrid: Adaptive Neutral Grid Trading System

## Context
Building a complete grid trading research platform from scratch at `d:\Projects\Cryptobot\ClaudeGrid`. Implements a Neutral (Hedged) Grid strategy with adaptive ATR-based geometric spacing, market regime filtering (Hurst + ADX), VPVR/POC anchoring, pullback-confirmed trailing grid resets, and an infinite GA → Bayesian → WFO optimization loop storing validated strategies to a database.

**User decisions:** 20% holdout, NSGAIISampler multi-objective (GA dropped), both rolling + anchored WFO, dual-timeframe backtesting (4h signals + **1m** fills), maker fee for limit fills / taker only for forced market exits, Bollinger Squeeze size cap, holdout quarantine + Deflated Sharpe Ratio.

---

## Architecture

```
d:\Projects\Cryptobot\ClaudeGrid\
├── requirements.txt
├── config.py
├── main.py                          # Infinite search loop
├── data/
│   ├── __init__.py
│   ├── fetcher.py                   # ccxt Binance OHLCV, pagination, retries
│   ├── screener.py                  # Age filter, liquidity, correlation check
│   └── cache.py                     # Parquet cache with staleness checks
├── strategy/
│   ├── __init__.py
│   ├── indicators.py                # ATR, ADX+DMI, Hurst (DFA), VPVR
│   └── grid_engine.py               # State machine + bar-by-bar simulator
├── backtester/
│   ├── __init__.py
│   ├── engine.py                    # Simulation orchestration, fee/slippage
│   └── metrics.py                   # Sharpe, Sortino, MaxDD, PF, Calmar, CAGR
├── optimization/
│   ├── __init__.py
│   ├── parameter_space.py           # Bounds, types, encode/decode
│   ├── optimizer.py                 # Optuna NSGAIISampler — multi-objective Pareto search
│   ├── monte_carlo.py               # Permutation significance test
│   └── wfo.py                       # Rolling WFO + Anchored WFO
├── database/
│   ├── __init__.py
│   ├── models.py                    # SQLAlchemy: Symbol, Strategy, WFOWindow
│   └── storage.py                   # CRUD, dedup by param hash
├── visualization/
│   ├── __init__.py
│   └── equity_chart.py              # Plotly multi-panel strategy chart
└── view_strategy.py                 # CLI: python view_strategy.py --id 42
```

---

## Dependencies (`requirements.txt`)

```
ccxt>=4.3.0
pandas>=2.2.0
numpy>=1.26.0
pyarrow>=15.0
pandas-ta>=0.3.14b      # ATR, ADX (no TA-Lib binary required)
optuna>=3.6.0
sqlalchemy>=2.0.0
scipy>=1.13.0           # DFA / Variance Ratio for Hurst
plotly>=5.22.0          # Interactive multi-panel equity charts
kaleido>=0.2.1          # Plotly static PNG/HTML export
joblib>=1.4.0           # Parallel GA fitness evaluation
tqdm>=4.66.0
loguru>=0.7.0
numba>=0.59.0           # JIT-compile Monte Carlo inner loop (200x backtests)
```

---

## Data Split Strategy (Anti-Overfitting)

Every symbol's full history is split **once at startup** and never changed:

```
|--- 80% OPTIMIZATION SET ---|--- 20% HOLDOUT ---|-- last 60 days --|
     GA + Bayesian see this      WFO final OOS       Visualization
                                                      (unseen, always
                                                       most recent)
```

- GA and Bayesian only ever receive the 80% optimization set
- The 20% holdout is used as the **final out-of-sample WFO test windows**
- The **last 60 calendar days** of the full dataset are reserved exclusively for visualization — they are the most recent data and are never used in any optimization or validation step
- The 60-day visualization slice sits at the end of (and is carved out of) the holdout

**Three-way split in practice:**
- `split_date`  = 80th percentile date of full history (optimization/holdout boundary)
- `viz_date`    = `most_recent_date - 60 days` (holdout/visualization boundary)
- Optimization set: `[start, split_date)`
- Holdout set:      `[split_date, viz_date)`
- Visualization:    `[viz_date, end]`

---

## File-by-File Implementation Plan

### 1. `config.py`
```python
@dataclass
class Config:
    # Data
    min_years: int = 5
    min_volume_usdt: float = 10_000_000
    holdout_fraction: float = 0.20

    # Fees & Slippage — separated by order type
    maker_fee: float = 0.0002        # 0.02% — grid limit fills (resting orders, zero slippage)
    taker_fee: float = 0.0005        # 0.05% — forced market exits only (regime kill, acct DD stop)

    # Dynamic slippage for forced market exits — scales with current volatility
    # Base slippage during normal conditions (ATR ≈ ref_atr):
    market_slippage_base: float = 0.0003   # 3 bps baseline
    # Maximum slippage during extreme volatility (flash crash, ATR >> ref_atr):
    market_slippage_max:  float = 0.0030   # 30 bps ceiling
    # ATR multiple at which slippage reaches the maximum:
    market_slippage_atr_cap: float = 3.0   # if current_atr >= 3 × ref_atr → use max slippage
    # Linear interpolation between base and max:
    # slippage = base + (max - base) * min(current_atr / (ref_atr * atr_cap), 1.0)

    # Quality thresholds for DB storage
    min_sharpe: float = 1.0
    min_sortino: float = 1.2
    min_return_dd_ratio: float = 2.0
    min_profit_factor: float = 1.3
    min_cagr: float = 0.15
    min_oos_consistency: float = 0.60   # ≥60% of OOS windows profitable
    min_wfo_oos_is_ratio: float = 0.50  # OOS perf ≥ 50% of IS perf

    # WFO
    wfo_is_bars: int = 1000          # IS window in bars (timeframe-agnostic)
    wfo_oos_bars: int = 300          # OOS window in bars
    wfo_step_bars: int = 150         # Step size in bars
    wfo_min_windows: int = 4         # Minimum windows required

    # GA
    pop_size: int = 80
    n_islands: int = 4               # Island model sub-populations
    migration_interval: int = 5      # Gens between island migrations
    migration_frac: float = 0.10     # Top 10% migrate
    n_generations_per_cycle: int = 40
    elite_frac: float = 0.10
    base_mutation_rate: float = 0.15
    diversity_threshold: float = 0.15  # Trigger adaptive mutation below this

    # Bayesian
    n_trials_plateau: int = 200
    plateau_threshold_frac: float = 0.80  # Score ≥ 80% of best = "plateau"

    # Dual-timeframe
    signal_timeframe: str = "4h"    # Indicators, regime filter, grid placement
    fill_timeframe: str = "1m"      # Fill simulation — 1m for near-exact limit order execution
    # 4h bar = 240 × 1m bars. Storage: ~200MB/symbol parquet (~6GB for 30 symbols).
    # 1m is the practical minimum without tick data to avoid multi-level bar traversal errors.

    # Bollinger Squeeze position size cap
    bb_period: int = 20
    bb_squeeze_pct_threshold: float = 0.10   # Lock to min size if BB width < 10th percentile

    # Funding rates
    simulate_funding: bool = True       # Apply 8h funding payments on net position
    funding_interval_hours: int = 8

    # Monte Carlo significance test
    mc_n_shuffles: int = 200            # Number of synthetic price series
    mc_significance: float = 0.05       # Must beat 95% of shuffles

    # Delta neutrality
    max_delta_pct: float = 0.05         # Max net long/short imbalance as % of capital
    # If abs(net_long_value - net_short_value) / capital > max_delta_pct → pause heavy side

    # Capital-level kill switch
    max_account_dd_pct: float = 0.25    # Stop trading if drawdown from peak > 25%

    # Warm-up
    # First max(atr_period, adx_period, hurst_window) bars excluded from trading
    # Handled automatically in GridEngine using indicator NaN mask

    # Holdout quarantine (multiple-testing bias prevention)
    holdout_max_failures: int = 3    # Quarantine symbol after N holdout failures
    holdout_quarantine_days: int = 30

    # Optimization
    n_optuna_trials: int = 500       # NSGAIISampler trials per search cycle

    # Search
    min_trades_per_backtest: int = 30
    min_trades_per_wfo_window: int = 8
    n_workers: int = -1              # joblib: -1 = all CPUs
```

---

### 2. `data/fetcher.py`
```python
class BinanceFetcher:
    def __init__(self, use_futures=True)
    # ccxt.binanceusdm (perpetual futures) or ccxt.binance (spot)

    def fetch_ohlcv(symbol, timeframe, since, until) -> pd.DataFrame
    # Paginated fetch, handles rate limits, retries on 429/503
    # Returns: DatetimeIndex(UTC), cols: open, high, low, close, volume

    def fetch_both(symbol) -> tuple[pd.DataFrame, pd.DataFrame]
    # Fetches signal_timeframe (4h) and fill_timeframe (1m) in sequence
    # Returns: (df_4h, df_1m) — both with UTC DatetimeIndex
    # 1m data for 5 years ≈ 2,628,000 bars per symbol (~200MB parquet)
    # Binance 1m history fully available for all USDT-M perpetuals with 5yr+ age
    #
    # Ghost bar handling (applied immediately after raw fetch):
    # Binance occasionally skips 1m bars during low-activity periods or API
    # outages. These gaps create NaN indicators, missed fill checks, and
    # misaligned timestamps in the inner simulation loop.
    #
    # Fix: after fetching, reindex df_1m to a perfect continuous 1m DatetimeIndex:
    #   full_index = pd.date_range(df_1m.index[0], df_1m.index[-1], freq="1min")
    #   df_1m = df_1m.reindex(full_index)
    #   df_1m["close"]  = df_1m["close"].ffill()    # carry last known price
    #   df_1m["open"]   = df_1m["open"].ffill()
    #   df_1m["high"]   = df_1m["high"].ffill()
    #   df_1m["low"]    = df_1m["low"].ffill()
    #   df_1m["volume"] = df_1m["volume"].fillna(0) # zero volume = no real activity
    #
    # Effect on simulation: filled bars have OHLC = last real close (flat bar).
    # A flat bar cannot trigger any fills (high == low == close, no level crossed).
    # Volume = 0 signals no real activity — VPVR correctly ignores these bars.

    def get_all_usdt_symbols() -> list[str]
    # All active USDT perpetual symbols

    def fetch_funding_rates(symbol, since, until) -> pd.Series
    # Fetch historical 8h funding rates from Binance via ccxt.fetch_funding_rate_history
    # Returns: DatetimeIndex(UTC, freq=8h), values = funding rate as fraction (e.g. 0.0001)
    # Cached to data/cache/{symbol}_funding.parquet

    def get_earliest_timestamp(symbol, timeframe) -> datetime
    # Binary-search or first-candle fetch to determine history length
    # Age check uses signal_timeframe (4h) only — sufficient for screening
```

---

### 3. `data/screener.py`
```python
class SymbolScreener:
    def screen(symbols: list[str], timeframe: str) -> list[str]
    # 1. Age filter: earliest_timestamp → history_years >= config.min_years
    # 2. Liquidity filter: avg 30-day volume >= config.min_volume_usdt
    # 3. Correlation filter (when >1 symbol passes):
    #    - Compute pairwise Pearson correlation on daily returns (last 2 years)
    #    - Cluster correlated symbols (threshold=0.85)
    #    - Keep only 1 representative per cluster (highest volume)
    # Returns: list sorted by volume desc, max ~30 symbols
```

---

### 4. `data/cache.py`
```python
class DataCache:
    # data/cache/{symbol}_4h.parquet    ← signal data
    # data/cache/{symbol}_1m.parquet    ← fill simulation data (~200MB each)
    # data/cache/{symbol}_funding.parquet ← 8h funding rates

    def get_or_fetch(symbol, timeframe) -> pd.DataFrame
    # Returns full history for one timeframe. Appends only new bars if cache exists.

    def get_or_fetch_both(symbol) -> tuple[pd.DataFrame, pd.DataFrame]
    # Returns (df_4h, df_1m). Fetches whichever is missing or stale.
    # Both are trimmed to the same date range (common start/end).

    def get_split(symbol) -> tuple[DataPair, DataPair, DataPair]
    # Returns (opt_pair, holdout_pair, viz_pair) — all three slices
    # DataPair = namedtuple("DataPair", ["df_4h", "df_1m", "funding"])
    # funding: pd.Series of 8h funding rates aligned to the same date range
    # Boundaries: split_date (80/20) and viz_date (last 60 calendar days)
    # Both dates stored in cache metadata — never recomputed after first call

    def is_fresh(symbol, timeframe, max_age_hours=4) -> bool
```

---

### 5. `strategy/indicators.py`
```python
def compute_atr(df, period=14) -> pd.Series
# Standard Wilder ATR via pandas-ta

def compute_adx(df, period=14) -> pd.DataFrame
# Columns: ADX, DI_plus, DI_minus via pandas-ta

def compute_hurst_dfa(series: pd.Series, window: int) -> pd.Series
# Detrended Fluctuation Analysis — replaces R/S (10x faster, more accurate)
# Algorithm:
#   1. Cumulative sum of (series - mean) → integrated series Y
#   2. For each of ~10 log-spaced box sizes n:
#      - Divide Y into non-overlapping segments of length n
#      - Fit linear trend per segment, compute RMS of residuals → F(n)
#   3. H = slope of log(F(n)) vs log(n)
# Interpretation: H<0.45 mean-reverting, H>0.55 trending, 0.45–0.55 random

def compute_vpvr(df, lookback: int, n_bins: int = 50) -> dict
# Volume Profile using last `lookback` bars
# Volume distribution: each bar's volume split uniformly across [low, high]
# (NOT just at close — captures intrabar price-volume distribution)
# Returns: {poc: float, hvn: list[float], lvn: list[float]}
# poc  = price bin with highest accumulated volume
# hvn  = bins with volume > 70th percentile (high-volume nodes)
# lvn  = bins with volume < 30th percentile (low-volume nodes)

def compute_bb_width(df, period=20) -> pd.Series
# Bollinger Band Width = (upper - lower) / middle
# Measures volatility as a normalized fraction of price
# via pandas-ta: df.ta.bbands(length=period)

def compute_bb_width_percentile(bb_width: pd.Series, lookback=20*6) -> pd.Series
# Rolling percentile rank of current BB width vs last `lookback` bars
# Returns 0.0–1.0; value < 0.10 → squeeze (width in bottom 10% of recent history)
# lookback default = 20 * 6 = 120 4h bars ≈ 20 trading days

def compute_volatility_adjusted_size(base_pct, atr, ref_atr,
                                     bb_width_pct: float,
                                     min_size_pct: float = 1.0) -> float
# 1. If bb_width_pct < config.bb_squeeze_pct_threshold (0.10):
#    → return min_size_pct  (Bollinger Squeeze — cap at minimum, breakout imminent)
# 2. Else:
#    → return base_pct * (max(ref_atr, 1e-8) / max(atr, 1e-8))  (normal inverse volatility sizing)
# ATR floor prevents ZeroDivisionError during exchange outages and ghost-bar flat periods
# Prevents maximizing position size right before a violent squeeze breakout
```

---

### 6. `strategy/grid_engine.py`
Core strategy — **dual-loop** architecture: outer loop on 4h signal bars, inner loop on 1m fill bars.

**Separation of concerns:**
- **4h loop** → computes indicators, manages regime state, sets/resets grid levels
- **1m loop** → simulates actual limit order fills within each 4h period

**Why 1m is necessary:**
A 5m bar on crypto futures can still traverse multiple tight grid levels during a liquidity cascade, funding flush, or stop-hunt wick. The worst-case traversal heuristic on a 5m bar is still a guess about an unknown intra-bar path. At 1m, price moves ~0.03–0.08% per bar on liquid futures. Grid levels are typically 0.3–2% apart. Multi-level fills within a single 1m bar are extremely rare — fill simulation is near-exact. The worst-case traversal heuristic remains as a final edge-case safety net only.

**Dual-Loop Structure:**
```python
def run(df_4h, df_1m, funding_rates, params, initial_capital, ref_atr):
    # Pre-compute all indicators on 4h data (vectorized)
    atr_4h      = compute_atr(df_4h, params.atr_period)
    adx_4h      = compute_adx(df_4h, params.adx_period)
    hurst_4h    = compute_hurst_dfa(df_4h["close"], params.hurst_window)
    bb_width_4h = compute_bb_width(df_4h, period=config.bb_period)
    bb_pct_4h   = compute_bb_width_percentile(bb_width_4h)
    # warmup_bars = index of first bar where all indicators are non-NaN
    warmup_bars = max(params.atr_period, params.adx_period, params.hurst_window, config.bb_period)
    # VPVR computed on-demand per 4h bar (uses rolling 4h window)

    state = GRID_MODE
    grid_levels = None
    extreme_price = None
    positions = []
    equity_1m = []
    peak_equity = initial_capital
    capital = initial_capital

    # ── OUTER LOOP: 4h signal bars ─────────────────────────────────────
    for i, bar_4h in enumerate(df_4h.itertuples()):

        # 0. Skip warm-up period — indicators not yet reliable
        if i < warmup_bars:
            period_1m = df_1m.loc[bar_4h.Index : bar_4h.Index + pd.Timedelta("4h")]
            for bar_1m in period_1m.itertuples():
                equity_1m.append(capital)
            continue

        # 1. Update regime using current 4h indicators
        h = hurst_4h.iloc[i]
        adx_val = adx_4h["ADX"].iloc[i]
        current_atr = atr_4h.iloc[i]
        regime_trending = (h > params.hurst_threshold or adx_val > params.adx_threshold)

        if regime_trending and state == GRID_MODE:
            # Regime kill: forced market exit of all open positions
            atr_ratio    = max(current_atr, 1e-8) / max(ref_atr, 1e-8)
            dynamic_slip = config.market_slippage_base + (
                (config.market_slippage_max - config.market_slippage_base)
                * min(atr_ratio / config.market_slippage_atr_cap, 1.0)
            )
            close_all_positions(positions, bar_4h.close,
                                slippage=dynamic_slip, fee=config.taker_fee)
            state = TRACKING_MODE
            extreme_price = bar_4h.close
            grid_levels = None

        # 2. (Re)build grid — LAZY: only when transitioning into GRID_MODE
        # VPVR is NOT recomputed on every bar in GRID_MODE. It is computed once,
        # exactly when a new grid needs to be formed (state just became GRID_MODE
        # with no active levels). This avoids O(vpvr_window) work at every 4h bar.
        if state == GRID_MODE and grid_levels is None:
            vpvr = compute_vpvr(df_4h.iloc[max(0, i - params.vpvr_window) : i])
            center = nearest_hvn(bar_4h.close, vpvr) if params.use_vpvr_anchor \
                     else bar_4h.close
            # LVN avoidance: snap each geometric level to nearest HVN
            grid_levels = build_grid_hvn_snapped(center, current_atr, vpvr, params)
            adjusted_size = compute_volatility_adjusted_size(
                params.position_size_pct, current_atr, ref_atr,
                bb_width_pct=bb_pct_4h.iloc[i],  # Bollinger Squeeze guard
                min_size_pct=1.0)

        # ── INNER LOOP: 1m fill bars within this 4h period ─────────────
        period_1m = df_1m.loc[bar_4h.Index : bar_4h.Index + pd.Timedelta("4h")]

        for bar_1m in period_1m.itertuples():

            # Account-level max drawdown kill switch
            current_equity = mark_to_market(positions, bar_1m.close)
            peak_equity = max(peak_equity, current_equity)
            if (peak_equity - current_equity) / peak_equity > config.max_account_dd_pct:
                # Dynamic slippage: scales linearly from base → max based on ATR ratio
                atr_ratio = max(current_atr, 1e-8) / max(ref_atr, 1e-8)  # floor prevents ZeroDivisionError during exchange outages / ghost-bar flat periods
                dynamic_slip = config.market_slippage_base + (
                    (config.market_slippage_max - config.market_slippage_base)
                    * min(atr_ratio / config.market_slippage_atr_cap, 1.0)
                )
                # Example: normal ATR → 3bps; flash crash (3× ATR) → 30bps
                close_all_positions(positions, bar_1m.open,
                                    slippage=dynamic_slip, fee=config.taker_fee)
                state = KILLED
                equity_1m.append(current_equity)
                break  # exit inner loop

            if state == GRID_MODE and grid_levels is not None:
                fills = simulate_fills_1m(bar_1m, grid_levels, params)
                for fill in fills:
                    # Delta neutrality check before accepting fill
                    net_delta = compute_net_delta(positions)
                    if fill.side == "buy"  and net_delta >  config.max_delta_pct * capital: continue
                    if fill.side == "sell" and net_delta < -config.max_delta_pct * capital: continue
                    process_fill(fill, positions, capital, adjusted_size)

                if bar_1m.close < grid_levels.lower_bound or \
                   bar_1m.close > grid_levels.upper_bound:
                    state = TRACKING_MODE
                    extreme_price = bar_1m.close
                    grid_levels = None

            elif state == TRACKING_MODE:
                if breakout_direction == "up":
                    extreme_price = max(extreme_price, bar_1m.high)
                else:
                    extreme_price = min(extreme_price, bar_1m.low)

                retraction = abs(bar_1m.close - extreme_price) / extreme_price
                regime_cleared = (h <= params.hurst_threshold and
                                  adx_val <= params.adx_threshold)
                if retraction >= params.pullback_pct / 100 and regime_cleared:
                    state = GRID_MODE

            # Funding payment: apply at every 8h timestamp
            if bar_1m.Index in funding_rates.index:
                rate = funding_rates.loc[bar_1m.Index]
                net_pos_value = compute_net_position_value(positions, bar_1m.close)
                funding_payment = net_pos_value * rate   # positive = longs pay shorts
                capital -= funding_payment

            equity_1m.append(mark_to_market(positions, bar_1m.close) + capital)

        if state == KILLED:
            break  # exit outer loop too

    return SimResult(
        equity_curve=pd.Series(equity_1m, index=df_1m.index[:len(equity_1m)]),
        trades=closed_trades,
        n_grid_resets=...,
        n_regime_pauses=...,
        killed_early=state == KILLED,
    )
```

**simulate_fills_1m() — fill logic within one 1m bar:**
```python
# Worst-case traversal applied to 1m bar (safety net — rarely triggers at 1m).
# A 1m bar spans ~0.03–0.08% on liquid futures → almost never spans 2 grid levels.
# Traversal: bearish bar → open → low → high → close
#            bullish bar → open → high → low → close
# Buy fills:  bar_1m.low  <= buy_level  → fill at buy_level   (NO slippage — limit order)
# Sell fills: bar_1m.high >= sell_level → fill at sell_level  (NO slippage — limit order)
# Fee:        notional * maker_fee  (0.02% — resting limit orders are maker fills)
#
# EXCEPTION — forced market exits (regime kill, account DD stop):
#   dynamic_slip = base + (max - base) * min(current_atr / (ref_atr * atr_cap), 1.0)
#   fill_price   = market_price * (1 ± dynamic_slip)
#   fee          = notional * taker_fee  (0.05%)
#   Examples:
#     Normal ATR (ratio=1.0): slippage = 3 bps
#     2× ATR (ratio=2.0):     slippage = ~17 bps
#     3× ATR flash crash:     slippage = 30 bps (ceiling)
```

**Grid levels structure:**
```python
@dataclass
class GridLevels:
    center: float
    buys:  list[float]   # n_levels prices below center
    sells: list[float]   # n_levels prices above center
    lower_bound: float   # outermost buy level (breakout trigger)
    upper_bound: float   # outermost sell level (breakout trigger)

# build_grid_hvn_snapped():
#   1. Compute raw geometric levels: distance[i] = atr * atr_multiplier * geometric_ratio^i
#   2. For each raw level: if it falls within an LVN zone (VPVR bin < 30th pct volume):
#      → snap to nearest HVN (within 0.5 * atr_multiplier * atr tolerance)
#      → if no HVN nearby, keep the raw level (rare edge case)
#   3. Result: levels cluster at high-liquidity nodes → higher fill frequency,
#      less chance of price blowing straight through a level without filling
```

**P&L Model (per 1m fill):**
```python
# Normal grid fill — resting limit order (maker):
entry_fill = limit_price               # no slippage on limit fills
entry_fee  = entry_fill * qty * maker_fee   # 0.02%

exit_fill  = limit_price               # no slippage on limit fills
exit_fee   = exit_fill  * qty * maker_fee   # 0.02%

pnl = (exit_fill - entry_fill) * qty - entry_fee - exit_fee
# Round-trip drag: 0.04% (vs 0.16% with old taker+slippage model)

# Forced market exit — regime kill switch or account DD stop (taker):
exit_fill  = market_price * (1 - market_slippage_pct)  # 0.03% slippage
exit_fee   = exit_fill * qty * taker_fee                # 0.05%
# Mirror for shorts
```

```python
@dataclass
class GridParams:
    atr_period: int           # 10–50
    atr_multiplier: float     # 0.3–2.0
    geometric_ratio: float    # 1.0–2.0
    n_levels: int             # 2–8
    pullback_pct: float       # 0.5–5.0
    hurst_window: int         # 50–200  (in 4h bars)
    hurst_threshold: float    # 0.45–0.65
    adx_period: int           # 10–30   (in 4h bars)
    adx_threshold: float      # 20–40
    vpvr_window: int          # 50–300  (in 4h bars)
    use_vpvr_anchor: bool
    position_size_pct: float  # 1.0–8.0 (base, before vol adjustment)
    max_open_levels: int      # 2–8

class GridEngine:
    def run(df_4h: pd.DataFrame, df_1m: pd.DataFrame,
            funding_rates: pd.Series,
            params: GridParams, initial_capital: float,
            ref_atr: float) -> SimResult
    # df_4h:          signal bars (indicators, regime, grid placement)
    # df_1m:          fill bars (limit order execution, equity curve)
    # funding_rates:  8h funding rate series for net position payments
    # ref_atr:        median ATR over optimization period (consistent sizing)
    # equity_curve output: 1m resolution, includes funding drag and fees
```

---

### 7. `backtester/engine.py`
```python
class Backtester:
    def run(params: GridParams,
            data: DataPair,            # DataPair(df_4h, df_1m) — pre-sliced
            initial_capital=10_000,
            ref_atr: float = None) -> BacktestResult
    # data is already sliced to IS or OOS date range before passing in
    # ref_atr always comes from the full optimization set (never re-computed per window)
    # Delegates to GridEngine.run(df_4h, df_1m, params, ...)

class BacktestResult:
    equity_curve: pd.Series   # 1m-resolution equity (unrealized + realized)
    trades: list[Trade]        # realized round-trips only
    params: GridParams
    n_grid_resets: int
    n_regime_pauses: int

# Helper: slice both timeframes to a date range together
def slice_data_pair(data: DataPair, start: datetime, end: datetime) -> DataPair:
    return DataPair(
        df_4h=data.df_4h.loc[start:end],
        df_1m=data.df_1m.loc[start:end],
    )
```

---

### 8. `backtester/metrics.py`
```python
def compute_metrics(equity_curve: pd.Series, trades: list) -> dict:
    # equity_curve is always at 1m resolution, includes funding payments
    # annualization_factor = 365 * 24 * 60  (1m bars per year = 525,600)
    return {
        "sharpe":               annualized sharpe (risk-free=0),
        "sortino":              annualized sortino (downside deviation only),
        "max_drawdown":         max peak-to-trough fraction,
        "calmar":               cagr / max_drawdown,
        "return_dd_ratio":      total_return / max_drawdown,
        "profit_factor":        gross_profit / gross_loss,
        "cagr":                 compound annual growth rate,
        "win_rate":             fraction of profitable trades,
        "avg_trade_pct":        avg trade return as % of capital,
        "n_trades":             total round-trips,
        "avg_hold_bars":        avg 1m bars per trade,
        "max_consec_losses":    longest losing trade streak,
        "dd_recovery_bars":     avg 1m bars to recover from each drawdown trough,
        "killed_early":         True if account DD kill switch triggered,
    }

def compute_deflated_sharpe(sharpe: float, n_trials: int,
                             n_bars: int, skewness: float,
                             kurtosis: float) -> float:
    """
    Deflated Sharpe Ratio (López de Prado, 2018).
    Penalizes Sharpe based on how many parameter combinations were tested
    before this strategy was found. Prevents lucky survivors from being
    mistaken for genuine edge.

    DSR = Sharpe * Phi_inverse(1 - (1 - Phi(sharpe_star)) / n_trials)
    where sharpe_star accounts for non-normality via skewness/kurtosis correction.

    Returns deflated Sharpe. A strategy passes if DSR >= config.min_sharpe.
    """
```

---

### 9. `optimization/parameter_space.py`
```python
PARAM_SPACE = {
    "atr_period":         ("int",   10,  50),
    "atr_multiplier":     ("float", 0.3, 2.0),
    "geometric_ratio":    ("float", 1.0, 2.0),
    "n_levels":           ("int",   2,   8),
    "pullback_pct":       ("float", 0.5, 5.0),
    "hurst_window":       ("int",   50,  200),
    "hurst_threshold":    ("float", 0.45, 0.65),
    "adx_period":         ("int",   10,  30),
    "adx_threshold":      ("float", 20,  40),
    "vpvr_window":        ("int",   50,  300),
    "use_vpvr_anchor":    ("bool",  False, True),
    "position_size_pct":  ("float", 1.0, 8.0),
    "max_open_levels":    ("int",   2,   8),
}

def random_params() -> GridParams
def params_to_vector(p: GridParams) -> np.ndarray   # normalized 0–1
def vector_to_params(v: np.ndarray) -> GridParams
    # Enforces hard constraints after decoding:
    #   max_open_levels = min(max_open_levels, n_levels)  ← constraint fix
def params_hash(p: GridParams) -> str               # SHA256 of rounded params
```

---

### 10. `optimization/optimizer.py`
Single optimization stage using **Optuna NSGAIISampler** (replaces the GA + Bayesian two-stage pipeline). NSGAIISampler is a built-in Optuna evolutionary sampler implementing Non-dominated Sorting Genetic Algorithm II — it performs multi-objective Pareto optimization natively without a separate GA codebase.

**Why NSGAIISampler over TPE + GA:**
- Eliminates the redundant GA → seed-Optuna pipeline entirely
- Multi-objective: simultaneously maximizes Sharpe AND Sortino while minimizing MaxDD — returns a Pareto front of non-dominated solutions
- TPE is a single-objective surrogate model; NSGAIISampler naturally handles conflicting objectives without manually weighting them
- 500 trials explores the space as thoroughly as the old GA (80 pop × 40 gens) + Bayesian (200 trials) combined, in one unified pass

```python
class MultiObjectiveOptimizer:
    def __init__(self, eval_fn, n_trials=500, n_workers=-1)
    # eval_fn: GridParams → BacktestResult (runs full backtest on opt_data)

    def run() -> optuna.Study
    # study = optuna.create_study(
    #     directions=["maximize", "maximize", "minimize"],  # sharpe, sortino, max_dd
    #     sampler=NSGAIISampler(population_size=50, seed=42)
    # )
    # Parallel trial evaluation via joblib (n_jobs=n_workers)
    # Each trial: sample params → decode via vector_to_params (applies constraints)
    #             → run backtest → return (sharpe, sortino, max_dd)

    def get_pareto_front(study: optuna.Study) -> list[GridParams]
    # Returns all Pareto-optimal trials (non-dominated solutions)
    # Filtered to: sharpe >= 1.0 AND sortino >= 1.2 AND max_dd <= 0.30

    def score_plateau_width(study: optuna.Study) -> dict[str, float]
    # For each parameter dimension, compute normalized plateau width
    # Uses Pareto-front trials only: range where ≥80% of Pareto solutions reside
    # Width score 0–1; higher = more robust to parameter perturbation

    def get_stable_params(study: optuna.Study,
                          min_plateau_width=0.15) -> list[GridParams]
    # Filter Pareto front to candidates with avg plateau width ≥ threshold
    # Ranked by: sharpe * (1 - max_dd) * plateau_width_score
    # Returns top-5 for WFO validation
```

**Parallel trial evaluation — memory-safe NumPy array passing:**
```python
# Problem: passing df_1m (~200MB DataFrame) to each loky worker via pickle
# causes 200MB × n_workers memory bloat and potential OOM crashes.
#
# Fix: convert df_1m to contiguous NumPy arrays ONCE before the study starts.
# Pass arrays to workers instead of the DataFrame — loky can share read-only
# numpy arrays via memory-mapped files without copying.

@dataclass
class NumpyOHLCV:
    """Compact, memory-mappable representation of OHLCV + pre-computed 4h indicators."""
    open:   np.ndarray   # float64, C-contiguous
    high:   np.ndarray
    low:    np.ndarray
    close:  np.ndarray
    volume: np.ndarray
    timestamps: np.ndarray  # int64 UTC nanoseconds (from DatetimeIndex)
    # Pre-computed indicator lookup tables (only populated for np_4h, None on np_1m).
    # Keyed by period/window value so workers do index lookups instead of calling pandas-ta.
    # Avoids O(500 trials × n_4h_bars) redundant pandas-ta calls inside workers.
    # atr_arrays[period]   → float64 array of ATR values for that period (10–50)
    # adx_arrays[period]   → float64 array of ADX values for that period (10–30)
    # hurst_arrays[window] → float64 array of Hurst exponents for that window (50–200)
    # bb_pct               → float64 array of BB width percentile (fixed config.bb_period)
    atr_arrays:   dict[int, np.ndarray] | None = None
    adx_arrays:   dict[int, np.ndarray] | None = None
    hurst_arrays: dict[int, np.ndarray] | None = None
    bb_pct:       np.ndarray | None = None

def prepare_numpy_arrays(df: pd.DataFrame) -> NumpyOHLCV:
    return NumpyOHLCV(
        open=np.ascontiguousarray(df["open"].values,   dtype=np.float64),
        high=np.ascontiguousarray(df["high"].values,   dtype=np.float64),
        low =np.ascontiguousarray(df["low"].values,    dtype=np.float64),
        close=np.ascontiguousarray(df["close"].values, dtype=np.float64),
        volume=np.ascontiguousarray(df["volume"].values, dtype=np.float64),
        timestamps=df.index.view(np.int64),  # no copy — view of existing array
    )

def prepare_4h_arrays(df_4h: pd.DataFrame) -> NumpyOHLCV:
    """Prepare np_4h with all indicator lookups pre-computed for every value in PARAM_SPACE."""
    base = prepare_numpy_arrays(df_4h)
    atr_arrays   = {p: compute_atr(df_4h, p).values
                    for p in range(10, 51)}          # all ATR periods in param space
    adx_arrays   = {p: compute_adx(df_4h, p)["ADX"].values
                    for p in range(10, 31)}          # all ADX periods in param space
    hurst_arrays = {w: compute_hurst_dfa(df_4h["close"], w).values
                    for w in range(50, 201)}         # all Hurst windows in param space
    bb_pct       = compute_bb_width_percentile(
                       compute_bb_width(df_4h, config.bb_period)).values
    return NumpyOHLCV(**vars(base),
                      atr_arrays=atr_arrays, adx_arrays=adx_arrays,
                      hurst_arrays=hurst_arrays, bb_pct=bb_pct)

class MultiObjectiveOptimizer:
    def __init__(self, eval_fn, opt_data: DataPair, n_trials=500, n_workers=-1):
        # Pre-convert ONCE at init, before any trial starts.
        # np_4h uses prepare_4h_arrays — includes pre-computed indicator lookups for
        # every parameter value in PARAM_SPACE (ATR 10–50, ADX 10–30, Hurst 50–200).
        # Workers do index lookups (np_4h.atr_arrays[params.atr_period][i]) instead of
        # calling pandas-ta, eliminating O(500 × n_4h_bars) redundant computation.
        self.np_4h   = prepare_4h_arrays(opt_data.df_4h)   # ~2s one-time cost
        self.np_1m   = prepare_numpy_arrays(opt_data.df_1m)
        self.funding = opt_data.funding.values  # numpy array, no pandas overhead

    def run(self) -> optuna.Study:
        # Joblib backend: "loky" (default on all platforms, supports memory mapping)
        # The NumpyOHLCV arrays are passed by reference to each worker.
        # On Linux (fork), workers share physical memory pages (zero copy).
        # On Windows (loky spawn), arrays are written to a temp mmap file once
        # and all workers map the same file — no per-worker copy.
        with parallel_backend("loky"):
            study.optimize(
                self._objective,       # receives NumpyOHLCV via closure
                n_trials=self.n_trials,
                n_jobs=self.n_workers,
            )

    def _objective(self, trial: optuna.Trial) -> tuple[float, float, float]:
        params = sample_params(trial)           # decode from trial suggestions
        # Worker receives self.np_1m (shared mmap) — no 200MB serialization
        result = run_backtest_numpy(params, self.np_1m, self.np_4h, self.funding)
        m = compute_metrics_numpy(result)
        if m["n_trades"] < config.min_trades_per_backtest or m["killed_early"]:
            return (-1.0, -1.0, 1.0)
        return (m["sharpe"], m["sortino"], m["max_drawdown"])
```

**Corresponding change in `grid_engine.py`:**
`GridEngine.run()` gains a `run_numpy()` variant that accepts `NumpyOHLCV` instead of DataFrames. The 4h outer loop uses `np_4h.timestamps` for alignment with `np_1m.timestamps` via binary search (`np.searchsorted`). This avoids any pandas `.loc[]` overhead inside the inner loop.

---

### 12. `optimization/wfo.py`
Both rolling and anchored WFO. Windows sized in **4h signal bars** (timeframe-agnostic for indicators). Each WFO window simulation automatically uses both the 4h and 1m slices for the corresponding date range.

```python
class WalkForwardOptimizer:
    def __init__(self, symbol,
                 opt_data: DataPair,      # 80% optimization set (df_4h + df_1m)
                 holdout_data: DataPair,  # 20% holdout set (df_4h + df_1m)
                 ref_atr: float)

    def run_rolling(params: GridParams) -> RollingWFOResult
    # Window boundaries defined by 4h bar indices (is_bars=1000, oos_bars=300)
    # For each window:
    #   window_data = slice_data_pair(opt_data, window_start, window_end)
    #   result = Backtester().run(params, window_data, ref_atr=ref_atr)
    # 1m bars for the same date range are sliced automatically via DataPair
    #
    # Compounding OOS equity: chain OOS windows sequentially
    #   window[0] OOS ends with equity E0 → window[1] OOS starts with E0 (not fresh $10k)
    #   Final compounded OOS equity curve = true out-of-sample performance over full period
    #   Both per-window metrics AND compounded total are stored

    def run_anchored(params: GridParams) -> AnchoredWFOResult
    # IS window expands: window[i] covers 4h bars [0 : is_bars + i*step_bars]
    # OOS = next oos_bars 4h bars (but fills simulated at 1m resolution)

    def run_holdout(params: GridParams) -> BacktestResult
    # Final clean validation on holdout_data (both df_4h and df_1m slices)
    # Never called until both rolling and anchored WFO pass

    def validate(params: GridParams) -> WFOResult
    # 1. run_rolling  → check vs thresholds
    # 2. run_anchored → check vs thresholds
    # 3. If both pass: run_holdout
    # is_valid = all three pass

@dataclass
class WFOResult:
    params: GridParams
    symbol: str
    rolling_windows: list[WindowResult]
    anchored_windows: list[WindowResult]
    holdout_metrics: dict | None
    avg_oos_sharpe: float
    avg_oos_sortino: float
    oos_is_ratio: float
    consistency_score: float           # fraction of OOS windows positive
    plateau_width_score: float
    is_valid: bool

# Per-window minimum guard:
# Discard window (mark invalid) if n_trades < config.min_trades_per_wfo_window
# n_trades measured from the 1m-resolution simulation
```

---

### 13. `optimization/monte_carlo.py`
Statistical significance test. Run **after** WFO passes, **before** DB storage.

**Algorithm:**
```python
def monte_carlo_significance(params: GridParams, data: DataPair,
                              ref_atr: float,
                              n_shuffles: int = 200) -> float:
    """
    Returns p-value: fraction of shuffled runs that beat or match the real result.
    Pass threshold: p_value <= config.mc_significance (0.05)
    """
    # 1. Run real backtest on optimization set
    real_result = Backtester().run(params, data, ref_atr=ref_atr)
    real_score  = composite_score(compute_metrics(real_result.equity_curve, real_result.trades))

    # 2. Generate n_shuffles synthetic price series by shuffling RETURNS
    #    (not raw prices) — preserves volatility distribution but destroys autocorrelation
    log_returns = np.diff(np.log(data.df_1m["close"].values))
    beat_count = 0

    for _ in range(n_shuffles):
        shuffled_returns = np.random.permutation(log_returns)
        shuffled_prices  = np.exp(np.concatenate([[np.log(data.df_1m["close"].iloc[0])],
                                                   shuffled_returns.cumsum()]))
        # Rebuild df_1m with shuffled close prices (keep OHLV structure consistent)
        synthetic_1m = rebuild_ohlcv_from_close(data.df_1m, shuffled_prices)
        synthetic_data = DataPair(data.df_4h, synthetic_1m, data.funding)

        synth_result = Backtester().run(params, synthetic_data, ref_atr=ref_atr)
        synth_score  = composite_score(compute_metrics(synth_result.equity_curve,
                                                       synth_result.trades))
        if synth_score >= real_score:
            beat_count += 1

    p_value = beat_count / n_shuffles
    return p_value   # pass if p_value <= 0.05
```

**Performance note:** 200 backtests × 1m resolution is expensive. Mitigation:
- Use `numba` JIT-compiled inner simulation loop for the shuffled runs
- Run shuffled backtests in parallel via `joblib` (same worker pool as GA)
- Shuffled runs use a lighter version of the engine (no state logging, no equity curve storage — just final score)

**Integration in main.py:**
```python
# After wfo.validate() passes:
p_value = monte_carlo_significance(params, opt_data, ref_atr)
if p_value > config.mc_significance:
    log.info(f"Strategy rejected: Monte Carlo p={p_value:.3f} (not significant)")
    continue   # not a real edge
```

---

### 14. `database/models.py`
```python
class Symbol(Base):
    __tablename__ = "symbols"
    id, symbol, timeframe, first_date, last_date, avg_volume_usdt

class Strategy(Base):
    __tablename__ = "strategies"
    id, symbol_id (FK), timeframe
    param_hash (UNIQUE)      # SHA256 of rounded params — prevents duplicates
    # All 13 GridParams columns
    # Full metrics: sharpe, sortino, max_drawdown, calmar, return_dd_ratio,
    #               profit_factor, cagr, win_rate, n_trades
    # WFO: avg_oos_sharpe, avg_oos_sortino, oos_is_ratio, consistency_score,
    #      plateau_width_score, compounded_oos_return, compounded_oos_max_dd
    # Monte Carlo: mc_p_value, mc_n_shuffles
    # Extra metrics: max_consec_losses, dd_recovery_bars
    # Holdout: holdout_sharpe, holdout_cagr, holdout_max_drawdown
    # Meta: created_at, ga_generation, search_cycle

class WFOWindow(Base):
    __tablename__ = "wfo_windows"
    id, strategy_id (FK), wfo_type ("rolling" | "anchored")
    window_index, is_start, is_end, oos_start, oos_end
    is_sharpe, is_sortino, is_return
    oos_sharpe, oos_sortino, oos_return, n_trades
```

---

### 14. `database/storage.py`
```python
class StrategyDB:
    def __init__(self, db_path="strategies.db")
    def save_strategy(wfo_result: WFOResult) -> int | None
    # Checks param_hash uniqueness before insert; returns None if duplicate
    def get_top_strategies(n=20, sort_by="avg_oos_sharpe") -> list[Strategy]
    def get_strategies_by_symbol(symbol) -> list[Strategy]
    def strategy_count() -> int
    def get_all_params_hashes() -> set[str]  # for fast dedup check
```

---

### 15. `main.py` — Infinite Search Loop

```python
def main():
    # ── Startup ──────────────────────────────────────────────────────
    db = StrategyDB()
    fetcher = BinanceFetcher()
    cache = DataCache()
    
    all_symbols = fetcher.get_all_usdt_symbols()
    screener = SymbolScreener()
    qualified = screener.screen(all_symbols, config.signal_timeframe)

    # Pre-fetch and cache 4h + 1m + funding for all qualified symbols
    for sym in tqdm(qualified, desc="Caching 4h + 1m + funding"):
        cache.get_or_fetch_both(sym)

    existing_hashes = db.get_all_params_hashes()
    # Per-symbol holdout failure tracking (multiple-testing bias prevention)
    holdout_failures:   dict[str, int]      = {}
    quarantine_until:   dict[str, datetime] = {}
    search_cycle = 0
    symbol_idx   = 0
    total_trials_per_symbol: dict[str, int] = {}  # for Deflated Sharpe

    # ── Infinite Loop ─────────────────────────────────────────────────
    while True:
        symbol = qualified[symbol_idx % len(qualified)]
        symbol_idx += 1

        # Holdout quarantine check
        if symbol in quarantine_until and datetime.utcnow() < quarantine_until[symbol]:
            log.warning(f"Skipping {symbol} — quarantined until {quarantine_until[symbol].date()}")
            continue

        search_cycle += 1

        # Load 4h + 1m + funding, split into opt / holdout / viz
        opt_data, holdout_data, viz_data = cache.get_split(symbol)
        # opt_data.df_4h  → 4h signal bars (80%)
        # opt_data.df_1m  → 1m fill bars for same date range
        # opt_data.funding → 8h funding rates
        # holdout_data.*  → same structure, covers [split_date, viz_date)
        # viz_data.*      → last 60 calendar days, never touched here

        # ref_atr computed on 4h optimization bars (used for consistent sizing)
        ref_atr = compute_atr(opt_data.df_4h, period=14).median()

        # ── Phase 1: NSGAIISampler multi-objective search ─────────────
        # MultiObjectiveOptimizer.__init__ calls prepare_4h_arrays() once here,
        # pre-computing ATR/ADX/Hurst/BB for all PARAM_SPACE values before any
        # trial starts. Workers receive shared mmap arrays with zero pandas overhead.
        optimizer = MultiObjectiveOptimizer(
            eval_fn=None,           # objective defined inside optimizer via closure
            opt_data=opt_data,
            n_trials=config.n_optuna_trials,
            n_workers=config.n_workers,
        )
        study = optimizer.run()
        total_trials_per_symbol[symbol] = \
            total_trials_per_symbol.get(symbol, 0) + config.n_optuna_trials
        stable = optimizer.get_stable_params(study)  # top-5 Pareto + plateau

        # ── Phase 2: WFO validation ────────────────────────────────────
        wfo = WalkForwardOptimizer(symbol, opt_data, holdout_data, ref_atr)

        for params in stable:
            h = params_hash(params)
            if h in existing_hashes:
                continue

            # Monte Carlo significance test
            p_value = monte_carlo_significance(params, opt_data, ref_atr)
            if p_value > config.mc_significance:
                log.debug(f"MC rejected: p={p_value:.3f} on {symbol}")
                continue

            result = wfo.validate(params)

            if not result.is_valid:
                # Increment holdout failure counter
                holdout_failures[symbol] = holdout_failures.get(symbol, 0) + 1
                if holdout_failures[symbol] >= config.holdout_max_failures:
                    quarantine_until[symbol] = (datetime.utcnow()
                                               + timedelta(days=config.holdout_quarantine_days))
                    log.warning(f"{symbol} quarantined for {config.holdout_quarantine_days}d "
                                f"after {holdout_failures[symbol]} holdout failures")
                continue

            # Deflated Sharpe Ratio — penalize for number of trials tested
            m = result.holdout_metrics
            dsr = compute_deflated_sharpe(
                sharpe=result.avg_oos_sharpe,
                n_trials=total_trials_per_symbol[symbol],
                n_bars=len(opt_data.df_1m),
                skewness=m.get("return_skewness", 0),
                kurtosis=m.get("return_kurtosis", 3),
            )
            if dsr < config.min_sharpe:
                log.debug(f"DSR rejected: DSR={dsr:.2f} (raw={result.avg_oos_sharpe:.2f}) "
                          f"after {total_trials_per_symbol[symbol]} trials on {symbol}")
                continue

            # All gates passed — save strategy
            db.save_strategy(result, dsr=dsr, mc_p_value=p_value)
            existing_hashes.add(h)
            holdout_failures[symbol] = 0  # reset failure counter on success
            log.success(
                f"[Cycle {search_cycle}] Strategy #{db.strategy_count()} saved | "
                f"{symbol} | OOS Sharpe={result.avg_oos_sharpe:.2f} | "
                f"DSR={dsr:.2f} | MC p={p_value:.3f} | "
                f"Holdout CAGR={result.holdout_metrics['cagr']:.1%} | "
                f"Consistency={result.consistency_score:.0%}"
            )

        log.info(f"Cycle {search_cycle} done | {symbol} | "
                f"DB total: {db.strategy_count()}")
```

---

---

### 16. `visualization/equity_chart.py`

Generates a 3-panel interactive Plotly chart for any stored strategy, run on the **last 60 calendar days** of that symbol's data (never seen during optimization or WFO).

**Virtual portfolio:** `$1,000 USDT` starting capital, same `GridParams` from the stored strategy, same `ref_atr` from the optimization period.

**Chart layout — 3 stacked panels:**

```
┌──────────────────────────────────────────────────────────┐
│ PANEL 1 — Price + Grid (candlestick, 1m bars)            │
│  • Candlestick price chart at 1m resolution              │
│  • Grid level lines (dashed) — redrawn at each reset     │
│    - Buy levels: green dashed                            │
│    - Sell levels: red dashed                             │
│  • Grid center / POC anchor: blue dotted line            │
│  • Shaded regions: orange = TRACKING_MODE               │
│                    grey   = regime paused (Hurst/ADX)    │
│  • Trade markers: ▲ green = long entry/exit fill         │
│                   ▼ red   = short entry/exit fill        │
│  • Grid reset events: vertical purple lines              │
├──────────────────────────────────────────────────────────┤
│ PANEL 2 — Equity Curve ($1,000 virtual portfolio)        │
│  • Line: portfolio value over 60 days (1m resolution)    │
│  • Horizontal dashed line at $1,000 (starting capital)   │
│  • Filled area: green above $1,000, red below            │
│  • Annotation box (top-right):                           │
│    Final: $X,XXX (+XX.X%) | MaxDD: -X.X%                │
│    Sharpe: X.XX | Trades: XX | Win Rate: XX%             │
├──────────────────────────────────────────────────────────┤
│ PANEL 3 — Drawdown from Equity Peak                      │
│  • Filled area chart, always negative, red               │
│  • Max drawdown annotated with a dashed horizontal line  │
└──────────────────────────────────────────────────────────┘
 ← 60-day date range (x-axis shared across all 3 panels) →
```

**Saved as:**
- `charts/{symbol}_{strategy_id}.html` — interactive (zoomable, hoverable)
- `charts/{symbol}_{strategy_id}.png`  — static image for quick preview

```python
def generate_chart(strategy: Strategy,
                   viz_data: DataPair,
                   ref_atr: float,
                   output_dir: str = "charts") -> str:
    """
    Runs a fresh backtest on viz_data (last 60 days, $1,000 capital),
    collects the equity curve, trade list, and grid state timeline,
    then builds and saves the Plotly figure.
    Returns: path to the saved HTML file.
    """
    # 1. Run backtest on viz_data with $1,000 initial capital
    result = Backtester().run(
        params=strategy.to_grid_params(),
        data=viz_data,
        initial_capital=1_000.0,
        ref_atr=ref_atr,
    )

    # 2. Build Plotly figure with make_subplots(rows=3, shared_xaxes=True)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.30, 0.15],
        vertical_spacing=0.02,
    )

    # Panel 1: price candlestick (1m bars, downsampled to 15m for readability)
    # + grid level traces + regime shading + trade markers

    # Panel 2: equity curve + $1,000 baseline + annotation box

    # Panel 3: drawdown filled area

    fig.update_layout(
        title=f"{strategy.symbol} | Strategy #{strategy.id} | "
              f"Last 60 Days | $1,000 Virtual Portfolio",
        template="plotly_dark",
        hovermode="x unified",
        height=900,
    )

    html_path = f"{output_dir}/{strategy.symbol}_{strategy.id}.html"
    png_path  = f"{output_dir}/{strategy.symbol}_{strategy.id}.png"
    fig.write_html(html_path)
    fig.write_image(png_path)   # requires kaleido
    return html_path
```

**State timeline collection (fed from `GridEngine.run`):**
The engine records a `state_log` alongside the equity curve:
```python
@dataclass
class StateEvent:
    timestamp: datetime
    event: str        # "GRID_FORMED" | "GRID_RESET" | "BREAKOUT" |
                      # "REGIME_PAUSE" | "REGIME_RESUME" | "PULLBACK_CONFIRMED"
    price: float
    grid_center: float | None
    grid_levels: GridLevels | None
```
This drives the regime shading, grid line traces, and vertical reset markers on Panel 1.

---

### 17. `view_strategy.py` — CLI Viewer

```bash
python view_strategy.py --id 42
python view_strategy.py --top 5 --sort sharpe
python view_strategy.py --symbol BTCUSDT --top 3
```

```python
def main():
    # Parse args: --id, --top, --sort, --symbol
    strategies = db.get_top_strategies(...)  # or get by id

    for strat in strategies:
        _, _, viz_data = cache.get_split(strat.symbol)
        ref_atr = ... # loaded from db or recomputed from opt period
        path = generate_chart(strat, viz_data, ref_atr)
        log.info(f"Chart saved: {path}")
        # Open in default browser automatically
        import webbrowser
        webbrowser.open(path)
```

---

## Implementation Order

1. `requirements.txt` → install
2. `config.py`
3. `data/fetcher.py` + `data/cache.py` (4h, 1m with ghost-bar fill, funding rates)
4. `data/screener.py`
5. `strategy/indicators.py` (ATR, ADX, DFA-Hurst, VPVR, BB Width + squeeze percentile)
6. `strategy/grid_engine.py` (dual-loop 4h/1m, lazy VPVR, warm-up, funding, delta neutrality, LVN snap, dynamic slippage, account DD kill)
7. `backtester/engine.py` + `backtester/metrics.py` (+ DSR, new metrics)
8. `optimization/parameter_space.py` (+ max_open_levels constraint)
9. `optimization/optimizer.py` (NSGAIISampler, NumpyOHLCV conversion, loky parallel, plateau scoring)
10. `optimization/wfo.py` (rolling + anchored + holdout + compounding OOS equity)
11. `optimization/monte_carlo.py` (permutation significance test, numba JIT)
12. `database/models.py` + `database/storage.py`
13. `visualization/equity_chart.py` + `view_strategy.py`
14. `main.py`

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backtesting timeframe | Dual: 4h signals + 1m fills | 4h for stable indicators; 1m near-exact fill simulation (~0.05% per bar, grid steps are 0.3–2%) |
| Fill trigger | bar_1m.low/high crosses level | No look-ahead bias; multi-level fills per 1m bar are extremely rare on liquid futures |
| Bar traversal | worst-case path within 1m bar | Final safety net only; almost never triggers at 1m granularity |
| Fee model | Maker (0.02%) + 0 slippage for limits; taker (0.05%) + dynamic 3–30 bps for forced exits | Slippage scales with ATR ratio — realistic flash crash cost without fixed underestimation |
| Ghost bars | 1m data reindexed to perfect continuous DatetimeIndex; ffill close, volume=0 | Eliminates simulation gaps, NaN indicators, and misaligned inner-loop timestamps |
| Memory management | df_1m → NumpyOHLCV arrays before parallel trials; loky backend | Prevents 200MB × N-workers OOM; workers share mmap'd arrays with zero serialization cost |
| VPVR compute | Lazy — computed only on TRACKING→GRID transition, not every 4h bar | Eliminates O(vpvr_window) overhead on bars where no new grid is being formed |
| Position size | Inverse ATR + Bollinger Squeeze cap | Locks size to minimum during squeeze (BB width < 10th pct) to prevent pre-breakout overexposure |
| Optimization | Optuna NSGAIISampler 500 trials | Single-stage multi-objective Pareto; replaces redundant GA→TPE pipeline |
| Multiple-testing bias | Holdout quarantine + Deflated Sharpe Ratio | Quarantine prevents holdout exhaustion; DSR penalizes Sharpe by total trial count |
| Hurst method | DFA (Detrended Fluctuation Analysis) | 10× faster than R/S, more statistically stable, same interpretation |
| VPVR distribution | volume spread across full H-L range per bar | More accurate POC than close-only histogram |
| WFO | rolling + anchored, both must pass | Rolling detects regime sensitivity; anchored validates long-term stability |
| Data split | 80% opt / holdout / last-60d viz | Zero contamination; visualization slice always the most recent unseen data |
| Metrics | Sharpe + Sortino + DSR + MaxDD + PF + CAGR + Calmar | DSR is the key robustness metric; Sortino better than Sharpe for asymmetric grid returns |
| Deduplication | SHA256 param hash | Prevents storing the same strategy discovered multiple times |

---

## Fees & Slippage Model

| Order type | Fee | Slippage | When |
|------------|-----|----------|------|
| Grid limit fill (maker) | 0.02% per side | **0%** (resting limit) | Normal grid fills |
| Forced market exit (taker) | 0.05% per side | **Dynamic 3–30 bps** | Regime kill / account DD stop |

Normal round-trip drag: **0.04%** (vs old 0.16% — 4× more realistic for resting limit orders).

**Dynamic slippage formula** (forced market exits only):
```python
atr_ratio    = current_atr / ref_atr
dynamic_slip = market_slippage_base + (
    (market_slippage_max - market_slippage_base)
    * min(atr_ratio / market_slippage_atr_cap, 1.0)
)
# Calibration:
#   ATR ratio 1.0 (normal)   → 3 bps
#   ATR ratio 2.0 (elevated) → 17 bps
#   ATR ratio ≥ 3.0 (crash)  → 30 bps (ceiling)
```

```python
# Normal limit fill (maker — zero slippage):
pnl = (exit_price - entry_price) * qty
    - (entry_notional + exit_notional) * maker_fee

# Forced market exit (taker + dynamic slippage):
exit_fill  = market_price * (1 - dynamic_slip)   # for longs
forced_pnl = (exit_fill - entry_price) * qty
           - entry_notional * maker_fee
           - exit_fill * qty * taker_fee
```

---

## Verification Steps

1. `python -c "from data.fetcher import BinanceFetcher; BinanceFetcher().get_all_usdt_symbols()"` — data connectivity
2. `python -m data.screener` — symbol screening with correct filters
3. `python -m backtester.engine` on BTCUSDT (4h + 1m) — dual-timeframe backtest sanity check
4. `python view_strategy.py --top 5` — generates and opens 5 interactive charts (once strategies exist in DB)
5. `python main.py` — infinite loop starts, logs cycle completions and strategy discoveries
