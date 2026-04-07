"""
strategy/grid_engine.py

Dual-loop grid trading simulation state machine.

Outer loop: 4h signal bars — compute indicators, manage regime, build/reset grid.
Inner loop: 1m fill bars  — simulate limit order fills, track equity curve.

Fee model:
  Resting limit orders (grid fills):       maker_fee = 0.02%,  zero slippage
  Forced market exits (regime/DD kill):    taker_fee = 0.05% + dynamic slippage

Dynamic slippage (forced exits only):
  slip = base + (max - base) * min(current_atr / (ref_atr * atr_cap), 1.0)
  Normal ATR (ratio=1.0) → 3 bps; flash crash (3× ATR) → 30 bps ceiling

ATR floor: max(current_atr, 1e-8) on every ATR division to prevent
ZeroDivisionError during exchange outages and ghost-bar flat periods.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import config
from strategy.indicators import (
    compute_atr,
    compute_adx,
    compute_hurst_dfa,
    compute_bb_width,
    compute_bb_width_percentile,
    compute_volatility_adjusted_size,
    compute_vpvr,
)

# ── State constants ────────────────────────────────────────────────────────────
GRID_MODE     = "GRID_MODE"
TRACKING_MODE = "TRACKING_MODE"
KILLED        = "KILLED"


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class GridParams:
    atr_period: int           = 14
    atr_multiplier: float     = 0.5
    geometric_ratio: float    = 1.2
    n_levels: int             = 4
    pullback_pct: float       = 1.5
    hurst_window: int         = 100
    hurst_threshold: float    = 0.55
    adx_period: int           = 14
    adx_threshold: float      = 25.0
    vpvr_window: int          = 100
    use_vpvr_anchor: bool     = True
    position_size_pct: float  = 2.0
    max_open_levels: int      = 4


@dataclass
class GridLevels:
    center:      float
    buys:        list   # ascending: buys[0] = farthest (lowest), buys[-1] = closest to center
    sells:       list   # ascending: sells[0] = closest to center, sells[-1] = farthest (highest)
    lower_bound: float  # = buys[0]  — breakout trigger
    upper_bound: float  # = sells[-1] — breakout trigger


@dataclass
class Trade:
    side:        str    # "long" or "short"
    entry_price: float
    exit_price:  float
    qty:         float
    entry_time:  object
    exit_time:   object
    pnl:         float
    hold_bars:   int    # 1m bars held


@dataclass
class Position:
    side:        str    # "long" or "short"
    entry_price: float
    qty:         float
    tp_price:    float
    entry_time:  object
    entry_fee:   float
    level_idx:   int    # index into grid buys[] or sells[]
    hold_bars:   int = 0


@dataclass
class SimResult:
    equity_curve:    pd.Series
    trades:          list
    n_grid_resets:   int
    n_regime_pauses: int
    killed_early:    bool


# ── Grid construction ──────────────────────────────────────────────────────────

def _build_geometric_levels(center: float, atr: float, params: GridParams) -> GridLevels:
    """
    Raw geometric grid.
    distance[i] = atr * atr_multiplier * geometric_ratio^i   (i=0 is closest to center)
    ATR floor prevents ZeroDivisionError.
    """
    atr_safe = max(atr, 1e-8)
    raw_buys:  list = []
    raw_sells: list = []
    for i in range(params.n_levels):
        dist = atr_safe * params.atr_multiplier * (params.geometric_ratio ** i)
        raw_buys.append(center - dist)
        raw_sells.append(center + dist)
    raw_buys.sort()   # ascending: [lowest … closest_to_center]
    raw_sells.sort()  # ascending: [closest_to_center … highest]
    return GridLevels(
        center=center,
        buys=raw_buys,
        sells=raw_sells,
        lower_bound=raw_buys[0],
        upper_bound=raw_sells[-1],
    )


def _snap_to_hvn(levels: list, vpvr: dict, atr: float, params: GridParams) -> list:
    """
    Snap each level that falls in an LVN zone to the nearest HVN bin.
    If no HVN is within tolerance, keep the raw level.
    """
    lvn_list = vpvr.get("lvn", [])
    hvn_list = vpvr.get("hvn", [])
    if not lvn_list or not hvn_list:
        return levels

    tol = 0.5 * params.atr_multiplier * max(atr, 1e-8)
    snapped = []
    for lv in levels:
        if any(abs(lv - l) < tol for l in lvn_list):
            candidates = [h for h in hvn_list if abs(lv - h) < tol]
            snapped.append(min(candidates, key=lambda h: abs(lv - h)) if candidates else lv)
        else:
            snapped.append(lv)
    return snapped


def build_grid_hvn_snapped(
    center: float, atr: float, vpvr: dict, params: GridParams
) -> GridLevels:
    """Build geometric grid with LVN-avoidance snapping toward nearest HVN."""
    gl = _build_geometric_levels(center, atr, params)
    snapped_buys  = sorted(_snap_to_hvn(gl.buys,  vpvr, atr, params))
    snapped_sells = sorted(_snap_to_hvn(gl.sells, vpvr, atr, params))
    return GridLevels(
        center=center,
        buys=snapped_buys,
        sells=snapped_sells,
        lower_bound=snapped_buys[0],
        upper_bound=snapped_sells[-1],
    )


def nearest_hvn(price: float, vpvr: dict) -> float:
    """Return the HVN bin center closest to price, or price itself if no HVNs."""
    hvn = vpvr.get("hvn", [])
    if not hvn:
        return price
    return min(hvn, key=lambda h: abs(h - price))


# ── P&L / equity helpers ──────────────────────────────────────────────────────

def _dynamic_slippage(current_atr: float, ref_atr: float) -> float:
    """
    Slippage for forced market exits (regime kill / account DD stop).
    Linearly interpolates from base (normal) to max (3× ATR flash crash).
    ATR floor on both operands prevents ZeroDivisionError.
    """
    atr_ratio = max(current_atr, 1e-8) / max(ref_atr, 1e-8)
    return config.market_slippage_base + (
        (config.market_slippage_max - config.market_slippage_base)
        * min(atr_ratio / config.market_slippage_atr_cap, 1.0)
    )


def _mark_to_market(
    open_longs: list, open_shorts: list, mark_price: float, capital: float
) -> float:
    """Total equity = free capital + sum of unrealized PnL on all open positions."""
    unrealized = sum((mark_price - p.entry_price) * p.qty for p in open_longs)
    unrealized += sum((p.entry_price - mark_price) * p.qty for p in open_shorts)
    return capital + unrealized


def _net_delta(
    open_longs: list, open_shorts: list, mark_price: float
) -> float:
    """Net position value in price units. Positive = net long, negative = net short."""
    return (sum(p.qty * mark_price for p in open_longs)
            - sum(p.qty * mark_price for p in open_shorts))


def _close_all(
    open_longs: list,
    open_shorts: list,
    exit_price: float,
    slippage: float,
    capital: float,
    trades: list,
    exit_time,
) -> float:
    """
    Force-close all positions at market price (taker fee + dynamic slippage).
    Modifies open_longs/open_shorts in-place (clears them).
    Returns updated capital (realized PnL credited).
    """
    for pos in open_longs:
        fill = exit_price * (1.0 - slippage)
        fee  = fill * pos.qty * config.taker_fee
        pnl  = (fill - pos.entry_price) * pos.qty - pos.entry_fee - fee
        capital += pnl
        trades.append(Trade(
            side="long", entry_price=pos.entry_price, exit_price=fill,
            qty=pos.qty, entry_time=pos.entry_time, exit_time=exit_time,
            pnl=pnl, hold_bars=pos.hold_bars,
        ))
    for pos in open_shorts:
        fill = exit_price * (1.0 + slippage)
        fee  = fill * pos.qty * config.taker_fee
        pnl  = (pos.entry_price - fill) * pos.qty - pos.entry_fee - fee
        capital += pnl
        trades.append(Trade(
            side="short", entry_price=pos.entry_price, exit_price=fill,
            qty=pos.qty, entry_time=pos.entry_time, exit_time=exit_time,
            pnl=pnl, hold_bars=pos.hold_bars,
        ))
    open_longs.clear()
    open_shorts.clear()
    return capital


# ── Fill simulation (per 1m bar) ──────────────────────────────────────────────

def _check_tp_exits(
    bar_1m,
    open_longs: list,
    open_shorts: list,
    capital: float,
    trades: list,
) -> float:
    """
    Process take-profit exits for all open positions.
    Called every 1m bar regardless of state (GRID_MODE or TRACKING_MODE),
    so old positions from previous grids can still close at their TPs.

    Limit fills — no slippage, maker fee.
    Returns updated capital.
    """
    bar_time = bar_1m.Index

    new_longs: list = []
    for pos in open_longs:
        if bar_1m.high >= pos.tp_price:
            fee = pos.tp_price * pos.qty * config.maker_fee
            pnl = (pos.tp_price - pos.entry_price) * pos.qty - pos.entry_fee - fee
            capital += pnl
            trades.append(Trade(
                side="long", entry_price=pos.entry_price, exit_price=pos.tp_price,
                qty=pos.qty, entry_time=pos.entry_time, exit_time=bar_time,
                pnl=pnl, hold_bars=pos.hold_bars,
            ))
        else:
            pos.hold_bars += 1
            new_longs.append(pos)
    open_longs[:] = new_longs

    new_shorts: list = []
    for pos in open_shorts:
        if bar_1m.low <= pos.tp_price:
            fee = pos.tp_price * pos.qty * config.maker_fee
            pnl = (pos.entry_price - pos.tp_price) * pos.qty - pos.entry_fee - fee
            capital += pnl
            trades.append(Trade(
                side="short", entry_price=pos.entry_price, exit_price=pos.tp_price,
                qty=pos.qty, entry_time=pos.entry_time, exit_time=bar_time,
                pnl=pnl, hold_bars=pos.hold_bars,
            ))
        else:
            pos.hold_bars += 1
            new_shorts.append(pos)
    open_shorts[:] = new_shorts

    return capital


def _open_new_positions(
    bar_1m,
    grid_levels: GridLevels,
    open_longs: list,
    open_shorts: list,
    capital: float,
    current_equity: float,
    adjusted_size: float,
    params: GridParams,
) -> float:
    """
    Open new limit positions at unfilled grid levels.

    Worst-case traversal (safety net at 1m — rarely triggers):
      Buy  fills: bar_1m.low  <= buy_level  → fill at buy_level  (no slippage, maker fee)
      Sell fills: bar_1m.high >= sell_level → fill at sell_level (no slippage, maker fee)

    Delta-neutrality check before each new entry.
    Returns updated capital (entry fees deducted).
    """
    bar_time = bar_1m.Index

    occupied_buy  = {p.level_idx for p in open_longs}
    occupied_sell = {p.level_idx for p in open_shorts}
    n_open = len(open_longs) + len(open_shorts)

    buys  = grid_levels.buys
    sells = grid_levels.sells

    # ── New long entries (buy levels below center) ─────────────────────────
    for i, buy_price in enumerate(buys):
        if n_open >= params.max_open_levels:
            break
        if i in occupied_buy:
            continue
        if bar_1m.low > buy_price:
            continue
        # Delta neutrality: skip if already net-long beyond threshold
        delta = _net_delta(open_longs, open_shorts, bar_1m.close)
        if delta > config.max_delta_pct * current_equity:
            continue
        qty = max((current_equity * adjusted_size / 100.0) / buy_price, 0.0)
        if qty == 0.0:
            continue
        entry_fee = buy_price * qty * config.maker_fee
        capital -= entry_fee
        # TP: next level up in buys[] (ascending order) or center for top buy
        tp = buys[i + 1] if i + 1 < len(buys) else grid_levels.center
        open_longs.append(Position(
            side="long", entry_price=buy_price, qty=qty,
            tp_price=tp, entry_time=bar_time, entry_fee=entry_fee, level_idx=i,
        ))
        occupied_buy.add(i)
        n_open += 1

    # ── New short entries (sell levels above center) ───────────────────────
    for j, sell_price in enumerate(sells):
        if n_open >= params.max_open_levels:
            break
        if j in occupied_sell:
            continue
        if bar_1m.high < sell_price:
            continue
        # Delta neutrality: skip if already net-short beyond threshold
        delta = _net_delta(open_longs, open_shorts, bar_1m.close)
        if delta < -config.max_delta_pct * current_equity:
            continue
        qty = max((current_equity * adjusted_size / 100.0) / sell_price, 0.0)
        if qty == 0.0:
            continue
        entry_fee = sell_price * qty * config.maker_fee
        capital -= entry_fee
        # TP: next level down in sells[] (ascending order) or center for bottom sell
        tp = sells[j - 1] if j > 0 else grid_levels.center
        open_shorts.append(Position(
            side="short", entry_price=sell_price, qty=qty,
            tp_price=tp, entry_time=bar_time, entry_fee=entry_fee, level_idx=j,
        ))
        occupied_sell.add(j)
        n_open += 1

    return capital


# ── Main engine ────────────────────────────────────────────────────────────────

class GridEngine:
    """
    Dual-loop neutral grid trading simulation.

    States and transitions:
      GRID_MODE     → TRACKING_MODE   if regime turns trending (Hurst > threshold OR ADX > threshold)
                                       OR price escapes grid bounds
      TRACKING_MODE → GRID_MODE       if regime cleared AND pullback_pct% retraction from extreme
      Any           → KILLED          if equity drawdown from peak > max_account_dd_pct
    """

    def run(
        self,
        df_4h: pd.DataFrame,
        df_1m: pd.DataFrame,
        funding_rates: pd.Series,
        params: GridParams,
        initial_capital: float,
        ref_atr: float,
    ) -> SimResult:
        """
        df_4h:         4h OHLCV (indicator computation, regime, grid placement)
        df_1m:         1m OHLCV (limit order fills, equity curve — 1m resolution output)
        funding_rates: 8h funding rate Series (UTC index, values = rate as fraction)
        ref_atr:       median ATR of full optimization period (stable sizing reference)
        """
        # ── Pre-compute all 4h indicators vectorized ───────────────────────────
        atr_4h      = compute_atr(df_4h, params.atr_period)
        adx_4h      = compute_adx(df_4h, params.adx_period)
        hurst_4h    = compute_hurst_dfa(df_4h["close"], params.hurst_window)
        bb_width_4h = compute_bb_width(df_4h, period=config.bb_period)
        bb_pct_4h   = compute_bb_width_percentile(bb_width_4h)

        warmup_bars = max(
            params.atr_period, params.adx_period,
            params.hurst_window, config.bb_period,
        )

        # ── Simulation state ───────────────────────────────────────────────────
        state:           str                    = GRID_MODE
        grid_levels:     Optional[GridLevels]   = None
        adjusted_size:   float                  = float(params.position_size_pct)
        extreme_price:   float                  = 0.0
        breakout_dir:    str                    = "up"
        open_longs:      List[Position]         = []
        open_shorts:     List[Position]         = []
        trades:          list                   = []
        equity_vals:     list                   = []
        equity_idx:      list                   = []
        capital:         float                  = float(initial_capital)
        peak_equity:     float                  = float(initial_capital)
        current_atr:     float                  = max(float(ref_atr), 1e-8)
        n_grid_resets:   int                    = 0
        n_regime_pauses: int                    = 0

        funding_index = (
            set(funding_rates.index)
            if funding_rates is not None and not funding_rates.empty
            else set()
        )

        # ── OUTER LOOP: 4h signal bars ─────────────────────────────────────────
        for i, bar_4h in enumerate(df_4h.itertuples()):

            # 0. Warm-up period — emit equity, skip strategy
            if i < warmup_bars:
                for bar_1m in self._period_1m(df_1m, bar_4h.Index).itertuples():
                    eq = _mark_to_market(open_longs, open_shorts, bar_1m.close, capital)
                    equity_vals.append(eq)
                    equity_idx.append(bar_1m.Index)
                continue

            # 1. Read 4h indicators for this bar
            h           = self._safe(hurst_4h.iloc[i],          0.5)
            adx_val     = self._safe(adx_4h["ADX"].iloc[i],     0.0)
            current_atr = max(self._safe(atr_4h.iloc[i], ref_atr), 1e-8)
            bb_pct      = self._safe(bb_pct_4h.iloc[i],         0.5)

            regime_trending = (h > params.hurst_threshold or adx_val > params.adx_threshold)

            # 2. Regime kill → force-close all, enter TRACKING_MODE
            if regime_trending and state == GRID_MODE:
                slip    = _dynamic_slippage(current_atr, ref_atr)
                capital = _close_all(
                    open_longs, open_shorts,
                    float(bar_4h.close), slip, capital, trades, bar_4h.Index,
                )
                state         = TRACKING_MODE
                extreme_price = float(bar_4h.close)
                breakout_dir  = (
                    "up"
                    if self._safe(adx_4h["DI_plus"].iloc[i], 0.0)
                       >= self._safe(adx_4h["DI_minus"].iloc[i], 0.0)
                    else "down"
                )
                grid_levels   = None
                n_regime_pauses += 1

            # 3. Build grid when entering GRID_MODE without active levels (lazy)
            if state == GRID_MODE and grid_levels is None:
                vpvr_slice = df_4h.iloc[max(0, i - params.vpvr_window) : i]
                vpvr = compute_vpvr(vpvr_slice, lookback=params.vpvr_window)
                center = (
                    nearest_hvn(float(bar_4h.close), vpvr)
                    if params.use_vpvr_anchor
                    else float(bar_4h.close)
                )
                grid_levels = build_grid_hvn_snapped(center, current_atr, vpvr, params)
                adjusted_size = compute_volatility_adjusted_size(
                    params.position_size_pct, current_atr, ref_atr,
                    bb_width_pct=bb_pct, min_size_pct=1.0,
                    bb_squeeze_threshold=config.bb_squeeze_pct_threshold,
                )
                n_grid_resets += 1

            # 4. INNER LOOP: 1m fill bars for this 4h period
            killed_in_period = False

            for bar_1m in self._period_1m(df_1m, bar_4h.Index).itertuples():

                # Account DD kill switch (checked before any fills)
                eq = _mark_to_market(open_longs, open_shorts, bar_1m.close, capital)
                peak_equity = max(peak_equity, eq)
                dd = (peak_equity - eq) / peak_equity if peak_equity > 0 else 0.0

                if dd > config.max_account_dd_pct:
                    slip    = _dynamic_slippage(current_atr, ref_atr)
                    capital = _close_all(
                        open_longs, open_shorts,
                        float(bar_1m.open), slip, capital, trades, bar_1m.Index,
                    )
                    state = KILLED
                    equity_vals.append(capital)
                    equity_idx.append(bar_1m.Index)
                    killed_in_period = True
                    break

                # Always check TP exits (handles positions from previous grids too)
                capital = _check_tp_exits(bar_1m, open_longs, open_shorts, capital, trades)

                # New entries only in GRID_MODE with active levels
                if state == GRID_MODE and grid_levels is not None:
                    current_equity = _mark_to_market(open_longs, open_shorts, bar_1m.close, capital)
                    capital = _open_new_positions(
                        bar_1m, grid_levels, open_longs, open_shorts,
                        capital, current_equity, adjusted_size, params,
                    )
                    # Price escaped grid → enter TRACKING_MODE (no forced close)
                    if (bar_1m.close < grid_levels.lower_bound or
                            bar_1m.close > grid_levels.upper_bound):
                        extreme_price = float(bar_1m.close)
                        breakout_dir  = (
                            "up" if bar_1m.close > grid_levels.upper_bound else "down"
                        )
                        state       = TRACKING_MODE
                        grid_levels = None

                elif state == TRACKING_MODE:
                    if breakout_dir == "up":
                        extreme_price = max(extreme_price, float(bar_1m.high))
                    else:
                        extreme_price = min(extreme_price, float(bar_1m.low))
                    retraction     = abs(bar_1m.close - extreme_price) / max(extreme_price, 1e-8)
                    regime_cleared = (
                        h <= params.hurst_threshold and adx_val <= params.adx_threshold
                    )
                    if retraction >= params.pullback_pct / 100.0 and regime_cleared:
                        state = GRID_MODE   # grid_levels is None → will build next 4h bar

                # Funding payment at each 8h timestamp
                if config.simulate_funding and bar_1m.Index in funding_index:
                    rate    = float(funding_rates.loc[bar_1m.Index])
                    net_val = (
                        sum(p.qty * float(bar_1m.close) for p in open_longs)
                        - sum(p.qty * float(bar_1m.close) for p in open_shorts)
                    )
                    capital -= net_val * rate

                eq = _mark_to_market(open_longs, open_shorts, bar_1m.close, capital)
                equity_vals.append(eq)
                equity_idx.append(bar_1m.Index)

            if state == KILLED or killed_in_period:
                break

        equity_curve = pd.Series(equity_vals, index=equity_idx, dtype=float)
        return SimResult(
            equity_curve=equity_curve,
            trades=trades,
            n_grid_resets=n_grid_resets,
            n_regime_pauses=n_regime_pauses,
            killed_early=(state == KILLED),
        )

    @staticmethod
    def _period_1m(df_1m: pd.DataFrame, bar_4h_time) -> pd.DataFrame:
        """Slice 1m data to the 4h period starting at bar_4h_time (inclusive 4h window)."""
        end = bar_4h_time + pd.Timedelta(hours=4) - pd.Timedelta(minutes=1)
        return df_1m.loc[bar_4h_time:end]

    @staticmethod
    def _safe(v, default: float) -> float:
        """Return float(v) if not NaN/None, else default."""
        try:
            if pd.isna(v):
                return float(default)
        except (TypeError, ValueError):
            pass
        return float(v)
