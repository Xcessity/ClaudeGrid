"""
optimization/wfo.py

Walk-Forward Optimization — rolling and anchored variants.

Windows are sized in 4h signal bars (timeframe-agnostic for indicator warmup).
Each window simulation automatically uses both the 4h and 1m slices for the
corresponding date range, via slice_data_pair().

Rolling WFO:
    IS  window[i] = [i*step, i*step + is_bars)
    OOS window[i] = [i*step + is_bars, i*step + is_bars + oos_bars)
    OOS equity is compounded: window[i] OOS starts with window[i-1] OOS end equity.

Anchored WFO:
    IS  window[i] = [0, is_bars + i*step)           — always anchored to start
    OOS window[i] = [is_bars + i*step, is_bars + i*step + oos_bars)

validate() sequence:
    1. run_rolling  — check quality thresholds
    2. run_anchored — check quality thresholds
    3. If both pass: run_holdout (final clean OOS on unseen holdout set)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config import config
from data.cache import DataPair
from backtester.engine import Backtester, BacktestResult, slice_data_pair
from backtester.metrics import compute_metrics
from strategy.grid_engine import GridParams


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    window_index: int
    wfo_type:     str          # "rolling" | "anchored"
    is_start:     pd.Timestamp
    is_end:       pd.Timestamp
    oos_start:    pd.Timestamp
    oos_end:      pd.Timestamp
    is_sharpe:    float
    is_sortino:   float
    is_return:    float
    oos_sharpe:   float
    oos_sortino:  float
    oos_return:   float
    n_trades:     int
    valid:        bool         # False if n_trades < min_trades_per_wfo_window


@dataclass
class WFOResult:
    params:            GridParams
    symbol:            str
    rolling_windows:   list[WindowResult]
    anchored_windows:  list[WindowResult]
    holdout_metrics:   Optional[dict]
    avg_oos_sharpe:    float
    avg_oos_sortino:   float
    oos_is_ratio:      float        # avg OOS sharpe / avg IS sharpe
    consistency_score: float        # fraction of valid OOS windows with positive return
    plateau_width_score: float      # injected from optimizer.score_plateau_width
    is_valid:          bool


# ── WalkForwardOptimizer ───────────────────────────────────────────────────────

class WalkForwardOptimizer:
    """
    Runs rolling and anchored WFO, then holdout validation.

    ref_atr must come from the FULL optimization dataset and must never be
    recomputed per-window — consistent sizing across all IS/OOS comparisons.
    """

    def __init__(
        self,
        symbol:       str,
        opt_data:     DataPair,
        holdout_data: DataPair,
        ref_atr:      float,
    ):
        self.symbol       = symbol
        self.opt_data     = opt_data
        self.holdout_data = holdout_data
        self.ref_atr      = ref_atr
        self._bt          = Backtester(ref_data=opt_data)

    # ── Rolling WFO ───────────────────────────────────────────────────────────

    def run_rolling(self, params: GridParams) -> list[WindowResult]:
        """
        Slide a fixed IS/OOS window across the optimization period.
        OOS equity is compounded across windows (sequential equity chain).
        """
        df_4h        = self.opt_data.df_4h
        n_4h         = len(df_4h)
        is_bars      = config.wfo_is_bars
        oos_bars     = config.wfo_oos_bars
        step_bars    = config.wfo_step_bars
        results:     list[WindowResult] = []
        compound_cap = 10_000.0

        window_idx = 0
        is_start_i = 0

        while True:
            is_end_i   = is_start_i + is_bars
            oos_start_i = is_end_i
            oos_end_i   = oos_start_i + oos_bars

            # Stop if we can't fit a full OOS window
            if oos_end_i > n_4h:
                break

            is_start_ts  = df_4h.index[is_start_i]
            is_end_ts    = df_4h.index[is_end_i - 1]
            oos_start_ts = df_4h.index[oos_start_i]
            oos_end_ts   = df_4h.index[min(oos_end_i - 1, n_4h - 1)]

            # IS backtest
            is_data  = slice_data_pair(self.opt_data, is_start_ts, is_end_ts)
            is_res   = self._bt.run(params, is_data, ref_atr=self.ref_atr)
            is_m     = compute_metrics(is_res.equity_curve, is_res.trades, is_res.killed_early)

            # OOS backtest — compounded starting capital from previous OOS window
            oos_data = slice_data_pair(self.opt_data, oos_start_ts, oos_end_ts)
            oos_res  = self._bt.run(
                params, oos_data,
                initial_capital=compound_cap,
                ref_atr=self.ref_atr,
            )
            oos_m = compute_metrics(oos_res.equity_curve, oos_res.trades, oos_res.killed_early)

            # Update compounded capital for the next window
            if len(oos_res.equity_curve) > 0:
                compound_cap = float(oos_res.equity_curve.iloc[-1])
                compound_cap = max(compound_cap, 1.0)   # floor: don't go negative

            valid = oos_m["n_trades"] >= config.min_trades_per_wfo_window

            results.append(WindowResult(
                window_index=window_idx,
                wfo_type="rolling",
                is_start=is_start_ts,
                is_end=is_end_ts,
                oos_start=oos_start_ts,
                oos_end=oos_end_ts,
                is_sharpe=float(is_m["sharpe"]),
                is_sortino=float(is_m["sortino"]),
                is_return=float(is_m.get("cagr", 0.0)),
                oos_sharpe=float(oos_m["sharpe"]),
                oos_sortino=float(oos_m["sortino"]),
                oos_return=float(oos_m.get("cagr", 0.0)),
                n_trades=int(oos_m["n_trades"]),
                valid=valid,
            ))

            is_start_i += step_bars
            window_idx += 1

        return results

    # ── Anchored WFO ──────────────────────────────────────────────────────────

    def run_anchored(self, params: GridParams) -> list[WindowResult]:
        """
        IS window always anchored to the start; expands by step_bars each iteration.
        OOS window slides forward by step_bars.
        """
        df_4h     = self.opt_data.df_4h
        n_4h      = len(df_4h)
        is_bars   = config.wfo_is_bars
        oos_bars  = config.wfo_oos_bars
        step_bars = config.wfo_step_bars
        results:  list[WindowResult] = []
        window_idx = 0

        while True:
            is_end_i    = is_bars + window_idx * step_bars
            oos_start_i = is_end_i
            oos_end_i   = oos_start_i + oos_bars

            if oos_end_i > n_4h:
                break

            is_start_ts  = df_4h.index[0]
            is_end_ts    = df_4h.index[is_end_i - 1]
            oos_start_ts = df_4h.index[oos_start_i]
            oos_end_ts   = df_4h.index[min(oos_end_i - 1, n_4h - 1)]

            is_data  = slice_data_pair(self.opt_data, is_start_ts, is_end_ts)
            is_res   = self._bt.run(params, is_data, ref_atr=self.ref_atr)
            is_m     = compute_metrics(is_res.equity_curve, is_res.trades, is_res.killed_early)

            oos_data = slice_data_pair(self.opt_data, oos_start_ts, oos_end_ts)
            oos_res  = self._bt.run(params, oos_data, ref_atr=self.ref_atr)
            oos_m    = compute_metrics(oos_res.equity_curve, oos_res.trades, oos_res.killed_early)

            valid = oos_m["n_trades"] >= config.min_trades_per_wfo_window

            results.append(WindowResult(
                window_index=window_idx,
                wfo_type="anchored",
                is_start=is_start_ts,
                is_end=is_end_ts,
                oos_start=oos_start_ts,
                oos_end=oos_end_ts,
                is_sharpe=float(is_m["sharpe"]),
                is_sortino=float(is_m["sortino"]),
                is_return=float(is_m.get("cagr", 0.0)),
                oos_sharpe=float(oos_m["sharpe"]),
                oos_sortino=float(oos_m["sortino"]),
                oos_return=float(oos_m.get("cagr", 0.0)),
                n_trades=int(oos_m["n_trades"]),
                valid=valid,
            ))

            window_idx += 1

        return results

    # ── Holdout ────────────────────────────────────────────────────────────────

    def run_holdout(self, params: GridParams) -> dict:
        """Final clean validation on the unseen holdout set. Never called until both WFO variants pass."""
        result = self._bt.run(params, self.holdout_data, ref_atr=self.ref_atr)
        return compute_metrics(result.equity_curve, result.trades, result.killed_early)

    # ── Aggregate validation ───────────────────────────────────────────────────

    def validate(self, params: GridParams, plateau_width_score: float = 0.0) -> WFOResult:
        """
        Full validation pipeline:
            1. Rolling WFO  → must pass quality thresholds
            2. Anchored WFO → must pass quality thresholds
            3. Holdout      → run only if both pass

        Returns a WFOResult with is_valid=True only when all three pass.
        """
        # ── 1. Rolling ────────────────────────────────────────────────────────
        rolling_windows = self.run_rolling(params)
        rolling_valid   = self._check_windows(rolling_windows, "rolling")

        # ── 2. Anchored ───────────────────────────────────────────────────────
        anchored_windows = self.run_anchored(params)
        anchored_valid   = self._check_windows(anchored_windows, "anchored")

        # Aggregate metrics across both sets combined (rolling wins for primary KPIs)
        all_valid_windows = [w for w in rolling_windows if w.valid]
        if not all_valid_windows:
            all_valid_windows = [w for w in anchored_windows if w.valid]

        avg_oos_sharpe  = float(np.mean([w.oos_sharpe  for w in all_valid_windows])) if all_valid_windows else 0.0
        avg_oos_sortino = float(np.mean([w.oos_sortino for w in all_valid_windows])) if all_valid_windows else 0.0
        avg_is_sharpe   = float(np.mean([w.is_sharpe   for w in all_valid_windows])) if all_valid_windows else 1e-10
        oos_is_ratio    = avg_oos_sharpe / max(abs(avg_is_sharpe), 1e-10)
        consistency     = float(sum(1 for w in all_valid_windows if w.oos_return > 0)) / max(len(all_valid_windows), 1)

        # ── 3. Holdout ────────────────────────────────────────────────────────
        holdout_metrics = None
        holdout_valid   = False
        if rolling_valid and anchored_valid:
            holdout_metrics = self.run_holdout(params)
            holdout_valid   = self._check_holdout(holdout_metrics)

        is_valid = rolling_valid and anchored_valid and holdout_valid

        logger.info(
            f"WFO validate: rolling={rolling_valid} anchored={anchored_valid} "
            f"holdout={holdout_valid} → valid={is_valid} "
            f"avg_oos_sharpe={avg_oos_sharpe:.2f} consistency={consistency:.0%}"
        )

        return WFOResult(
            params=params,
            symbol=self.symbol,
            rolling_windows=rolling_windows,
            anchored_windows=anchored_windows,
            holdout_metrics=holdout_metrics,
            avg_oos_sharpe=avg_oos_sharpe,
            avg_oos_sortino=avg_oos_sortino,
            oos_is_ratio=oos_is_ratio,
            consistency_score=consistency,
            plateau_width_score=plateau_width_score,
            is_valid=is_valid,
        )

    # ── Private threshold checkers ─────────────────────────────────────────────

    def _check_windows(self, windows: list[WindowResult], label: str) -> bool:
        """
        A set of WFO windows passes if:
          - At least config.wfo_min_windows valid windows exist
          - avg OOS sharpe  >= config.min_sharpe
          - avg OOS sortino >= config.min_sortino
          - oos_is_ratio    >= config.min_wfo_oos_is_ratio
          - consistency     >= config.min_oos_consistency
        """
        valid_wins = [w for w in windows if w.valid]
        if len(valid_wins) < config.wfo_min_windows:
            logger.debug(
                f"WFO {label}: only {len(valid_wins)} valid windows "
                f"(need {config.wfo_min_windows})"
            )
            return False

        avg_oos_sharpe  = np.mean([w.oos_sharpe  for w in valid_wins])
        avg_oos_sortino = np.mean([w.oos_sortino for w in valid_wins])
        avg_is_sharpe   = np.mean([w.is_sharpe   for w in valid_wins])
        oos_is_ratio    = float(avg_oos_sharpe) / max(abs(float(avg_is_sharpe)), 1e-10)
        consistency     = sum(1 for w in valid_wins if w.oos_return > 0) / len(valid_wins)

        passes = (
            avg_oos_sharpe  >= config.min_sharpe
            and avg_oos_sortino >= config.min_sortino
            and oos_is_ratio    >= config.min_wfo_oos_is_ratio
            and consistency     >= config.min_oos_consistency
        )
        logger.debug(
            f"WFO {label}: sharpe={avg_oos_sharpe:.2f} sortino={avg_oos_sortino:.2f} "
            f"oos_is={oos_is_ratio:.2f} consistency={consistency:.0%} → {passes}"
        )
        return passes

    def _check_holdout(self, m: dict) -> bool:
        return (
            m["sharpe"]          >= config.min_sharpe
            and m["sortino"]     >= config.min_sortino
            and m["max_drawdown"] <= 0.30
            and m["cagr"]        >= config.min_cagr
        )
