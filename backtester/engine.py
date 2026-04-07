"""
backtester/engine.py

Thin orchestration layer over GridEngine.

Responsibilities:
  - Compute a stable ref_atr from the full optimization dataset once
  - Delegate simulation to GridEngine
  - Wrap SimResult in BacktestResult

ref_atr is always derived from the full optimization set, never per-window,
so sizing stays consistent across all WFO IS/OOS windows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.cache import DataPair
from strategy.grid_engine import GridEngine, GridParams, SimResult, Trade
from strategy.indicators import compute_atr


@dataclass
class BacktestResult:
    equity_curve:    pd.Series  # 1m-resolution (unrealized + realized PnL)
    trades:          list       # list[Trade] — closed round-trips only
    params:          GridParams
    n_grid_resets:   int
    n_regime_pauses: int
    killed_early:    bool


def slice_data_pair(data: DataPair, start, end) -> DataPair:
    """Trim both timeframes and funding in a DataPair to [start, end]."""
    fund = data.funding
    if fund is not None and not fund.empty:
        fund_sliced = fund.loc[start:end]
    else:
        fund_sliced = fund
    return DataPair(
        df_4h=data.df_4h.loc[start:end],
        df_1m=data.df_1m.loc[start:end],
        funding=fund_sliced,
    )


class Backtester:
    """
    Runs a complete backtest for given GridParams on a DataPair slice.

    Usage:
        bt = Backtester(ref_data=opt_pair)   # ref_atr computed from opt_pair
        result = bt.run(params, data=opt_pair, initial_capital=10_000)
    """

    def __init__(self, ref_data: DataPair = None):
        """
        ref_data: full optimization-period DataPair used to compute ref_atr.
                  If None, ref_atr is estimated from whatever data is passed to run().
        """
        self._ref_data         = ref_data
        self._ref_atr_cache:   dict = {}
        self._engine           = GridEngine()

    def run(
        self,
        params: GridParams,
        data: DataPair,
        initial_capital: float = 10_000.0,
        ref_atr: float = None,
        np_4h=None,
    ) -> BacktestResult:
        """
        params:          grid strategy parameters
        data:            pre-sliced DataPair (IS or OOS window)
        initial_capital: starting equity in USDT
        ref_atr:         explicit ref_atr override (optional)
        """
        if ref_atr is None:
            ref_atr = self._get_ref_atr(params.atr_period, data)

        funding = (
            data.funding
            if data.funding is not None and not data.funding.empty
            else pd.Series(dtype=float)
        )

        sim: SimResult = self._engine.run(
            df_4h=data.df_4h,
            df_1m=data.df_1m,
            funding_rates=funding,
            params=params,
            initial_capital=initial_capital,
            ref_atr=ref_atr,
            np_4h=np_4h,
        )

        return BacktestResult(
            equity_curve=sim.equity_curve,
            trades=sim.trades,
            params=params,
            n_grid_resets=sim.n_grid_resets,
            n_regime_pauses=sim.n_regime_pauses,
            killed_early=sim.killed_early,
        )

    def _get_ref_atr(self, atr_period: int, fallback_data: DataPair) -> float:
        """
        Median ATR of the reference dataset, cached per atr_period.
        Falls back to the passed data if no ref_data was provided.
        """
        if atr_period in self._ref_atr_cache:
            return self._ref_atr_cache[atr_period]

        source_df = (
            self._ref_data.df_4h
            if self._ref_data is not None
            else fallback_data.df_4h
        )
        atr_series = compute_atr(source_df, period=atr_period).dropna()
        ref_atr = float(np.median(atr_series)) if len(atr_series) > 0 else 1.0
        ref_atr = max(ref_atr, 1e-8)

        self._ref_atr_cache[atr_period] = ref_atr
        return ref_atr
