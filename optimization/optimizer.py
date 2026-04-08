"""
optimization/optimizer.py

Multi-objective Optuna optimizer using NSGAIISampler.
Three objectives: maximize Sharpe, maximize Sortino, minimize MaxDD.

Memory-safe parallel evaluation (loky workers):
──────────────────────────────────────────────
Problem: passing df_1m (~200 MB DataFrame) to each loky worker via pickle
causes 200 MB × n_workers memory bloat and potential OOM crashes.

Fix: convert all DataFrames to contiguous NumPy arrays ONCE in __init__,
before any trial starts. Joblib/loky automatically memory-maps large NumPy
arrays when transferring to worker processes — no per-worker copy.

Pre-computed indicator lookup tables (NumpyOHLCV.atr_arrays etc.) eliminate
the O(500 trials × n_4h_bars) redundant pandas-ta calls that would otherwise
happen inside workers. Workers index directly: np_4h.atr_arrays[period][bar_i].
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from loguru import logger

from config import config
from data.cache import DataPair
from backtester.engine import Backtester, BacktestResult
from backtester.metrics import compute_metrics
from strategy.indicators import (
    compute_atr,
    compute_adx,
    compute_hurst_dfa,
    compute_bb_width,
    compute_bb_width_percentile,
)
from optimization.parameter_space import sample_optuna_params, PARAM_SPACE

# Silence Optuna's per-trial INFO logs — only warnings and errors surface.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── NumpyOHLCV ─────────────────────────────────────────────────────────────────

@dataclass
class NumpyOHLCV:
    """
    Compact, memory-mappable representation of OHLCV + pre-computed 4h indicators.

    All arrays are float64, C-contiguous so loky can mmap them across workers.
    timestamps is int64 UTC nanoseconds (view of the original DatetimeIndex).

    Indicator lookup tables (only populated for np_4h, None on np_1m):
        atr_arrays[period]   → float64 ATR array for every period in PARAM_SPACE (10–50)
        adx_arrays[period]   → float64 ADX array for every period in PARAM_SPACE (10–30)
        hurst_arrays[window] → float64 Hurst array for every window in PARAM_SPACE (50–200)
        bb_pct               → float64 BB width percentile array (fixed config.bb_period)

    Workers do index lookups (e.g. np_4h.atr_arrays[params.atr_period][i]) rather
    than calling pandas-ta, eliminating redundant per-trial computation.
    """
    open:       np.ndarray          # float64, C-contiguous
    high:       np.ndarray
    low:        np.ndarray
    close:      np.ndarray
    volume:     np.ndarray
    timestamps: np.ndarray          # int64, UTC nanoseconds

    atr_arrays:    Optional[dict] = field(default=None)  # dict[int, np.ndarray]
    adx_arrays:    Optional[dict] = field(default=None)  # dict[int, np.ndarray]
    di_plus_arrays:  Optional[dict] = field(default=None)  # dict[int, np.ndarray]
    di_minus_arrays: Optional[dict] = field(default=None)  # dict[int, np.ndarray]
    hurst_arrays:  Optional[dict] = field(default=None)  # dict[int, np.ndarray]
    bb_pct:        Optional[np.ndarray] = field(default=None)


# ── Array preparation ──────────────────────────────────────────────────────────

def prepare_numpy_arrays(df: pd.DataFrame) -> NumpyOHLCV:
    """
    Convert a standard OHLCV DataFrame to a NumpyOHLCV without indicator tables.
    Used for df_1m (fill bars) — indicators are not needed there.
    """
    return NumpyOHLCV(
        open=np.ascontiguousarray(df["open"].values,   dtype=np.float64),
        high=np.ascontiguousarray(df["high"].values,   dtype=np.float64),
        low =np.ascontiguousarray(df["low"].values,    dtype=np.float64),
        close=np.ascontiguousarray(df["close"].values, dtype=np.float64),
        volume=np.ascontiguousarray(df["volume"].values, dtype=np.float64),
        timestamps=np.asarray(df.index.view(np.int64), dtype=np.int64),
    )


def prepare_4h_arrays(df_4h: pd.DataFrame) -> NumpyOHLCV:
    """
    Prepare np_4h with ALL indicator lookup tables pre-computed for every
    value in PARAM_SPACE. One-time cost (~2–5 s on a full 5-year dataset).

    ATR  : periods 10–50   (41 arrays)
    ADX  : periods 10–30   (21 arrays)
    Hurst: windows 50–200  (151 arrays)
    BB % : single array at config.bb_period

    After this returns, workers never call pandas-ta — only array indexing.
    """
    logger.info("prepare_4h_arrays: pre-computing all indicator lookups …")
    base = prepare_numpy_arrays(df_4h)

    atr_lo,  atr_hi  = int(PARAM_SPACE["atr_period"][1]),   int(PARAM_SPACE["atr_period"][2])
    adx_lo,  adx_hi  = int(PARAM_SPACE["adx_period"][1]),   int(PARAM_SPACE["adx_period"][2])
    hst_lo,  hst_hi  = int(PARAM_SPACE["hurst_window"][1]), int(PARAM_SPACE["hurst_window"][2])

    atr_arrays: dict[int, np.ndarray] = {}
    for p in range(atr_lo, atr_hi + 1):
        arr = compute_atr(df_4h, p)
        atr_arrays[p] = np.ascontiguousarray(arr.values, dtype=np.float64)

    adx_arrays:      dict[int, np.ndarray] = {}
    di_plus_arrays:  dict[int, np.ndarray] = {}
    di_minus_arrays: dict[int, np.ndarray] = {}
    for p in range(adx_lo, adx_hi + 1):
        adx_result = compute_adx(df_4h, p)
        adx_arrays[p]      = np.ascontiguousarray(adx_result["ADX"].values,     dtype=np.float64)
        di_plus_arrays[p]  = np.ascontiguousarray(adx_result["DI_plus"].values,  dtype=np.float64)
        di_minus_arrays[p] = np.ascontiguousarray(adx_result["DI_minus"].values, dtype=np.float64)

    hurst_arrays: dict[int, np.ndarray] = {}
    for w in range(hst_lo, hst_hi + 1):
        arr = compute_hurst_dfa(df_4h["close"], w)
        hurst_arrays[w] = np.ascontiguousarray(arr.values, dtype=np.float64)

    bb_width = compute_bb_width(df_4h, config.bb_period)
    bb_pct   = compute_bb_width_percentile(bb_width)
    bb_pct_arr = np.ascontiguousarray(bb_pct.values, dtype=np.float64)

    logger.info(
        f"prepare_4h_arrays: done — "
        f"{len(atr_arrays)} ATR, {len(adx_arrays)} ADX (+ DI+/DI-), "
        f"{len(hurst_arrays)} Hurst arrays cached"
    )
    return NumpyOHLCV(
        open=base.open, high=base.high, low=base.low,
        close=base.close, volume=base.volume, timestamps=base.timestamps,
        atr_arrays=atr_arrays, adx_arrays=adx_arrays,
        di_plus_arrays=di_plus_arrays, di_minus_arrays=di_minus_arrays,
        hurst_arrays=hurst_arrays, bb_pct=bb_pct_arr,
    )


# ── Backtest bridge ────────────────────────────────────────────────────────────

def _numpy_to_dataframe(np_data: NumpyOHLCV) -> pd.DataFrame:
    """
    Reconstruct a pandas DataFrame from a NumpyOHLCV.
    The underlying numpy arrays are shared (mmap) — this wraps them in a
    DataFrame view without copying, so worker memory usage stays minimal.
    """
    idx = pd.DatetimeIndex(
        pd.to_datetime(np_data.timestamps, unit="ns", utc=True)
    )
    return pd.DataFrame(
        {
            "open":   np_data.open,
            "high":   np_data.high,
            "low":    np_data.low,
            "close":  np_data.close,
            "volume": np_data.volume,
        },
        index=idx,
    )


def _run_trial(
    params,
    np_4h: NumpyOHLCV,
    np_1m: NumpyOHLCV,
    funding: pd.Series,
    ref_atr: float,
) -> dict:
    """
    Reconstruct DataFrames from the memory-mapped NumpyOHLCV arrays and
    run a full backtest. Returns the metrics dict.

    This function is called in loky worker processes. The NumpyOHLCV arrays
    arrive via mmap (no 200 MB copy per worker). DataFrame reconstruction
    is O(n) wrapping, not a data copy, so per-worker RAM stays bounded.
    """
    try:
        df_4h = _numpy_to_dataframe(np_4h)
        df_1m = _numpy_to_dataframe(np_1m)
        data  = DataPair(df_4h=df_4h, df_1m=df_1m, funding=funding)

        bt     = Backtester()
        result = bt.run(params, data, initial_capital=10_000.0, ref_atr=ref_atr, np_4h=np_4h)
        return compute_metrics(result.equity_curve, result.trades, result.killed_early)
    except Exception as exc:
        logger.debug(f"Trial backtest error ({type(exc).__name__}): {exc}")
        return {
            "sharpe": -1.0, "sortino": -1.0, "max_drawdown": 1.0,
            "n_trades": 0, "killed_early": True,
        }


# ── MultiObjectiveOptimizer ────────────────────────────────────────────────────

class MultiObjectiveOptimizer:
    """
    Optuna NSGAIISampler — three-objective Pareto search:
        maximize Sharpe, maximize Sortino, minimize MaxDD

    Usage:
        opt = MultiObjectiveOptimizer(opt_data=opt_pair, n_trials=500)
        study = opt.run()
        top5  = opt.get_stable_params(study)
    """

    def __init__(
        self,
        opt_data: DataPair,
        n_trials: int = 500,
        n_workers: int = -1,
        eval_fn=None,           # unused — kept for API compatibility
    ):
        logger.info("MultiObjectiveOptimizer: converting DataFrames → NumPy arrays …")

        # Pre-convert ONCE before any trial starts.
        # Workers receive shared mmap arrays — no 200 MB serialization per trial.
        self.np_4h    = prepare_4h_arrays(opt_data.df_4h)
        self.np_1m    = prepare_numpy_arrays(opt_data.df_1m)

        # Funding is tiny (~5 K rows over 5 years) — keep as pd.Series, fine to pickle.
        self.funding  = (
            opt_data.funding
            if opt_data.funding is not None
            else pd.Series(dtype=float)
        )

        # Stable ref_atr from the full optimization period (period=14 default).
        atr14 = self.np_4h.atr_arrays.get(14)
        if atr14 is not None:
            valid = atr14[~np.isnan(atr14)]
            self.ref_atr = float(np.median(valid)) if len(valid) > 0 else 1.0
        else:
            self.ref_atr = 1.0
        self.ref_atr = max(self.ref_atr, 1e-8)

        self.n_trials  = n_trials
        self.n_workers = n_workers

    # ── Optuna study ──────────────────────────────────────────────────────────

    def run(self) -> optuna.Study:
        """
        Run the multi-objective NSGAIISampler study.

        NSGAIISampler implements NSGA-II natively inside Optuna:
          - population_size=50: each generation evaluates 50 trials
          - mutation_prob=None: auto (1/n_params)
          - crossover_prob=0.9: standard NSGA-II default
          - seed=42: reproducible Pareto front across runs

        directions: sharpe↑  sortino↑  max_dd↓
        """
        sampler = optuna.samplers.NSGAIISampler(
            population_size=50,
            mutation_prob=None,   # auto: 1 / n_params
            crossover_prob=0.9,
            seed=42,
        )
        study = optuna.create_study(
            directions=["maximize", "maximize", "minimize"],
            sampler=sampler,
        )

        # Capture arrays in closure — loky will mmap them to workers.
        np_4h    = self.np_4h
        np_1m    = self.np_1m
        funding  = self.funding
        ref_atr  = self.ref_atr
        min_trades = config.min_trades_per_backtest

        def objective(trial: optuna.Trial):
            params = sample_optuna_params(trial)
            m = _run_trial(params, np_4h, np_1m, funding, ref_atr)
            if m["n_trades"] < min_trades or m.get("killed_early", False):
                return (-1.0, -1.0, 1.0)
            return (m["sharpe"], m["sortino"], m["max_drawdown"])

        logger.info(
            f"Starting NSGAIISampler study: {self.n_trials} trials, "
            f"n_jobs={self.n_workers}"
        )
        with parallel_backend("loky"):
            study.optimize(
                objective,
                n_trials=self.n_trials,
                n_jobs=self.n_workers,
                show_progress_bar=True,
            )
        logger.info("Optuna study complete.")
        return study

    # ── Pareto front extraction ───────────────────────────────────────────────

    def get_pareto_front(self, study: optuna.Study) -> list:
        """
        Return all Pareto-optimal GridParams filtered to quality thresholds:
            sharpe   >= config.min_sharpe   (1.0)
            sortino  >= config.min_sortino  (1.2)
            max_dd   <= 0.30
        """
        candidates = []
        for trial in study.best_trials:          # Optuna returns Pareto-front trials
            sharpe, sortino, max_dd = trial.values
            if (
                sharpe  >= config.min_sharpe
                and sortino >= config.min_sortino
                and max_dd  <= 0.30
            ):
                from optimization.parameter_space import sample_optuna_params
                # Re-decode params from the trial's stored suggest values.
                params = _params_from_trial(trial)
                candidates.append(params)
        return candidates

    # ── Plateau width scoring ─────────────────────────────────────────────────

    def score_plateau_width(self, study: optuna.Study) -> dict[str, float]:
        """
        For each parameter dimension, compute the normalized plateau width
        across Pareto-front trials: the fraction of [low, high] where ≥80%
        of Pareto solutions reside.

        Width score 0–1; higher = more robust to parameter perturbation.
        Returns a dict: param_name → plateau_width_score.
        """
        pareto_trials = study.best_trials
        if len(pareto_trials) < 3:
            return {name: 0.0 for name in PARAM_SPACE}

        from optimization.parameter_space import PARAM_SPACE as PS
        scores: dict[str, float] = {}

        for name, (ptype, low, high) in PS.items():
            vals = []
            for t in pareto_trials:
                v = t.params.get(name)
                if v is None:
                    continue
                if ptype == "bool":
                    vals.append(1.0 if v else 0.0)
                else:
                    span = float(high) - float(low)
                    vals.append((float(v) - float(low)) / span if span > 0 else 0.0)

            if len(vals) < 3:
                scores[name] = 0.0
                continue

            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            target_count = int(np.ceil(0.80 * n))

            # Minimum sliding window that contains target_count values
            min_width = 1.0
            for i in range(n - target_count + 1):
                w = vals_sorted[i + target_count - 1] - vals_sorted[i]
                min_width = min(min_width, w)

            scores[name] = float(np.clip(1.0 - min_width, 0.0, 1.0))

        return scores

    # ── Stable parameter selection ────────────────────────────────────────────

    def get_stable_params(
        self,
        study: optuna.Study,
        min_plateau_width: float = 0.15,
    ) -> list:
        """
        Filter Pareto front to candidates with avg plateau width ≥ threshold.
        Ranked by: sharpe * (1 - max_dd) * avg_plateau_width_score.
        Returns top-5 GridParams for WFO validation.
        """
        candidates = self.get_pareto_front(study)
        if not candidates:
            return []

        pw_scores = self.score_plateau_width(study)
        avg_pw    = np.mean(list(pw_scores.values())) if pw_scores else 0.0

        # Build (score, params) list
        scored = []
        for trial in study.best_trials:
            sharpe, sortino, max_dd = trial.values
            if sharpe < config.min_sharpe or sortino < config.min_sortino or max_dd > 0.30:
                continue
            composite = sharpe * (1.0 - max_dd) * max(avg_pw, 1e-3)
            scored.append((composite, trial))

        scored.sort(key=lambda x: x[0], reverse=True)
        top5_trials = [t for _, t in scored[:5]]
        return [_params_from_trial(t) for t in top5_trials]


# ── Helper: reconstruct GridParams from a completed Optuna trial ───────────────

def _params_from_trial(trial: optuna.trial.FrozenTrial):
    """Decode GridParams from the stored suggest-values of a completed trial."""
    from strategy.grid_engine import GridParams
    from optimization.parameter_space import PARAM_SPACE, _apply_constraints
    vals: dict = {}
    for name, (ptype, low, high) in PARAM_SPACE.items():
        v = trial.params.get(name)
        if v is None:
            # Fall back to midpoint if the param is somehow missing
            if ptype == "int":
                v = (int(low) + int(high)) // 2
            elif ptype == "float":
                v = (float(low) + float(high)) / 2.0
            else:
                v = False
        vals[name] = v
    return _apply_constraints(GridParams(**vals))
