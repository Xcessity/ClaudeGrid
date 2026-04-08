"""
main.py — ClaudeGrid infinite search loop

Cycle per symbol:
    1. NSGAIISampler multi-objective optimization (Sharpe↑ Sortino↑ MaxDD↓)
    2. Monte Carlo permutation significance test
    3. Rolling + Anchored WFO → Holdout validation
    4. Deflated Sharpe Ratio (multiple-testing correction)
    5. DB storage if all gates pass

Holdout quarantine:
    Each symbol gets at most config.holdout_max_failures consecutive holdout
    failures before it is quarantined for config.holdout_quarantine_days days.
    This prevents exhausting the holdout's statistical power on one symbol.
"""
from __future__ import annotations

import itertools
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from loguru import logger
from tqdm import tqdm

from config import config
from data.cache import DataCache
from data.fetcher import BinanceFetcher
from data.screener import SymbolScreener
from database.storage import StrategyDB
from backtester.metrics import compute_deflated_sharpe
from optimization.monte_carlo import monte_carlo_significance
from optimization.optimizer import MultiObjectiveOptimizer
from optimization.parameter_space import params_hash
from optimization.wfo import WalkForwardOptimizer
from strategy.indicators import compute_atr

# ── Logging setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True,
)
logger.add(
    "logs/search_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="14 days",
    level="DEBUG",
    enqueue=True,
)


class _Spinner:
    """Context manager: shows a moving spinner + elapsed time on stderr."""
    _CHARS = r"|/-\\"

    def __init__(self, label: str):
        self._label = label
        self._stop  = threading.Event()
        self._t     = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        t0 = time.monotonic()
        for ch in itertools.cycle(self._CHARS):
            if self._stop.is_set():
                break
            elapsed = int(time.monotonic() - t0)
            sys.stderr.write(f"\r  {ch}  {self._label}  [{elapsed}s]   ")
            sys.stderr.flush()
            time.sleep(0.12)
        sys.stderr.write("\r" + " " * 80 + "\r")
        sys.stderr.flush()

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()


def main() -> None:
    import os
    os.makedirs("logs", exist_ok=True)

    # ── Startup ────────────────────────────────────────────────────────────────
    db      = StrategyDB()
    fetcher = BinanceFetcher()
    cache   = DataCache()

    logger.info("ClaudeGrid starting — screening symbols …")
    all_symbols = fetcher.get_all_usdt_symbols()
    screener    = SymbolScreener()
    qualified   = screener.screen(all_symbols, config.signal_timeframe)

    if not qualified:
        logger.error("No symbols passed screening — check filters and connectivity.")
        sys.exit(1)

    logger.info(f"{len(qualified)} symbols qualified: {qualified[:5]} …")

    # Pre-fetch and cache 4h + 1m + funding for every qualified symbol
    logger.info("Pre-fetching data for all qualified symbols …")
    for sym in tqdm(qualified, desc="Caching 4h+1m+funding"):
        try:
            cache.get_or_fetch_both(sym)
        except Exception as exc:
            logger.warning(f"  {sym}: fetch failed — {exc}")

    # Load existing hashes to skip duplicates without hitting the DB on every trial
    existing_hashes: set[str] = db.get_all_params_hashes()

    # Per-symbol state (survives across cycles)
    holdout_failures:           dict[str, int]      = {}
    quarantine_until:           dict[str, datetime] = {}
    total_trials_per_symbol:    dict[str, int]      = {}
    evaluated_hashes:           set[str]            = set()
    search_cycle = 0
    symbol_idx   = 0

    logger.info("Entering infinite search loop …")

    # ── Infinite Loop ──────────────────────────────────────────────────────────
    while True:
        symbol = qualified[symbol_idx % len(qualified)]
        symbol_idx += 1

        # ── Holdout quarantine check ─────────────────────────────────────────
        until = quarantine_until.get(symbol)
        if until is not None and datetime.now(timezone.utc) < until:
            logger.warning(
                f"[{symbol}] Quarantined until {until.date()} — skipping"
            )
            continue

        search_cycle += 1
        logger.info(f"── Cycle {search_cycle} | {symbol} ──────────────────────")

        # ── Load data slices ─────────────────────────────────────────────────
        try:
            opt_data, holdout_data, viz_data = cache.get_split(symbol)
        except Exception as exc:
            logger.error(f"[{symbol}] Data split failed: {exc}")
            continue

        if len(opt_data.df_4h) < config.wfo_is_bars + config.wfo_oos_bars:
            logger.warning(f"[{symbol}] Insufficient optimization data — skipping")
            continue

        # ref_atr: median ATR(14) over the full optimization period
        try:
            atr_series = compute_atr(opt_data.df_4h, period=14).dropna()
            ref_atr    = float(np.median(atr_series)) if len(atr_series) > 0 else 1.0
            ref_atr    = max(ref_atr, 1e-8)
        except Exception as exc:
            logger.error(f"[{symbol}] ref_atr computation failed: {exc}")
            continue

        # ── Phase 1: Multi-objective Pareto search ───────────────────────────
        logger.info(f"[{symbol}] Running NSGAIISampler ({config.n_optuna_trials} trials) …")
        try:
            optimizer = MultiObjectiveOptimizer(
                opt_data=opt_data,
                n_trials=config.n_optuna_trials,
                n_workers=config.n_workers,
            )
            study  = optimizer.run()   # tqdm progress bar shown by Optuna
            stable = optimizer.get_stable_params(study)   # top-5 Pareto + plateau
            pw_scores = optimizer.score_plateau_width(study)
            avg_pw = float(np.mean(list(pw_scores.values()))) if pw_scores else 0.0
        except Exception as exc:
            logger.error(f"[{symbol}] Optimization failed: {exc}")
            continue

        total_trials_per_symbol[symbol] = (
            total_trials_per_symbol.get(symbol, 0) + config.n_optuna_trials
        )

        if not stable:
            logger.info(f"[{symbol}] No stable Pareto candidates found this cycle")
            continue

        logger.info(f"[{symbol}] {len(stable)} stable candidates → WFO validation")

        # ── Phase 2: WFO + holdout validation ───────────────────────────────
        wfo = WalkForwardOptimizer(symbol, opt_data, holdout_data, ref_atr)

        for params in stable:
            p_hash = params_hash(params)
            if p_hash in existing_hashes or p_hash in evaluated_hashes:
                logger.debug(f"[{symbol}] Duplicate params — skipping")
                continue
            evaluated_hashes.add(p_hash)

            # ── WFO: rolling + anchored + holdout ────────────────────────────
            try:
                with _Spinner(f"[{symbol}] WFO validate"):
                    result = wfo.validate(params, plateau_width_score=avg_pw)
            except Exception as exc:
                logger.error(f"[{symbol}] WFO failed: {exc}")
                continue

            if not result.is_valid:
                holdout_failures[symbol] = holdout_failures.get(symbol, 0) + 1
                logger.debug(
                    f"[{symbol}] WFO invalid "
                    f"(failures={holdout_failures[symbol]}/{config.holdout_max_failures})"
                )
                if holdout_failures[symbol] >= config.holdout_max_failures:
                    until_dt = datetime.now(timezone.utc) + timedelta(
                        days=config.holdout_quarantine_days
                    )
                    quarantine_until[symbol] = until_dt
                    logger.warning(
                        f"[{symbol}] Quarantined for {config.holdout_quarantine_days}d "
                        f"after {holdout_failures[symbol]} holdout failures"
                    )
                continue

            # ── Monte Carlo significance test (only for WFO-passing candidates) ──
            try:
                with _Spinner(f"[{symbol}] Monte Carlo ({config.mc_n_shuffles} shuffles)"):
                    p_value = monte_carlo_significance(
                        params, opt_data, ref_atr,
                        n_shuffles=config.mc_n_shuffles,
                    )
            except Exception as exc:
                logger.warning(f"[{symbol}] Monte Carlo failed: {exc}")
                continue

            if p_value > config.mc_significance:
                logger.debug(
                    f"[{symbol}] MC rejected: p={p_value:.3f} "
                    f"(threshold={config.mc_significance})"
                )
                continue

            # ── Deflated Sharpe Ratio ────────────────────────────────────────
            hm = result.holdout_metrics or {}
            n_bars = len(opt_data.df_1m)
            try:
                dsr = compute_deflated_sharpe(
                    sharpe=result.avg_oos_sharpe,
                    n_trials=total_trials_per_symbol[symbol],
                    n_bars=n_bars,
                    skewness=hm.get("return_skewness", 0.0),
                    kurtosis=hm.get("return_kurtosis", 3.0),
                )
            except Exception as exc:
                logger.warning(f"[{symbol}] DSR computation failed: {exc}")
                dsr = result.avg_oos_sharpe   # fallback: use raw Sharpe

            if dsr < config.min_sharpe:
                logger.debug(
                    f"[{symbol}] DSR rejected: DSR={dsr:.2f} "
                    f"(raw Sharpe={result.avg_oos_sharpe:.2f}, "
                    f"n_trials={total_trials_per_symbol[symbol]})"
                )
                continue

            # ── All gates passed — persist ───────────────────────────────────
            try:
                new_id = db.save_strategy(
                    result,
                    dsr=dsr,
                    mc_p_value=p_value,
                    mc_n_shuffles=config.mc_n_shuffles,
                    search_cycle=search_cycle,
                )
            except Exception as exc:
                logger.error(f"[{symbol}] DB save failed: {exc}")
                continue

            if new_id is None:
                logger.debug(f"[{symbol}] Strategy already in DB (race condition)")
                continue

            existing_hashes.add(p_hash)
            holdout_failures[symbol] = 0  # reset on success

            logger.success(
                f"[Cycle {search_cycle}] Strategy #{new_id} saved | "
                f"{symbol} | "
                f"OOS Sharpe={result.avg_oos_sharpe:.2f} | "
                f"DSR={dsr:.2f} | "
                f"MC p={p_value:.3f} | "
                f"Holdout CAGR={hm.get('cagr', 0):.1%} | "
                f"Consistency={result.consistency_score:.0%}"
            )

        logger.info(
            f"[Cycle {search_cycle}] {symbol} done | DB total: {db.strategy_count()}"
        )


if __name__ == "__main__":
    main()
