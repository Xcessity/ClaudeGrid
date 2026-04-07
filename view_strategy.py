"""
view_strategy.py — CLI chart viewer

Usage:
    python view_strategy.py --id 42
    python view_strategy.py --top 5 --sort avg_oos_sharpe
    python view_strategy.py --symbol BTCUSDT --top 3
    python view_strategy.py --top 1 --no-browser
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser

import numpy as np
from loguru import logger

from config import config
from data.cache import DataCache
from database.storage import StrategyDB
from strategy.indicators import compute_atr
from visualization.equity_chart import generate_chart


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View stored ClaudeGrid strategies as interactive Plotly charts."
    )
    parser.add_argument("--id",       type=int,   default=None, help="Strategy ID to view")
    parser.add_argument("--top",      type=int,   default=1,    help="View top-N strategies")
    parser.add_argument("--sort",     type=str,   default="avg_oos_sharpe",
                        help="Sort column (default: avg_oos_sharpe)")
    parser.add_argument("--symbol",   type=str,   default=None, help="Filter by symbol")
    parser.add_argument("--no-browser", action="store_true",
                        help="Save charts but do not open browser")
    args = parser.parse_args()

    db    = StrategyDB()
    cache = DataCache()

    # ── Load strategies ───────────────────────────────────────────────────────
    if args.id is not None:
        strat = db.get_strategy_by_id(args.id)
        if strat is None:
            logger.error(f"Strategy #{args.id} not found in database.")
            sys.exit(1)
        strategies = [strat]
    elif args.symbol is not None:
        strategies = db.get_strategies_by_symbol(args.symbol)[: args.top]
        if not strategies:
            logger.error(f"No strategies found for {args.symbol}.")
            sys.exit(1)
    else:
        strategies = db.get_top_strategies(n=args.top, sort_by=args.sort)
        if not strategies:
            logger.warning("No strategies in database yet.")
            sys.exit(0)

    logger.info(f"Generating charts for {len(strategies)} strategy/strategies …")

    for strat in strategies:
        symbol = strat.symbol
        logger.info(f"  #{strat.id} {symbol} | OOS Sharpe={strat.avg_oos_sharpe:.2f}")

        try:
            _, _, viz_data = cache.get_split(symbol)

            # ref_atr must use period=14, matching how main.py computes it.
            # Using strat.atr_period would break position sizing / dynamic slippage
            # because the stored strategy was sized against the period-14 ref_atr.
            opt_data, _, _ = cache.get_split(symbol)
            atr_series = compute_atr(opt_data.df_4h, period=14).dropna()
            ref_atr = float(np.median(atr_series)) if len(atr_series) > 0 else 1.0
            ref_atr = max(ref_atr, 1e-8)

            path = generate_chart(strat, viz_data, ref_atr)

            if not args.no_browser:
                abs_path = os.path.abspath(path).replace(os.sep, "/")
                webbrowser.open(f"file:///{abs_path}")

        except Exception as exc:
            logger.error(f"  Failed to generate chart for #{strat.id} {symbol}: {exc}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
