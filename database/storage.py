"""
database/storage.py

CRUD interface for the strategy database.

Design:
  - SQLite via SQLAlchemy (file: strategies.db)
  - Deduplication by param_hash (SHA-256 of rounded GridParams)
  - All DB operations are session-scoped and auto-committed
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from config import config
from database.models import Base, Strategy, Symbol, WFOWindow, init_db
from optimization.wfo import WFOResult

DB_PATH = Path(__file__).parent.parent / "strategies.db"


class StrategyDB:
    """
    Thin SQLAlchemy session wrapper for storing and querying strategies.

    Usage:
        db = StrategyDB()
        db.save_strategy(wfo_result, dsr=1.3, mc_p_value=0.02, search_cycle=1)
        top = db.get_top_strategies(n=5, sort_by="avg_oos_sharpe")
    """

    def __init__(self, db_path: Path = DB_PATH):
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        init_db(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    # ── Public API ─────────────────────────────────────────────────────────────

    def save_strategy(
        self,
        result: WFOResult,
        dsr: float = 0.0,
        mc_p_value: float = 1.0,
        search_cycle: int = 0,
    ) -> int | None:
        """
        Persist a validated WFOResult to the database.

        Returns the new strategy id, or None if the param_hash already exists.
        """
        from optimization.parameter_space import params_hash

        p_hash = params_hash(result.params)

        with self._Session() as session:
            # Dedup check
            existing = (
                session.query(Strategy)
                .filter_by(param_hash=p_hash)
                .first()
            )
            if existing is not None:
                logger.debug(f"Duplicate strategy skipped: hash={p_hash[:12]}…")
                return None

            # Ensure the Symbol row exists
            sym_row = (
                session.query(Symbol)
                .filter_by(symbol=result.symbol)
                .first()
            )
            if sym_row is None:
                sym_row = Symbol(
                    symbol=result.symbol,
                    timeframe=config.signal_timeframe,
                )
                session.add(sym_row)
                session.flush()

            # Summarise compounded OOS performance from rolling windows
            rolling_valid = [w for w in result.rolling_windows if w.valid]
            compounded_oos_return = (
                sum(w.oos_return for w in rolling_valid) / len(rolling_valid)
                if rolling_valid else 0.0
            )
            # Worst single-window OOS return as a proxy for compounded max DD
            compounded_oos_max_dd = (
                abs(min((w.oos_return for w in rolling_valid), default=0.0))
            )

            hm = result.holdout_metrics or {}
            params = result.params

            strat = Strategy(
                symbol_id=sym_row.id,
                timeframe=config.signal_timeframe,
                param_hash=p_hash,

                # GridParams
                atr_period=params.atr_period,
                atr_multiplier=params.atr_multiplier,
                geometric_ratio=params.geometric_ratio,
                n_levels=params.n_levels,
                pullback_pct=params.pullback_pct,
                hurst_window=params.hurst_window,
                hurst_threshold=params.hurst_threshold,
                adx_period=params.adx_period,
                adx_threshold=params.adx_threshold,
                vpvr_window=params.vpvr_window,
                use_vpvr_anchor=params.use_vpvr_anchor,
                position_size_pct=params.position_size_pct,
                max_open_levels=params.max_open_levels,

                # IS / full-period metrics (from holdout, the cleanest OOS)
                sharpe=hm.get("sharpe", 0.0),
                sortino=hm.get("sortino", 0.0),
                max_drawdown=hm.get("max_drawdown", 1.0),
                calmar=hm.get("calmar", 0.0),
                return_dd_ratio=hm.get("return_dd_ratio", 0.0),
                profit_factor=hm.get("profit_factor", 0.0),
                cagr=hm.get("cagr", 0.0),
                win_rate=hm.get("win_rate", 0.0),
                n_trades=hm.get("n_trades", 0),
                max_consec_losses=hm.get("max_consec_losses", 0),
                dd_recovery_bars=hm.get("dd_recovery_bars", 0.0),

                # WFO aggregates
                avg_oos_sharpe=result.avg_oos_sharpe,
                avg_oos_sortino=result.avg_oos_sortino,
                oos_is_ratio=result.oos_is_ratio,
                consistency_score=result.consistency_score,
                plateau_width_score=result.plateau_width_score,
                compounded_oos_return=compounded_oos_return,
                compounded_oos_max_dd=compounded_oos_max_dd,

                # Monte Carlo
                mc_p_value=mc_p_value,
                mc_n_shuffles=config.mc_n_shuffles,

                # Deflated Sharpe
                deflated_sharpe=dsr,

                # Holdout metrics
                holdout_sharpe=hm.get("sharpe", 0.0),
                holdout_sortino=hm.get("sortino", 0.0),
                holdout_cagr=hm.get("cagr", 0.0),
                holdout_max_drawdown=hm.get("max_drawdown", 1.0),

                # Meta
                search_cycle=search_cycle,
            )
            session.add(strat)
            session.flush()
            new_id = strat.id

            # WFO windows
            for w in result.rolling_windows + result.anchored_windows:
                session.add(WFOWindow(
                    strategy_id=new_id,
                    wfo_type=w.wfo_type,
                    window_index=w.window_index,
                    is_start=w.is_start.to_pydatetime() if hasattr(w.is_start, "to_pydatetime") else w.is_start,
                    is_end=w.is_end.to_pydatetime() if hasattr(w.is_end, "to_pydatetime") else w.is_end,
                    oos_start=w.oos_start.to_pydatetime() if hasattr(w.oos_start, "to_pydatetime") else w.oos_start,
                    oos_end=w.oos_end.to_pydatetime() if hasattr(w.oos_end, "to_pydatetime") else w.oos_end,
                    is_sharpe=w.is_sharpe,
                    is_sortino=w.is_sortino,
                    is_return=w.is_return,
                    oos_sharpe=w.oos_sharpe,
                    oos_sortino=w.oos_sortino,
                    oos_return=w.oos_return,
                    n_trades=w.n_trades,
                ))

            session.commit()
            logger.success(
                f"Strategy #{new_id} saved — {result.symbol} "
                f"OOS Sharpe={result.avg_oos_sharpe:.2f} "
                f"DSR={dsr:.2f} MC p={mc_p_value:.3f}"
            )
            return new_id

    def get_top_strategies(
        self,
        n: int = 20,
        sort_by: str = "avg_oos_sharpe",
    ) -> list[Strategy]:
        """Return top-n strategies sorted descending by sort_by column."""
        with self._Session() as session:
            col = getattr(Strategy, sort_by, Strategy.avg_oos_sharpe)
            rows = (
                session.query(Strategy)
                .order_by(col.desc())
                .limit(n)
                .all()
            )
            # Eagerly load symbol_rel to avoid detached-instance errors
            for r in rows:
                _ = r.symbol_rel
            return rows

    def get_strategies_by_symbol(self, symbol: str) -> list[Strategy]:
        """Return all stored strategies for a given symbol."""
        with self._Session() as session:
            rows = (
                session.query(Strategy)
                .join(Symbol)
                .filter(Symbol.symbol == symbol)
                .order_by(Strategy.avg_oos_sharpe.desc())
                .all()
            )
            for r in rows:
                _ = r.symbol_rel
            return rows

    def get_strategy_by_id(self, strategy_id: int) -> Strategy | None:
        """Return a single strategy by primary key, or None."""
        with self._Session() as session:
            row = session.query(Strategy).filter_by(id=strategy_id).first()
            if row:
                _ = row.symbol_rel
            return row

    def strategy_count(self) -> int:
        """Total number of stored strategies."""
        with self._Session() as session:
            return session.query(func.count(Strategy.id)).scalar() or 0

    def get_all_params_hashes(self) -> set[str]:
        """Return all param_hashes currently in the database (for fast dedup)."""
        with self._Session() as session:
            rows = session.query(Strategy.param_hash).all()
            return {r[0] for r in rows}
