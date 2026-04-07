"""
database/models.py

SQLAlchemy ORM models for the strategy database.

Tables:
    symbols      — tracked trading pairs with history metadata
    strategies   — validated, stored grid strategies (full metrics)
    wfo_windows  — per-window IS/OOS results for each stored strategy
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from strategy.grid_engine import GridParams


class Base(DeclarativeBase):
    pass


# ── Symbol ─────────────────────────────────────────────────────────────────────

class Symbol(Base):
    __tablename__ = "symbols"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    symbol           = Column(String, nullable=False, unique=True)
    timeframe        = Column(String, nullable=False)
    first_date       = Column(DateTime)
    last_date        = Column(DateTime)
    avg_volume_usdt  = Column(Float)

    strategies = relationship("Strategy", back_populates="symbol_rel", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Symbol {self.symbol}>"


# ── Strategy ───────────────────────────────────────────────────────────────────

class Strategy(Base):
    """
    One fully-validated grid strategy.

    All 13 GridParams are stored as individual columns.
    Full IS + holdout metrics, WFO aggregates, Monte Carlo p-value,
    and Deflated Sharpe Ratio are included.
    """
    __tablename__ = "strategies"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id   = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    timeframe   = Column(String, nullable=False)
    param_hash  = Column(String, nullable=False)

    # ── GridParams columns ──────────────────────────────────────────────────
    atr_period        = Column(Integer)
    atr_multiplier    = Column(Float)
    geometric_ratio   = Column(Float)
    n_levels          = Column(Integer)
    pullback_pct      = Column(Float)
    hurst_window      = Column(Integer)
    hurst_threshold   = Column(Float)
    adx_period        = Column(Integer)
    adx_threshold     = Column(Float)
    vpvr_window       = Column(Integer)
    use_vpvr_anchor   = Column(Boolean)
    position_size_pct = Column(Float)
    max_open_levels   = Column(Integer)

    # ── IS / full-period metrics ────────────────────────────────────────────
    sharpe           = Column(Float)
    sortino          = Column(Float)
    max_drawdown     = Column(Float)
    calmar           = Column(Float)
    return_dd_ratio  = Column(Float)
    profit_factor    = Column(Float)
    cagr             = Column(Float)
    win_rate         = Column(Float)
    n_trades         = Column(Integer)
    max_consec_losses   = Column(Integer)
    dd_recovery_bars    = Column(Float)

    # ── WFO aggregate metrics ───────────────────────────────────────────────
    avg_oos_sharpe        = Column(Float)
    avg_oos_sortino       = Column(Float)
    oos_is_ratio          = Column(Float)
    consistency_score     = Column(Float)
    plateau_width_score   = Column(Float)
    compounded_oos_return = Column(Float)
    compounded_oos_max_dd = Column(Float)

    # ── Monte Carlo ─────────────────────────────────────────────────────────
    mc_p_value   = Column(Float)
    mc_n_shuffles = Column(Integer)

    # ── Deflated Sharpe ─────────────────────────────────────────────────────
    deflated_sharpe = Column(Float)

    # ── Holdout metrics ─────────────────────────────────────────────────────
    holdout_sharpe       = Column(Float)
    holdout_sortino      = Column(Float)
    holdout_cagr         = Column(Float)
    holdout_max_drawdown = Column(Float)

    # ── Meta ─────────────────────────────────────────────────────────────────
    created_at   = Column(DateTime, default=datetime.utcnow)
    search_cycle = Column(Integer)

    __table_args__ = (
        UniqueConstraint("param_hash", name="uq_param_hash"),
    )

    symbol_rel  = relationship("Symbol", back_populates="strategies")
    wfo_windows = relationship("WFOWindow", back_populates="strategy", cascade="all, delete-orphan")

    def to_grid_params(self) -> GridParams:
        """Reconstruct GridParams from the stored columns."""
        return GridParams(
            atr_period=self.atr_period,
            atr_multiplier=self.atr_multiplier,
            geometric_ratio=self.geometric_ratio,
            n_levels=self.n_levels,
            pullback_pct=self.pullback_pct,
            hurst_window=self.hurst_window,
            hurst_threshold=self.hurst_threshold,
            adx_period=self.adx_period,
            adx_threshold=self.adx_threshold,
            vpvr_window=self.vpvr_window,
            use_vpvr_anchor=bool(self.use_vpvr_anchor),
            position_size_pct=self.position_size_pct,
            max_open_levels=self.max_open_levels,
        )

    @property
    def symbol(self) -> str:
        return self.symbol_rel.symbol if self.symbol_rel else ""

    def __repr__(self) -> str:
        return (
            f"<Strategy #{self.id} {self.symbol} "
            f"sharpe={self.avg_oos_sharpe:.2f} "
            f"cagr={self.holdout_cagr:.1%}>"
        )


# ── WFOWindow ──────────────────────────────────────────────────────────────────

class WFOWindow(Base):
    """One IS/OOS window from rolling or anchored WFO."""
    __tablename__ = "wfo_windows"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id  = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    wfo_type     = Column(String, nullable=False)   # "rolling" | "anchored"
    window_index = Column(Integer, nullable=False)

    is_start  = Column(DateTime)
    is_end    = Column(DateTime)
    oos_start = Column(DateTime)
    oos_end   = Column(DateTime)

    is_sharpe    = Column(Float)
    is_sortino   = Column(Float)
    is_return    = Column(Float)
    oos_sharpe   = Column(Float)
    oos_sortino  = Column(Float)
    oos_return   = Column(Float)
    n_trades     = Column(Integer)

    strategy = relationship("Strategy", back_populates="wfo_windows")

    def __repr__(self) -> str:
        return (
            f"<WFOWindow strategy={self.strategy_id} "
            f"{self.wfo_type}[{self.window_index}] "
            f"oos_sharpe={self.oos_sharpe:.2f}>"
        )


def init_db(engine) -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
