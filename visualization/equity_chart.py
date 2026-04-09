"""
visualization/equity_chart.py

Generates a 3-panel interactive Plotly chart for a stored strategy,
run against the last 60 calendar days of that symbol's data
(visualization slice — never seen during optimization or WFO).

Panel 1 — Price + Grid  (1m candlestick downsampled to 15m, grid lines,
                          regime shading, trade markers, reset events)
Panel 2 — Equity Curve  ($1,000 virtual portfolio, annotation box)
Panel 3 — Drawdown      (filled area, max-DD annotation)

Output:
    charts/{symbol}_{strategy_id}.html   — interactive (zoomable/hoverable)
    charts/{symbol}_{strategy_id}.png    — static image (requires kaleido)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from backtester.engine import Backtester
from backtester.metrics import compute_metrics
from data.cache import DataPair
from database.models import Strategy


def generate_chart(
    strategy: Strategy,
    viz_data: DataPair,
    ref_atr: float,
    output_dir: str = "charts",
) -> str:
    """
    Run a fresh backtest on viz_data (last 60 days, $1,000 capital),
    then build and save the Plotly figure.

    Returns the path to the saved HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    params = strategy.to_grid_params()

    # ── Run backtest on the visualization slice ───────────────────────────────
    bt = Backtester()
    result = bt.run(
        params=params,
        data=viz_data,
        initial_capital=1_000.0,
        ref_atr=ref_atr,
    )

    equity_curve = result.equity_curve
    trades       = result.trades
    metrics      = compute_metrics(equity_curve, trades, result.killed_early)

    df_1m = viz_data.df_1m

    # ── Downsample 1m bars to 15m for the candlestick (readability) ──────────
    df_15m = _resample_ohlcv(df_1m, "15min")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.30, 0.15],
        vertical_spacing=0.02,
        subplot_titles=[
            f"{strategy.symbol} — Price + Grid (15m candles)",
            "Equity Curve  ($1,000 virtual portfolio)",
            "Drawdown from Equity Peak",
        ],
    )

    _add_price_panel(fig, df_15m, trades, row=1)
    _add_equity_panel(fig, equity_curve, metrics, row=2)
    _add_drawdown_panel(fig, equity_curve, metrics, row=3)

    sym  = strategy.symbol
    sid  = strategy.id or "new"
    fig.update_layout(
        title=(
            f"{sym} | Strategy #{sid} | Last 60 Days | $1,000 Virtual Portfolio<br>"
            f"<sup>OOS Sharpe={strategy.avg_oos_sharpe:.2f}  "
            f"Holdout CAGR={strategy.holdout_cagr:.1%}  "
            f"DSR={strategy.deflated_sharpe:.2f}  "
            f"MC p={strategy.mc_p_value:.3f}</sup>"
        ),
        template="plotly_dark",
        hovermode="x unified",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis3=dict(rangeslider=dict(visible=False)),
    )

    html_path = f"{output_dir}/{sym}_{sid}.html"
    png_path  = f"{output_dir}/{sym}_{sid}.png"

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    fig.write_html(html_path)
    logger.info(f"Chart saved: {html_path}")

    try:
        fig.write_image(png_path)
        logger.info(f"PNG saved:   {png_path}")
    except Exception as exc:
        logger.warning(f"PNG export skipped (kaleido not available?): {exc}")

    return html_path


# ── Panel builders ─────────────────────────────────────────────────────────────

def _add_price_panel(
    fig: go.Figure,
    df_15m: pd.DataFrame,
    trades: list,
    row: int,
) -> None:
    """Candlestick + trade markers."""
    fig.add_trace(
        go.Candlestick(
            x=df_15m.index,
            open=df_15m["open"],
            high=df_15m["high"],
            low=df_15m["low"],
            close=df_15m["close"],
            name="Price (15m)",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=row, col=1,
    )

    # Trade entry / exit markers
    if trades:
        long_entries  = [t for t in trades if t.side == "long"]
        short_entries = [t for t in trades if t.side == "short"]

        for group, color, symbol_marker, label in [
            (long_entries,  "#00e676", "triangle-up",   "Long fills"),
            (short_entries, "#ff1744", "triangle-down", "Short fills"),
        ]:
            if not group:
                continue
            entry_times  = [t.entry_time for t in group]
            entry_prices = [t.entry_price for t in group]
            exit_times   = [t.exit_time  for t in group]
            exit_prices  = [t.exit_price for t in group]

            fig.add_trace(
                go.Scatter(
                    x=entry_times, y=entry_prices,
                    mode="markers",
                    marker=dict(symbol=symbol_marker, size=8, color=color),
                    name=f"{label} entry",
                    legendgroup=label,
                ),
                row=row, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=exit_times, y=exit_prices,
                    mode="markers",
                    marker=dict(symbol="x", size=6, color=color, opacity=0.7),
                    name=f"{label} exit",
                    legendgroup=label,
                    showlegend=False,
                ),
                row=row, col=1,
            )

    fig.update_yaxes(title_text="Price (USDT)", row=row, col=1)


def _add_equity_panel(
    fig: go.Figure,
    equity_curve: pd.Series,
    metrics: dict,
    row: int,
) -> None:
    """Equity curve with $1,000 baseline and stats annotation."""
    if equity_curve.empty:
        return

    x = equity_curve.index
    y = equity_curve.values
    baseline = 1_000.0

    # Green above baseline, red below
    y_above = np.where(y >= baseline, y, baseline)
    y_below = np.where(y <= baseline, y, baseline)

    fig.add_trace(
        go.Scatter(
            x=x, y=y_above,
            fill="tozeroy",
            fillcolor="rgba(38, 166, 154, 0.25)",
            line=dict(color="#26a69a", width=1.5),
            name="Equity (above $1k)",
            showlegend=False,
        ),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y_below,
            fill="tozeroy",
            fillcolor="rgba(239, 83, 80, 0.25)",
            line=dict(color="#ef5350", width=1.5),
            name="Equity (below $1k)",
            showlegend=False,
        ),
        row=row, col=1,
    )
    # Baseline
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="rgba(255,255,255,0.4)",
        annotation_text="$1,000",
        annotation_position="right",
        row=row, col=1,
    )

    # Annotation box (top-right of panel)
    final  = float(equity_curve.iloc[-1])
    pct    = (final / baseline - 1) * 100
    sign   = "+" if pct >= 0 else ""
    ann_text = (
        f"Final: ${final:,.0f}  ({sign}{pct:.1f}%)<br>"
        f"MaxDD: -{metrics.get('max_drawdown', 0)*100:.1f}%<br>"
        f"Sharpe: {metrics.get('sharpe', 0):.2f}<br>"
        f"Trades: {metrics.get('n_trades', 0)}  "
        f"Win: {metrics.get('win_rate', 0)*100:.0f}%"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        xanchor="right", yanchor="top",
        text=ann_text,
        showarrow=False,
        bordercolor="#888",
        borderwidth=1,
        bgcolor="rgba(30,30,30,0.8)",
        font=dict(size=11),
        row=row, col=1,
    )

    fig.update_yaxes(title_text="Portfolio Value (USDT)", row=row, col=1)


def _add_drawdown_panel(
    fig: go.Figure,
    equity_curve: pd.Series,
    metrics: dict,
    row: int,
) -> None:
    """Drawdown filled area with max-DD annotation."""
    if equity_curve.empty:
        return

    eq  = equity_curve.values
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / np.maximum(peak, 1e-10) * 100  # in %

    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=dd,
            fill="tozeroy",
            fillcolor="rgba(239, 83, 80, 0.40)",
            line=dict(color="#ef5350", width=1),
            name="Drawdown",
            showlegend=False,
        ),
        row=row, col=1,
    )

    max_dd_pct = metrics.get("max_drawdown", 0) * 100
    fig.add_hline(
        y=-max_dd_pct,
        line_dash="dash",
        line_color="rgba(255,100,100,0.7)",
        annotation_text=f"Max DD: -{max_dd_pct:.1f}%",
        annotation_position="right",
        row=row, col=1,
    )

    fig.update_yaxes(title_text="Drawdown (%)", row=row, col=1)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resample_ohlcv(df_1m: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate 1m bars into a coarser OHLCV DataFrame."""
    return df_1m.resample(freq).agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",    "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])
