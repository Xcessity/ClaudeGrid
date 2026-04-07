from dataclasses import dataclass


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
    market_slippage_base: float = 0.0003   # 3 bps baseline
    market_slippage_max: float = 0.0030    # 30 bps ceiling
    market_slippage_atr_cap: float = 3.0   # if current_atr >= 3 × ref_atr → use max slippage

    # Quality thresholds for DB storage
    min_sharpe: float = 1.0
    min_sortino: float = 1.2
    min_return_dd_ratio: float = 2.0
    min_profit_factor: float = 1.3
    min_cagr: float = 0.15
    min_oos_consistency: float = 0.60   # ≥60% of OOS windows profitable
    min_wfo_oos_is_ratio: float = 0.50  # OOS perf ≥ 50% of IS perf

    # WFO
    wfo_is_bars: int = 1000
    wfo_oos_bars: int = 300
    wfo_step_bars: int = 150
    wfo_min_windows: int = 4

    # Dual-timeframe
    signal_timeframe: str = "4h"
    fill_timeframe: str = "1m"

    # Bollinger Squeeze position size cap
    bb_period: int = 20
    bb_squeeze_pct_threshold: float = 0.10

    # Funding rates
    simulate_funding: bool = True
    funding_interval_hours: int = 8

    # Monte Carlo significance test
    mc_n_shuffles: int = 200
    mc_significance: float = 0.05

    # Delta neutrality
    max_delta_pct: float = 0.05

    # Capital-level kill switch
    max_account_dd_pct: float = 0.25

    # Holdout quarantine (multiple-testing bias prevention)
    holdout_max_failures: int = 3
    holdout_quarantine_days: int = 30

    # Optimization
    n_optuna_trials: int = 500

    # Search
    min_trades_per_backtest: int = 30
    min_trades_per_wfo_window: int = 8
    n_workers: int = -1              # joblib: -1 = all CPUs

    # Visualization
    viz_lookback_days: int = 60


config = Config()
