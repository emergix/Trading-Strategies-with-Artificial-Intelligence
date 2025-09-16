import pandas as pd
from ladder_backtester import LadderConfig, backtest_ladder_strategy, make_synthetic_price_series

# 1) Charger tes données avec colonnes ['Date','Close']
# df = pd.read_sql(... )  # ou pd.read_csv(...)

# Démo synthétique :
df = make_synthetic_price_series()

# 2) Paramétrer la stratégie
config = LadderConfig(
    buy_levels=[0.05, 0.10, 0.20],
    buy_sizes=[0.10, 0.15, 0.25],      # % du cash par palier
    sell_levels=[0.10, 0.20, 0.35],
    sell_sizes=[0.10, 0.15, 0.25],     # % de la position par palier
    reference_mode="trailing_max",     # 'peak' | 'trailing_max' | 'rolling_mean' | 'anchored'
    rolling_window=None,
    max_position_pct=1.0,
    fee_bps=2.0,
    slippage_bps=0.0,
    allow_multiple_triggers_per_day=True
)

# 3) Backtest
res = backtest_ladder_strategy(
    df=df,
    t1="2022-01-10",
    t2="2024-12-31",
    initial_cash=100_000,
    config=config,
    ref_init=None,
    rf_annual=0.0
)

# Résultats
metrics = res["metrics"]
trades  = res["trades"]
equity  = res["equity"]
