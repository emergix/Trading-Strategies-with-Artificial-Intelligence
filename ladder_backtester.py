"""
Ladder-style daily backtester for buy/sell scale strategies.
Author: ChatGPT

Usage:
    - Provide a pandas DataFrame with columns ["Date", "Close"] (Date convertible to datetime).
    - Configure LadderConfig with buy/sell levels and sizes.
    - Call backtest_ladder_strategy(df, t1, t2, initial_cash, config, ref_init=None, rf_annual=0.0)
    - Returns dict with equity/cash/position/reference series, trades DataFrame, metrics DataFrame.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any, Tuple

# ---------- Utilities ----------

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df indexed by DatetimeIndex and contains 'Close'."""
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df = df.set_index("Date")
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        raise ValueError("DataFrame must have a 'Date' column or a DatetimeIndex.")
    if "Close" not in df.columns:
        raise ValueError("DataFrame must have a 'Close' column (daily close price).")
    return df


def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """Compute Max Drawdown and return (mdd_pct, peak_date, trough_date)."""
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    trough_idx = dd.idxmin()
    mdd = dd.loc[trough_idx]
    peak_idx = cummax.loc[:trough_idx].idxmax()
    return float(mdd), peak_idx, trough_idx


def cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate based on first/last values and elapsed years."""
    start_val = equity.iloc[0]
    end_val = equity.iloc[-1]
    num_days = (equity.index[-1] - equity.index[0]).days
    if num_days <= 0 or start_val <= 0:
        return np.nan
    years = num_days / 365.25
    return (end_val / start_val) ** (1.0 / years) - 1.0


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio for daily returns series; rf is annual risk-free rate."""
    if returns.std(ddof=0) == 0 or returns.dropna().empty:
        return np.nan
    mean_ann = returns.mean() * periods_per_year
    std_ann = returns.std(ddof=0) * math.sqrt(periods_per_year)
    excess = mean_ann - rf
    return excess / std_ann if std_ann > 0 else np.nan


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio using downside deviation only."""
    downside = returns.clip(upper=0.0)
    downside_std = downside.std(ddof=0) * math.sqrt(periods_per_year)
    if downside_std == 0 or returns.dropna().empty:
        return np.nan
    mean_ann = returns.mean() * periods_per_year
    excess = mean_ann - rf
    return excess / downside_std if downside_std > 0 else np.nan


def calmar_ratio(equity: pd.Series) -> float:
    """Calmar = CAGR / |MaxDD|."""
    mdd, _, _ = max_drawdown(equity)
    mdd_abs = abs(mdd)
    if mdd_abs == 0:
        return np.nan
    return cagr(equity) / mdd_abs


# ---------- Strategy Config ----------

@dataclass
class LadderConfig:
    # Buy ladder: price drop relative to reference (peak or moving anchor)
    buy_levels: List[float]            # e.g., [0.05, 0.10, 0.20] for -5%, -10%, -20%
    buy_sizes: List[float]             # fraction of AVAILABLE CASH to deploy at each triggered level
    # Sell ladder: price rise relative to reference
    sell_levels: List[float]           # e.g., [0.10, 0.20, 0.35] for +10%, +20%, +35%
    sell_sizes: List[float]            # fraction of CURRENT POSITION to sell at each triggered level
    # Reference update logic
    reference_mode: Literal["peak", "trailing_max", "rolling_mean", "anchored"]
    rolling_window: Optional[int] = None   # used if reference_mode == "rolling_mean"
    # Risk / sizing
    max_position_pct: float = 1.0      # cap max position as fraction of equity (1.0 = 100%)
    # Costs
    fee_bps: float = 2.0               # broker fee in bps per transaction
    slippage_bps: float = 0.0          # slippage model (linear, per side)
    # Execution
    allow_multiple_triggers_per_day: bool = True


# ---------- Backtester ----------

def backtest_ladder_strategy(
    df: pd.DataFrame,
    t1: str,
    t2: str,
    initial_cash: float,
    config: LadderConfig,
    ref_init: Optional[float] = None,
    rf_annual: float = 0.0,
) -> Dict[str, Any]:
    """
    Backtest a daily ladder-style buy/sell strategy between t1 and t2.
    df: DataFrame with ['Date','Close'] or DatetimeIndex + 'Close'.
    t1,t2: date strings or pd.Timestamp.
    initial_cash: starting cash.
    config: LadderConfig.
    ref_init: initial reference price (if None, uses first close in window).
    rf_annual: annual risk-free used for Sharpe/Sortino.
    """
    px = ensure_datetime_index(df)[["Close"]].copy()
    px = px.loc[(px.index >= pd.to_datetime(t1)) & (px.index <= pd.to_datetime(t2))].copy()
    if px.empty:
        raise ValueError("No data in the selected [t1, t2] window.")

    # Reference initialization
    if config.reference_mode in ("peak", "trailing_max"):
        ref = ref_init if ref_init is not None else float(px["Close"].iloc[0])
    elif config.reference_mode == "rolling_mean":
        if not config.rolling_window:
            raise ValueError("rolling_window must be set when reference_mode='rolling_mean'.")
        ref = float(px["Close"].iloc[0])
    elif config.reference_mode == "anchored":
        ref = ref_init if ref_init is not None else float(px["Close"].iloc[0])
    else:
        raise ValueError("Invalid reference_mode.")

    # State
    cash = float(initial_cash)
    units = 0.0
    equity_curve = []
    position_values = []
    cash_series = []
    ref_series = []
    trades: List[Dict[str, Any]] = []

    # Sort ladders: buys from largest drop to smallest, sells from smallest rise to largest
    buy_levels = sorted(config.buy_levels, reverse=True)
    buy_sizes = [s for _, s in sorted(zip(config.buy_levels, config.buy_sizes), reverse=True)]
    sell_levels = sorted(config.sell_levels)
    sell_sizes = [s for _, s in sorted(zip(config.sell_levels, config.sell_sizes))]

    for date, row in px.iterrows():
        price = float(row["Close"])

        # Update reference
        if config.reference_mode in ("peak", "trailing_max"):
            ref = max(ref, price)
        elif config.reference_mode == "rolling_mean":
            start_idx = max(0, px.index.get_loc(date) - config.rolling_window + 1)
            ref = float(px["Close"].iloc[start_idx: px.index.get_loc(date) + 1].mean())
        elif config.reference_mode == "anchored":
            pass  # keep initial anchor unless changed externally

        # --- BUY ladder ---
        for lvl, frac_cash in zip(buy_levels, buy_sizes):
            threshold = ref * (1.0 - float(lvl))
            if price <= threshold and cash > 0:
                equity = cash + units * price
                target_position_value = config.max_position_pct * equity
                current_position_value = units * price
                buy_value = min(frac_cash * cash, max(0.0, target_position_value - current_position_value))
                if buy_value > 0:
                    fee = buy_value * (config.fee_bps / 1e4)
                    slip = buy_value * (config.slippage_bps / 1e4)
                    effective_value = buy_value - fee - slip
                    delta_units = effective_value / price
                    units += delta_units
                    cash -= buy_value
                    trades.append({
                        "Date": date, "Side": "BUY", "Price": price, "Ref": ref,
                        "Level": -float(lvl), "CashUsed": buy_value, "UnitsDelta": delta_units
                    })
                    if not config.allow_multiple_triggers_per_day:
                        break

        # --- SELL ladder ---
        for lvl, frac_pos in zip(sell_levels, sell_sizes):
            threshold = ref * (1.0 + float(lvl))
            if price >= threshold and units > 0:
                sell_units = frac_pos * units
                sell_value = sell_units * price
                fee = sell_value * (config.fee_bps / 1e4)
                slip = sell_value * (config.slippage_bps / 1e4)
                net_value = sell_value - fee - slip
                units -= sell_units
                cash += net_value
                trades.append({
                    "Date": date, "Side": "SELL", "Price": price, "Ref": ref,
                    "Level": +float(lvl), "CashReceived": net_value, "UnitsDelta": -sell_units
                })
                if not config.allow_multiple_triggers_per_day:
                    break

        equity = cash + units * price
        equity_curve.append((date, equity))
        position_values.append((date, units * price))
        cash_series.append((date, cash))
        ref_series.append((date, ref))

    equity_series = pd.Series(data=[v for _, v in equity_curve],
                              index=[d for d, _ in equity_curve], name="Equity")
    pos_series = pd.Series(data=[v for _, v in position_values],
                           index=[d for d, _ in position_values], name="PositionValue")
    cash_series = pd.Series(data=[v for _, v in cash_series],
                            index=[d for d, _ in cash_series], name="Cash")
    ref_series = pd.Series(data=[v for _, v in ref_series],
                           index=[d for d, _ in ref_series], name="Reference")

    rets = equity_series.pct_change().fillna(0.0)

    mdd, peak_date, trough_date = max_drawdown(equity_series)
    def _calmar(eq): 
        try: 
            return calmar_ratio(eq)
        except Exception:
            return np.nan

    metrics = {
        "StartDate": equity_series.index[0],
        "EndDate": equity_series.index[-1],
        "StartEquity": float(equity_series.iloc[0]),
        "EndEquity": float(equity_series.iloc[-1]),
        "PnL": float(equity_series.iloc[-1] - equity_series.iloc[0]),
        "CAGR": float(cagr(equity_series)),
        "Sharpe": float(sharpe_ratio(rets, rf=rf_annual)),
        "Sortino": float(sortino_ratio(rets, rf=rf_annual)),
        "MaxDrawdown": float(mdd),
        "MDD_PeakDate": peak_date,
        "MDD_TroughDate": trough_date,
        "Calmar": float(_calmar(equity_series)),
        "Trades": len(trades),
        "FinalCash": float(cash_series.iloc[-1]),
        "FinalUnitsValue": float(pos_series.iloc[-1]),
        "FinalUnits": float((pos_series.iloc[-1] / px["Close"].iloc[-1]) if px["Close"].iloc[-1] != 0 else 0.0),
        "ExposurePctAvg": float((pos_series / equity_series).replace([np.inf, -np.inf], np.nan).fillna(0).mean())
    }

    trades_df = pd.DataFrame(trades)
    metrics_df = pd.DataFrame([metrics])

    # Plots (optional)
    plt.figure(figsize=(12, 4))
    plt.plot(px.index, px["Close"])
    plt.title("Underlying Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(equity_series.index, equity_series.values)
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()

    return {
        "equity": equity_series,
        "cash": cash_series,
        "position_value": pos_series,
        "reference": ref_series,
        "trades": trades_df,
        "metrics": metrics_df,
    }


# ---------- Optional: synthetic data generator for quick tests ----------

def make_synthetic_price_series(
    start: str = "2022-01-01",
    end: str = "2025-01-01",
    seed: int = 42,
    drift: float = 0.12,
    vol: float = 0.60
) -> pd.DataFrame:
    """Create a GBM-like synthetic series for quick demos."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="B")
    n = len(dates)
    dt = 1/252
    shocks = rng.normal((drift - 0.5*vol**2)*dt, vol*np.sqrt(dt), size=n)
    log_price = np.cumsum(shocks)
    price = 100.0 * np.exp(log_price)
    return pd.DataFrame({"Date": dates, "Close": price})


if __name__ == "__main__":
    # Minimal self-test
    df_demo = make_synthetic_price_series()
    from dataclasses import dataclass
    config_demo = LadderConfig(
        buy_levels=[0.05, 0.10, 0.20],
        buy_sizes=[0.10, 0.15, 0.25],
        sell_levels=[0.10, 0.20, 0.35],
        sell_sizes=[0.10, 0.15, 0.25],
        reference_mode="trailing_max",
        rolling_window=None,
        max_position_pct=1.00,
        fee_bps=2.0,
        slippage_bps=0.0,
        allow_multiple_triggers_per_day=True
    )
    results = backtest_ladder_strategy(
        df=df_demo,
        t1="2022-01-10",
        t2="2024-12-31",
        initial_cash=100_000.0,
        config=config_demo,
        ref_init=None,
        rf_annual=0.0
    )
    print(results["metrics"])
