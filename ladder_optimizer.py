
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import sys

# Ensure we can import the user's backtester
if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")

from ladder_backtester import (
    LadderConfig,
    backtest_ladder_strategy,
    make_synthetic_price_series,
)

__all__ = [
    "SearchBounds",
    "OptimizationResult",
    "compute_objective",
    "random_search_optimize",
    "_sorted_levels_from_raw",
    "_sizes_from_raw",
]


# -------------------------- Utilities --------------------------

def _sorted_levels_from_raw(raw: np.ndarray, low: float, high: float) -> List[float]:
    """
    Map an unconstrained vector 'raw' (length n) to strictly increasing levels in (low, high).
    Uses softplus to ensure positive deltas, then normalizes cumulative sums into (low, high).
    """
    n = len(raw)
    if n <= 0:
        return []
    # softplus to get positive deltas
    deltas = np.log1p(np.exp(raw))  # >0
    cum = np.cumsum(deltas)
    eps = 1e-6
    # normalize cum to (low, high)
    scaled = low + (high - low) * (cum - cum.min() + eps) / (cum.max() - cum.min() + 2 * eps)
    # Ensure strict monotonicity numerically
    for i in range(1, len(scaled)):
        if scaled[i] <= scaled[i-1]:
            scaled[i] = min(high, scaled[i-1] + 1e-9)
    return scaled.tolist()


def _sizes_from_raw(raw: np.ndarray, total_cap: float) -> List[float]:
    """
    Map an unconstrained vector to non-negative sizes that sum to <= total_cap (<=1).
    Uses softmax * total_cap.
    """
    if len(raw) == 0:
        return []
    x = raw - raw.max()  # stability
    w = np.exp(x)
    probs = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    return (total_cap * probs).tolist()


# -------------------------- Objective helpers --------------------------

def compute_objective(metrics_df: pd.DataFrame, objective: str, risk_pref: float = 1.0) -> float:
    """
    Higher is better.
    objective in {"calmar", "sharpe", "cagr", "custom"}
    For "custom": score = CAGR - risk_pref * |MaxDrawdown|
    NOTE: 'pnl' / 'total_return' are handled directly in random_search_optimize (need equity series).
    """
    if metrics_df is None or metrics_df.empty:
        return -1e9
    row = metrics_df.iloc[0]
    if objective == "calmar":
        val = row.get("Calmar", np.nan)
    elif objective == "sharpe":
        val = row.get("Sharpe", np.nan)
    elif objective == "cagr":
        val = row.get("CAGR", np.nan)
    elif objective == "custom":
        cagr = row.get("CAGR", np.nan)
        mdd = abs(row.get("MaxDrawdown", np.nan))
        if np.isnan(cagr) or np.isnan(mdd):
            return -1e9
        val = cagr - risk_pref * mdd
    else:
        raise ValueError("Unknown objective")
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return -1e9
    return float(val)


@dataclass
class SearchBounds:
    buy_level_low: float = 0.01
    buy_level_high: float = 0.60
    sell_level_low: float = 0.01
    sell_level_high: float = 0.80
    buy_total_cap: float = 0.95   # total cash fraction deployable across all buy steps
    sell_total_cap: float = 0.95  # total position fraction reducible across all sell steps
    fee_bps_low: float = 0.0
    fee_bps_high: float = 10.0
    slippage_bps_low: float = 0.0
    slippage_bps_high: float = 10.0


@dataclass
class OptimizationResult:
    best_score: float
    best_params: Dict[str, Any]
    leaderboard: pd.DataFrame
    history: pd.DataFrame


def random_search_optimize(
    df: pd.DataFrame,
    t1: str,
    t2: str,
    initial_cash: float,
    n_buy: int = 3,
    n_sell: int = 3,
    reference_mode: str = "trailing_max",
    rolling_window: Optional[int] = None,
    max_position_pct: float = 1.0,
    rf_annual: float = 0.0,
    objective: str = "calmar",
    risk_pref: float = 1.0,
    iterations: int = 500,
    seed: Optional[int] = 123,
    bounds: Optional[SearchBounds] = None,
    allow_multiple_triggers_per_day: bool = True,
    vary_costs: bool = False,
) -> OptimizationResult:
    """
    Randomized search over ladder parameters subject to natural constraints.
    - Levels strictly increasing within their [low, high] intervals.
    - Sizes non-negative; sum to <= total_cap (buy_total_cap / sell_total_cap).
    - Optionally vary fee/slippage within bounds; otherwise keep defaults from backtester.

    Supported objectives:
      • "calmar" | "sharpe" | "cagr" | "custom" (via compute_objective)
      • "pnl" | "equity" | "final_equity"  -> maximize final equity (net of costs)
      • "total_return" | "totret"          -> maximize total return over [t1..t2]

    Returns an OptimizationResult with best score/params and a full history DataFrame.
    """
    if bounds is None:
        bounds = SearchBounds()

    rng = np.random.default_rng(seed)

    history_rows: List[Dict[str, Any]] = []
    best_score = -1e18
    best_row: Optional[Dict[str, Any]] = None

    for i in range(iterations):
        # Sample unconstrained latent vectors
        raw_buy_levels = rng.normal(size=n_buy)
        raw_sell_levels = rng.normal(size=n_sell)
        raw_buy_sizes = rng.normal(size=n_buy)
        raw_sell_sizes = rng.normal(size=n_sell)

        buy_levels = _sorted_levels_from_raw(raw_buy_levels, bounds.buy_level_low, bounds.buy_level_high)
        sell_levels = _sorted_levels_from_raw(raw_sell_levels, bounds.sell_level_low, bounds.sell_level_high)

        buy_sizes = _sizes_from_raw(raw_buy_sizes, bounds.buy_total_cap)
        sell_sizes = _sizes_from_raw(raw_sell_sizes, bounds.sell_total_cap)

        # Optionally vary costs
        if vary_costs:
            fee_bps = float(rng.uniform(bounds.fee_bps_low, bounds.fee_bps_high))
            slippage_bps = float(rng.uniform(bounds.slippage_bps_low, bounds.slippage_bps_high))
        else:
            fee_bps = 2.0
            slippage_bps = 0.0

        config = LadderConfig(
            buy_levels=buy_levels,
            buy_sizes=buy_sizes,
            sell_levels=sell_levels,
            sell_sizes=sell_sizes,
            reference_mode=reference_mode,
            rolling_window=rolling_window,
            max_position_pct=max_position_pct,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            allow_multiple_triggers_per_day=allow_multiple_triggers_per_day
        )

        metrics_row: Dict[str, Any] = {}
        try:
            res = backtest_ladder_strategy(
                df=df, t1=t1, t2=t2, initial_cash=initial_cash, config=config,
                ref_init=None, rf_annual=rf_annual,
                plot_price=False, plot_equity=False,
                record_trades=False, record_journal=False, compute_metrics=True
            )

            # Handle PnL-like objectives that need the equity curve
            obj = objective.lower()
            if obj in {"pnl", "equity", "final_equity"}:
                eq = res.get("equity", None)
                if eq is None or len(eq) == 0:
                    score = -1e9
                    final_equity = np.nan
                else:
                    final_equity = float(eq.iloc[-1])
                    score = final_equity
                metrics_row = res["metrics"].iloc[0].to_dict() if res.get("metrics", None) is not None else {}
                metrics_row["FinalEquity"] = final_equity

            elif obj in {"total_return", "totret"}:
                eq = res.get("equity", None)
                if eq is None or len(eq) < 2:
                    score = -1e9
                    total_return = np.nan
                else:
                    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
                    score = total_return
                metrics_row = res["metrics"].iloc[0].to_dict() if res.get("metrics", None) is not None else {}
                metrics_row["TotalReturn"] = total_return

            else:
                score = compute_objective(res["metrics"], objective=obj, risk_pref=risk_pref)
                metrics_row = res["metrics"].iloc[0].to_dict() if res.get("metrics", None) is not None else {}

        except Exception as e:
            score = -1e9
            metrics_row = {"Error": str(e)}

        row = {
            "iter": i + 1,
            "score": score,
            "buy_levels": buy_levels,
            "buy_sizes": buy_sizes,
            "sell_levels": sell_levels,
            "sell_sizes": sell_sizes,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            **metrics_row
        }
        history_rows.append(row)

        if score > best_score:
            best_score = score
            best_row = row

    history = pd.DataFrame(history_rows).sort_values("score", ascending=False).reset_index(drop=True)

    # Build a compact leaderboard with top-k rows expanded into strings for readability
    k = min(25, len(history))
    top = history.head(k).copy()

    def fmt_list(lst):
        try:
            return "[" + ", ".join(f"{float(x):.3f}" for x in lst) + "]"
        except Exception:
            return str(lst)

    for col in ("buy_levels", "buy_sizes", "sell_levels", "sell_sizes"):
        if col in top.columns:
            top[col] = top[col].apply(fmt_list)

    # Pack best params
    if best_row is None:
        best_params = {
            "buy_levels": [], "buy_sizes": [], "sell_levels": [], "sell_sizes": [],
            "fee_bps": 2.0, "slippage_bps": 0.0
        }
    else:
        best_params = {k: v for k, v in best_row.items()
                       if k in {"buy_levels","buy_sizes","sell_levels","sell_sizes","fee_bps","slippage_bps"}}

    return OptimizationResult(
        best_score=float(best_score),
        best_params=best_params,
        leaderboard=top,
        history=history
    )


# --------------- Optional quick demo (synthetic series) ---------------

def _demo():
    df_demo = make_synthetic_price_series(start="2022-01-01", end="2025-06-30", seed=7, drift=0.08, vol=0.55)
    # Example: maximize PnL (final equity)
    opt = random_search_optimize(
        df=df_demo,
        t1="2022-01-10",
        t2="2025-06-30",
        initial_cash=100_000.0,
        n_buy=3, n_sell=3,
        reference_mode="trailing_max",
        max_position_pct=1.0,
        objective="pnl",             # <--- new objective
        iterations=200,
        seed=1234
    )
    # Save artifacts
    leaderboard_path = "/mnt/data/ladder_param_leaderboard.csv"
    opt.leaderboard.to_csv(leaderboard_path, index=False)
    best_json_path = "/mnt/data/best_ladder_params.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump({"best_score": opt.best_score, "best_params": opt.best_params}, f, ensure_ascii=False, indent=2)
    return best_json_path, leaderboard_path, opt.best_params

if __name__ == "__main__":
    print(_demo())
