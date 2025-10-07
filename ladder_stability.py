
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math

import sys
import numpy as np
import pandas as pd

if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")

from ladder_backtester import LadderConfig, backtest_ladder_strategy
from ladder_optimizer import random_search_optimize, SearchBounds

# ----------------- Time splitting (rolling-origin CV) -----------------

def _rolling_origin_splits(
    df: pd.DataFrame, t1: str, t2: str,
    n_splits: int = 3,
    min_train_points: int = 120,
    min_test_points: int = 60
):
    """
    Rolling-origin CV avec contraintes minimales sur train/test.
    train = [t1 .. test_start-1], test = [test_start .. test_end]
    Retourne: liste de tuples (train_t1, train_t2, test_t1, test_t2)
    """
    sub = df.loc[t1:t2]
    if sub.empty:
        raise ValueError("No data in the given window.")
    dates = pd.Index(sub.index).unique().sort_values()
    L = len(dates)

    # Nombre max de splits possibles avec tailles minimales
    max_splits = (L - min_train_points) // max(1, min_test_points)
    if max_splits < 1:
        raise ValueError(
            f"Insufficient data for CV: need >= {min_train_points + min_test_points} points, got {L}."
        )
    n_splits = min(n_splits, int(max_splits))

    # Taille de bloc test (>= min_test_points)
    remaining = L - min_train_points
    block = max(min_test_points, remaining // n_splits)

    splits = []
    for i in range(n_splits):
        test_start_idx = min_train_points + i * block
        test_end_idx = min(L - 1, test_start_idx + block - 1)
        if test_start_idx <= 0 or test_start_idx >= L:
            continue
        train_t1 = dates[0].strftime("%Y-%m-%d")
        train_t2 = dates[test_start_idx - 1].strftime("%Y-%m-%d")
        test_t1 = dates[test_start_idx].strftime("%Y-%m-%d")
        test_t2 = dates[test_end_idx].strftime("%Y-%m-%d")
        splits.append((train_t1, train_t2, test_t1, test_t2))
    return splits


# ----------------- Jitter robustness -----------------

def _enforce_monotone_levels(levels: List[float], low: float, high: float, min_gap: float = 0.0) -> List[float]:
    lv = sorted([max(low, min(high, x)) for x in levels])
    for i in range(1, len(lv)):
        if lv[i] <= lv[i-1] + min_gap:
            lv[i] = min(high, lv[i-1] + max(min_gap, 1e-6))
    return lv

def _jitter_params(params: Dict[str, Any],
                   level_abs_jitter: float = 0.01,
                   size_rel_jitter: float = 0.10,
                   fee_bps_abs_jitter: float = 2.0,
                   slippage_bps_abs_jitter: float = 2.0,
                   bounds: Optional[SearchBounds] = None,
                   min_gap: float = 0.0,
                   ) -> Dict[str, Any]:
    """
    Apply small random perturbations to levels (absolute +-), sizes (relative +-),
    and costs (absolute +-), then re-normalize constraints.
    """
    rng = np.random.default_rng()

    buy_lv = np.array(params["buy_levels"], dtype=float)
    sell_lv = np.array(params["sell_levels"], dtype=float)
    buy_sz = np.array(params["buy_sizes"], dtype=float)
    sell_sz = np.array(params["sell_sizes"], dtype=float)

    if bounds is None:
        bounds = SearchBounds()

    # Jitter levels (absolute bumps), then enforce monotone and bounds
    buy_lv = (buy_lv + rng.normal(0.0, level_abs_jitter, size=buy_lv.shape)).tolist()
    sell_lv = (sell_lv + rng.normal(0.0, level_abs_jitter, size=sell_lv.shape)).tolist()

    buy_lv = _enforce_monotone_levels(buy_lv, bounds.buy_level_low, bounds.buy_level_high, min_gap=min_gap)
    sell_lv = _enforce_monotone_levels(sell_lv, bounds.sell_level_low, bounds.sell_level_high, min_gap=min_gap)

    # Jitter sizes (relative bumps), then clip and renormalize to cap
    def jitter_sizes(sz, cap):
        if sz.sum() <= 0:
            return np.zeros_like(sz)
        mult = np.exp(rng.normal(0.0, size_rel_jitter, size=sz.shape))  # lognormal-ish
        new = sz * mult
        new = np.clip(new, 0.0, None)
        s = new.sum()
        if s > 0:
            new = new * (cap / s)
        return new

    buy_sz = jitter_sizes(buy_sz, bounds.buy_total_cap)
    sell_sz = jitter_sizes(sell_sz, bounds.sell_total_cap)

    fee_bps = params.get("fee_bps", 2.0) + float(rng.normal(0.0, fee_bps_abs_jitter))
    slippage_bps = params.get("slippage_bps", 0.0) + float(rng.normal(0.0, slippage_bps_abs_jitter))

    fee_bps = float(np.clip(fee_bps, bounds.fee_bps_low, bounds.fee_bps_high))
    slippage_bps = float(np.clip(slippage_bps, bounds.slippage_bps_low, bounds.slippage_bps_high))

    return {
        "buy_levels": buy_lv,
        "buy_sizes": buy_sz.tolist(),
        "sell_levels": sell_lv,
        "sell_sizes": sell_sz.tolist(),
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps
    }


def _metrics_from_backtest(res: Dict[str, Any]) -> Dict[str, float]:
    met = {"CAGR": np.nan, "MaxDrawdown": np.nan, "Calmar": np.nan, "Sharpe": np.nan}
    try:
        m = res.get("metrics", None)
        if m is not None and not m.empty:
            row = m.iloc[0]
            for k in met.keys():
                if k in row:
                    met[k] = float(row[k])
    except Exception:
        pass
    return met


def _config_from_params(params: Dict[str, Any],
                        reference_mode: str,
                        rolling_window: Optional[int],
                        max_position_pct: float) -> LadderConfig:
    return LadderConfig(
        buy_levels=params["buy_levels"],
        buy_sizes=params["buy_sizes"],
        sell_levels=params["sell_levels"],
        sell_sizes=params["sell_sizes"],
        reference_mode=reference_mode,
        rolling_window=rolling_window,
        max_position_pct=max_position_pct,
        fee_bps=params.get("fee_bps", 2.0),
        slippage_bps=params.get("slippage_bps", 0.0),
        allow_multiple_triggers_per_day=True
    )


# ----------------- Param similarity (stability across folds) -----------------

def _flatten_params(params: Dict[str, Any]) -> np.ndarray:
    vec = np.array(params["buy_levels"] + params["buy_sizes"] + params["sell_levels"] + params["sell_sizes"], dtype=float)
    # normalize to unit norm to compare by cosine
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

def _param_similarity(params_list: List[Dict[str, Any]]) -> float:
    if len(params_list) < 2:
        return np.nan
    vecs = [_flatten_params(p) for p in params_list]
    sims = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            a, b = vecs[i], vecs[j]
            num = float(np.dot(a, b))
            den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            sims.append(num / den if den > 0 else np.nan)
    return float(np.nanmean(sims)) if sims else np.nan


# ----------------- Main: walk-forward + jitter summary by N -----------------

def summarize_with_stability(
    df: pd.DataFrame,
    t1: str,
    t2: str,
    initial_cash: float,
    levels_list: List[int] = [3,4,5,6],
    objective: str = "calmar",
    iterations: int = 400,
    seed: int = 123,
    n_splits: int = 3,
    jitters: int = 50,
    level_abs_jitter: float = 0.01,
    size_rel_jitter: float = 0.10,
    reference_mode: str = "trailing_max",
    rolling_window: Optional[int] = None,
    max_position_pct: float = 1.0,
    rf_annual: float = 0.0,
    bounds: Optional[SearchBounds] = None,
    min_gap: float = 0.0,
) -> pd.DataFrame:
    """
    For each N in levels_list:
      - Do rolling-origin CV with n_splits test windows.
      - For each split: optimize on train, evaluate on test.
      - Aggregate IS (train best) and OOS (test with best params) metrics.
      - Run jitter robustness on OOS best params to estimate sensitivity.
      - Compute a StabilityScore from OOS/IS ratio, jitter std, and param similarity across folds.
    Returns a DataFrame with one row per N.
    """
    splits = _rolling_origin_splits(df, t1, t2, n_splits=n_splits)
    if not splits:
        raise ValueError("Insufficient data to create CV splits.")

    rows = []
    for n in levels_list:
        train_metrics = []
        test_metrics = []
        best_params_per_fold = []

        for k, (tr1, tr2, te1, te2) in enumerate(splits):
            opt = random_search_optimize(
                df=df, t1=tr1, t2=tr2,
                initial_cash=initial_cash,
                n_buy=n, n_sell=n,
                reference_mode=reference_mode,
                rolling_window=rolling_window,
                max_position_pct=max_position_pct,
                rf_annual=rf_annual,
                objective=objective,
                iterations=iterations,
                seed=seed + 100*n + k,
                bounds=bounds,
                allow_multiple_triggers_per_day=True,
                vary_costs=False
            )
            # best on train
            train_best = opt.history.iloc[0] if not opt.history.empty else None
            if train_best is not None:
                train_metrics.append({
                    "CAGR": float(train_best.get("CAGR", np.nan)),
                    "MaxDrawdown": float(abs(train_best.get("MaxDrawdown", np.nan))),
                    "Calmar": float(train_best.get("Calmar", np.nan)),
                    "Sharpe": float(train_best.get("Sharpe", np.nan))
                })

            best_params = opt.best_params
            best_params_per_fold.append(best_params)

            # evaluate on test
            cfg = _config_from_params(best_params, reference_mode, rolling_window, max_position_pct)
            res_test = backtest_ladder_strategy(
                df=df, t1=te1, t2=te2, initial_cash=initial_cash,
                config=cfg, ref_init=None, rf_annual=rf_annual,
                plot_price=False, plot_equity=False,
                record_trades=False, record_journal=False, compute_metrics=True
            )
            test_metrics.append(_metrics_from_backtest(res_test))

        # Aggregate IS/OOS
        def agg(lst, key):
            vals = [d.get(key, np.nan) for d in lst if d is not None]
            return float(np.nanmean(vals)) if vals else np.nan

        IS_CAGR = agg(train_metrics, "CAGR")
        IS_MDD = agg(train_metrics, "MaxDrawdown")
        IS_Calmar = agg(train_metrics, "Calmar")
        IS_Sharpe = agg(train_metrics, "Sharpe")

        OOS_CAGR = agg(test_metrics, "CAGR")
        OOS_MDD = agg(test_metrics, "MaxDrawdown")
        OOS_Calmar = agg(test_metrics, "Calmar")
        OOS_Sharpe = agg(test_metrics, "Sharpe")

        # Ratios OOS/IS (robustness to overfitting)
        def ratio(o, i):
            if i is None or np.isnan(i) or i == 0:
                return np.nan
            return float(o / i)

        R_Calmar = ratio(OOS_Calmar, IS_Calmar)
        R_CAGR = ratio(OOS_CAGR, IS_CAGR)
        R_Sharpe = ratio(OOS_Sharpe, IS_Sharpe)

        # Param stability across folds
        param_sim = _param_similarity(best_params_per_fold)

        # Jitter robustness on the last fold's best params (as proxy)
        jitter_scores = []
        if best_params_per_fold:
            base = best_params_per_fold[-1]
            for j in range(jitters):
                jit = _jitter_params(base,
                                     level_abs_jitter=level_abs_jitter,
                                     size_rel_jitter=size_rel_jitter,
                                     bounds=bounds,
                                     min_gap=min_gap)
                cfg_j = _config_from_params(jit, reference_mode, rolling_window, max_position_pct)
                res_j = backtest_ladder_strategy(
                    df=df, t1=te1, t2=te2, initial_cash=initial_cash,
                    config=cfg_j, ref_init=None, rf_annual=rf_annual,
                    plot_price=False, plot_equity=False,
                    record_trades=False, record_journal=False, compute_metrics=True
                )
                met_j = _metrics_from_backtest(res_j)
                jitter_scores.append(met_j.get("Calmar", np.nan))

        JitterCalmar_Mean = float(np.nanmean(jitter_scores)) if jitter_scores else np.nan
        JitterCalmar_Std = float(np.nanstd(jitter_scores)) if jitter_scores else np.nan

        # Simple StabilityScore (0..1 approx): combine OOS/IS ratio, low jitter std, param similarity
        # Clip ratio to [0,1.5] then /1.5 to map roughly to [0..1].
        def clip01(x, hi=1.5):
            if x is None or np.isnan(x):
                return np.nan
            return max(0.0, min(1.0, x / hi))

        s1 = clip01(R_Calmar)
        s2 = 1.0 / (1.0 + max(0.0, JitterCalmar_Std))  # smaller std -> closer to 1
        s3 = 0.0 if (param_sim is None or np.isnan(param_sim)) else max(0.0, min(1.0, (param_sim + 1)/2))  # cosine [-1,1] -> [0,1]

        # Weighted sum (tunable): emphasize OOS ratio and jitter stability
        weights = (0.5, 0.3, 0.2)
        StabilityScore = np.nan
        if not np.isnan(s1) and not np.isnan(s2) and not np.isnan(s3):
            StabilityScore = float(weights[0]*s1 + weights[1]*s2 + weights[2]*s3)

        # Pick one representative best_params (last fold) to display
        disp = best_params_per_fold[-1] if best_params_per_fold else {"buy_levels": [], "buy_sizes": [], "sell_levels": [], "sell_sizes": [], "fee_bps": 2.0, "slippage_bps": 0.0}

        rows.append({
            "NLevels": n,
            "IS_CAGR": IS_CAGR,
            "IS_MDD": IS_MDD,
            "IS_Calmar": IS_Calmar,
            "IS_Sharpe": IS_Sharpe,
            "OOS_CAGR": OOS_CAGR,
            "OOS_MDD": OOS_MDD,
            "OOS_Calmar": OOS_Calmar,
            "OOS_Sharpe": OOS_Sharpe,
            "R_Calmar": R_Calmar,
            "R_CAGR": R_CAGR,
            "R_Sharpe": R_Sharpe,
            "JitterCalmar_Mean": JitterCalmar_Mean,
            "JitterCalmar_Std": JitterCalmar_Std,
            "ParamSimilarity": param_sim,
            "StabilityScore": StabilityScore,
            "BuyLevels": "[" + ", ".join(f"{x:.3f}" for x in disp["buy_levels"]) + "]",
            "BuySizes": "[" + ", ".join(f"{x:.3f}" for x in disp["buy_sizes"]) + "]",
            "SellLevels": "[" + ", ".join(f"{x:.3f}" for x in disp["sell_levels"]) + "]",
            "SellSizes": "[" + ", ".join(f"{x:.3f}" for x in disp["sell_sizes"]) + "]",
            "fee_bps": disp.get("fee_bps", 2.0),
            "slippage_bps": disp.get("slippage_bps", 0.0)
        })

    df_summary = pd.DataFrame(rows).sort_values(["StabilityScore","OOS_Calmar","OOS_Sharpe"], ascending=False).reset_index(drop=True)
    return df_summary
