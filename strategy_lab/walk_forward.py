"""
Walk-forward validation and a single-use final holdout check.

Discipline this file enforces:
  * The last `holdout_days` are never touched by fold generation. Look at them
    once, after the hyperparameters are locked.
  * Each fold starts flat and resets strategy state, so one fold cannot
    inherit a position (or a warm HMM fit) from the previous one.
  * Configurations are ranked on `pooled_score`: the daily PL series from
    every fold concatenated, then scored once. A per-fold score is a Sharpe
    estimate on ~50 observations and is extremely noisy (fold-to-fold standard
    deviation of ~200 points on this data), so ranking on the mean or median
    of fold scores mostly ranks luck. Pooling uses every observation in one
    estimate and is also what eval.py actually measures - a single continuous
    run - so it is both more stable and better aligned with the target.
    Per-fold statistics are still reported, to show *consistency*.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from The_Limitless_Liquidity_Providers import PortfolioConfig, PortfolioEngine  # noqa: E402
from strategy_lab.backtester import BacktestResult, compute_score, run_backtest  # noqa: E402


@dataclass
class Fold:
    test_start: int
    test_end: int

    def __repr__(self) -> str:
        return f"Fold({self.test_start}:{self.test_end})"


def generate_folds(
    n_days: int,
    min_history_days: int,
    test_size: int = 60,
    step: Optional[int] = None,
    holdout_days: int = 150,
) -> List[Fold]:
    """Non-overlapping test windows, with the tail reserved as holdout.

    `min_history_days` is not a training set - these strategies refit from
    their trailing window on every call. It is simply the amount of history
    that must exist before the first scored day, so no fold is scored on the
    strategy's cold start.
    """
    step = step or test_size
    usable_end = n_days - holdout_days
    folds: List[Fold] = []
    test_start = min_history_days
    while test_start + test_size <= usable_end:
        folds.append(Fold(test_start, test_start + test_size))
        test_start += step
    return folds


def make_position_function(config: PortfolioConfig) -> Tuple[Callable, PortfolioEngine]:
    """Fresh engine bound to `config`, plus the function the backtester calls."""
    engine = PortfolioEngine(config)
    return engine.target_positions, engine


def evaluate_config_on_folds(
    prices_by_day: np.ndarray,
    config: PortfolioConfig,
    folds: List[Fold],
) -> List[BacktestResult]:
    """Run one configuration across every fold, resetting state between folds."""
    position_function, engine = make_position_function(config)
    results = []
    for fold in folds:
        engine.reset()
        results.append(
            run_backtest(prices_by_day, position_function, fold.test_start, fold.test_end)
        )
    return results


def aggregate_fold_results(results: List[BacktestResult]) -> Dict[str, float]:
    """Collapse per-fold results into the row used to rank configurations."""
    scores = np.array([r.score for r in results])
    sharpes = np.array([r.sharpe for r in results])

    pooled_pl = np.concatenate([r.daily_pl for r in results]) if results else np.zeros(1)
    pooled = compute_score(pooled_pl)

    return {
        "pooled_score": pooled["score"],
        "pooled_sharpe": pooled["sharpe"],
        "pooled_mean_pl": pooled["mean_pl"],
        "median_fold_score": float(np.median(scores)),
        "std_fold_score": float(np.std(scores)),
        "worst_fold_score": float(np.min(scores)),
        "fraction_profitable_folds": float(np.mean(scores > 0)),
        "median_fold_sharpe": float(np.median(sharpes)),
        "worst_drawdown": float(np.min([r.worst_drawdown for r in results])),
        "mean_commission": float(np.mean([r.mean_daily_commission for r in results])),
        "mean_gross_exposure": float(np.mean([r.mean_gross_dollar_exposure for r in results])),
        "mean_net_exposure": float(np.mean([r.mean_net_dollar_exposure for r in results])),
        "mean_turnover": float(np.mean([r.mean_daily_turnover for r in results])),
    }


def walk_forward_validate(
    prices_by_day: np.ndarray,
    configs: Dict[str, PortfolioConfig],
    folds: List[Fold],
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate several named configurations across the same folds.

    Returns a DataFrame indexed by config name, sorted by pooled score.
    """
    rows = {}
    for i, (name, config) in enumerate(configs.items(), start=1):
        if verbose:
            print(f"  [{i}/{len(configs)}] {name}", flush=True)
        results = evaluate_config_on_folds(prices_by_day, config, folds)
        rows[name] = aggregate_fold_results(results)
    frame = pd.DataFrame(rows).T
    return frame.sort_values("pooled_score", ascending=False)


def run_holdout_check(
    prices_by_day: np.ndarray,
    config: PortfolioConfig,
    holdout_days: int = 150,
) -> BacktestResult:
    """Single-use check on the reserved tail.

    Run this ONCE, after the hyperparameters are locked from the folds. If you
    run it repeatedly and pick the best, it stops being a holdout and becomes
    just another (very small) validation set.
    """
    position_function, engine = make_position_function(config)
    engine.reset()
    # -1 for the warm-up day, so `holdout_days` days actually get scored.
    start = prices_by_day.shape[0] - holdout_days - 1
    return run_backtest(prices_by_day, position_function, start, prices_by_day.shape[0])


def run_evaluation_window(
    prices_by_day: np.ndarray,
    config: PortfolioConfig,
    n_test_days: int = 250,
) -> BacktestResult:
    """Reproduce exactly what `python eval.py` will print.

    eval.py scores the final `numTestDays` days of whatever prices.txt it is
    given, as one continuous run. Use this as the last sanity check before
    submitting - it agrees with eval.py to the cent.
    """
    position_function, engine = make_position_function(config)
    engine.reset()
    # eval.py's startDay = nt - numTestDays is its warm-up day; the scored
    # window is the numTestDays days after it.
    start = prices_by_day.shape[0] - n_test_days - 1
    return run_backtest(prices_by_day, position_function, start, prices_by_day.shape[0])
