"""
Hyperparameter sweeps.

WHAT TO VARY, AND WHY
---------------------
The original sweep varied `vol_span` alone. That is the least important knob
in the whole strategy: it only rescales returns before the factor fit, and the
results across spans 2..30 differed by less than the fold-to-fold noise. The
parameters that actually move the score, roughly in order of impact:

  1. sizing.signal_gain           - how much of the position limit gets used.
                                    The score is LINEAR in mean(PL) and the
                                    Sharpe term is scale-invariant, so unused
                                    limit is forgone score. This single knob
                                    was worth more than everything else
                                    combined.
  2. stat_arb.reversion_window    - the horizon of the reversion being traded.
  3. stat_arb.n_factors           - how much systematic risk is stripped out.
                                    Interacts strongly with (2): more factors
                                    -> shorter usable reversion window.
  4. stat_arb.zscore_method       - cross-sectional vs per-name OU
                                    standardisation.
  5. stat_arb.factor_model        - plain PCA vs ALGO-as-factor-1 + PCA.
  6. stat_arb.factor_fit_window   - loading estimation window.
  7. stat_arb.max_half_life /
     require_stationary_residual  - how aggressively residuals are screened.
  8. hedge.enabled / target_hedge_ratio
  9. sizing.rebalance_threshold_dollars - commission control.
 10. stat_arb.volatility_span     - genuinely minor; sweep it last.

Sweeps are run COORDINATE-WISE (one axis at a time, from the current best
config) rather than as a full grid. A full grid over 10 axes is both
computationally silly and a great way to overfit 9 folds; a coordinate sweep
costs O(sum of axis lengths) instead of O(product) and makes it obvious which
axes the score is actually sensitive to.

USAGE
-----
    python -m strategy_lab.hyperparameter_sweep              # the standard sweep
    python -m strategy_lab.hyperparameter_sweep --axis n_factors
    python -m strategy_lab.hyperparameter_sweep --holdout    # locked config only
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from The_Limitless_Liquidity_Providers import PortfolioConfig  # noqa: E402
from strategy_lab.data_loading import load_prices_by_day  # noqa: E402
from strategy_lab.walk_forward import (  # noqa: E402
    generate_folds,
    run_evaluation_window,
    run_holdout_check,
    walk_forward_validate,
)


def config_with(base: PortfolioConfig, dotted_path: str, value: Any) -> PortfolioConfig:
    """Deep-copy `base` and set one nested field, e.g. 'stat_arb.n_factors'."""
    updated = copy.deepcopy(base)
    section_name, field_name = dotted_path.split(".")
    section = getattr(updated, section_name)
    if not hasattr(section, field_name):
        raise AttributeError(f"{section_name} has no field {field_name!r}")
    setattr(section, field_name, value)
    return updated


# The axes swept by default, in the priority order argued for above.
SWEEP_AXES: Dict[str, List[Any]] = {
    "sizing.signal_gain": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0],
    "stat_arb.reversion_window": [5, 10, 15, 20, 25, 30, 40, 60],
    "stat_arb.n_factors": [1, 3, 5, 10, 15, 20, 25, 30],
    "mean_reversion.lookback_window": [2, 3, 5, 7, 10, 20, 40],
    "mean_reversion.entry_zscore": [0.0, 0.5, 1.0, 1.5],
    "mean_reversion.beta_hedge_against_algo": [True, False],
    "stat_arb.zscore_method": ["cross_sectional", "ou"],
    "stat_arb.factor_model": ["pca", "algo_plus_pca"],
    "stat_arb.factor_fit_window": [126, 189, 252, 378],
    "hedge.enabled": [False, True],
    "hedge.target_hedge_ratio": [0.5, 0.8, 1.0],
    "momentum.min_variance_ratio": [1.0, 1.05, 1.1, 1.2],
    "momentum.lookback_windows": [(20, 60), (60, 120), (120, 250)],
    "momentum.trend_gate_enabled": [True, False],
    "stat_arb.max_half_life": [10, 15, 25, 40, 1000],
    "stat_arb.require_stationary_residual": [False, True],
    "sizing.rebalance_threshold_dollars": [0.0, 250.0, 500.0, 1000.0],
    "sizing.position_smoothing_days": [1.0, 2.0, 3.0],
    "stat_arb.volatility_span": [10, 20, 40, 60, 90],
}

# Ensemble weights are swept separately: they are a simplex, not independent
# axes, so a coordinate sweep over three numbers that get normalised anyway
# would mostly re-discover the same ratio. Sweeping the ratio directly is both
# cheaper and easier to read.
ENSEMBLE_WEIGHT_GRID: List[Dict[str, float]] = [
    {"stat_arb": 1.0, "mean_reversion": 0.0, "momentum": 0.0},
    {"stat_arb": 0.0, "mean_reversion": 1.0, "momentum": 0.0},
    {"stat_arb": 0.0, "mean_reversion": 0.0, "momentum": 1.0},
    {"stat_arb": 1.0, "mean_reversion": 0.25, "momentum": 0.0},
    {"stat_arb": 1.0, "mean_reversion": 0.5, "momentum": 0.0},
    {"stat_arb": 1.0, "mean_reversion": 1.0, "momentum": 0.0},
    {"stat_arb": 0.5, "mean_reversion": 1.0, "momentum": 0.0},
    {"stat_arb": 1.0, "mean_reversion": 0.5, "momentum": 0.05},
    {"stat_arb": 1.0, "mean_reversion": 0.5, "momentum": 0.10},
    {"stat_arb": 1.0, "mean_reversion": 0.5, "momentum": 0.25},
    {"stat_arb": 1.0, "mean_reversion": 1.0, "momentum": 1.0},
]


def sweep_ensemble_weights(
    prices_by_day: np.ndarray, base_config: PortfolioConfig, folds, verbose: bool = True
) -> pd.DataFrame:
    """Walk-forward every weight combination in ENSEMBLE_WEIGHT_GRID."""
    configs = {}
    for weights in ENSEMBLE_WEIGHT_GRID:
        label = ":".join(
            f"{weights[k]:g}" for k in ("stat_arb", "mean_reversion", "momentum")
        )
        updated = copy.deepcopy(base_config)
        updated.ensemble.baseline_weights = dict(weights)
        configs[f"SA:MR:MO = {label}"] = updated
    if verbose:
        print(f"\n=== ensemble weights ({len(configs)} combinations x {len(folds)} folds) ===")
    return walk_forward_validate(prices_by_day, configs, folds, verbose=verbose)


def sweep_one_axis(
    prices_by_day: np.ndarray,
    base_config: PortfolioConfig,
    dotted_path: str,
    values: List[Any],
    folds,
    verbose: bool = True,
) -> pd.DataFrame:
    """Walk-forward every value of a single hyperparameter."""
    configs = {
        f"{dotted_path.split('.')[-1]}={value}": config_with(base_config, dotted_path, value)
        for value in values
    }
    if verbose:
        print(f"\n=== {dotted_path} ({len(values)} values x {len(folds)} folds) ===")
    return walk_forward_validate(prices_by_day, configs, folds, verbose=verbose)


def coordinate_sweep(
    prices_by_day: np.ndarray,
    base_config: PortfolioConfig,
    axes: Dict[str, List[Any]],
    folds,
    adopt_best: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Sweep each axis in turn, optionally carrying the winner forward.

    With `adopt_best=True` this is a greedy coordinate ascent: each axis is
    swept against the best configuration found so far. That captures the
    important interactions (n_factors x reversion_window especially) at a
    fraction of the cost of a grid, but it is greedy - if two axes interact
    strongly, run the second pass by simply calling this again with the
    returned config.
    """
    current = copy.deepcopy(base_config)
    tables: Dict[str, pd.DataFrame] = {}

    report_columns = [
        "pooled_score", "pooled_sharpe", "pooled_mean_pl", "median_fold_score",
        "std_fold_score", "worst_fold_score", "fraction_profitable_folds", "mean_commission",
    ]

    for dotted_path, values in axes.items():
        started = time.time()
        table = sweep_one_axis(prices_by_day, current, dotted_path, values, folds, verbose)
        tables[dotted_path] = table
        if verbose:
            print(table[report_columns].round(2).to_string())
            print(f"  ({time.time() - started:.1f}s)")

        if not adopt_best:
            continue

        spread = table["pooled_score"].max() - table["pooled_score"].min()
        if spread < 1e-6:
            # Every value scored identically - this axis is inert given the
            # rest of the config (e.g. the half-life screen does nothing while
            # zscore_method='cross_sectional'). Adopting an arbitrary winner
            # here would be pure noise-fitting, so keep the base value.
            if verbose:
                print(f"  -> {dotted_path} has no effect on this config; keeping current value")
            continue

        best_label = table.index[0]
        best_value = values[
            [f"{dotted_path.split('.')[-1]}={v}" for v in values].index(best_label)
        ]
        current = config_with(current, dotted_path, best_value)
        if verbose:
            print(f"  -> adopting {dotted_path} = {best_value!r}  (spread {spread:.1f})")

    tables["__final_config__"] = current  # type: ignore[assignment]
    return tables


def describe_config(config: PortfolioConfig) -> str:
    lines = ["Locked configuration:"]
    for section_name in ("stat_arb", "sizing", "hedge", "ensemble"):
        section = getattr(config, section_name)
        lines.append(f"  {section_name}:")
        for field_name, value in vars(section).items():
            lines.append(f"    {field_name} = {value!r}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", default=None, help="sweep a single dotted axis only")
    parser.add_argument("--test-size", type=int, default=60, help="days per fold")
    parser.add_argument(
        "--step",
        type=int,
        default=25,
        help="days between fold starts; < test-size gives overlapping folds, which "
        "are not independent but give a far better-conditioned pooled estimate",
    )
    parser.add_argument("--holdout-days", type=int, default=150)
    parser.add_argument("--no-adopt", action="store_true", help="sweep each axis against the base config")
    parser.add_argument("--holdout", action="store_true", help="run the single-use holdout check and stop")
    parser.add_argument("--weights", action="store_true", help="sweep ensemble weights only")
    parser.add_argument(
        "--test-days",
        type=int,
        default=500,
        help="numTestDays for the eval.py reproduction (the competition scores 500)",
    )
    args = parser.parse_args()

    prices_by_day, instrument_names = load_prices_by_day()
    n_days, n_instruments = prices_by_day.shape
    print(f"Prices: {n_days} days x {n_instruments} instruments (days down the rows)")
    print(f"Instrument 0 = {instrument_names[0]}")

    base_config = PortfolioConfig()

    if args.holdout:
        print("\n" + describe_config(base_config))
        result = run_holdout_check(prices_by_day, base_config, args.holdout_days)
        print(f"\n--- HOLDOUT ({args.holdout_days} days, single use) ---")
        for key, value in result.summary_row().items():
            print(f"  {key:22s} {value:>12.2f}")
        evaluation = run_evaluation_window(prices_by_day, base_config, args.test_days)
        print(f"\n--- eval.py window (last {args.test_days} days) ---")
        for key, value in evaluation.summary_row().items():
            print(f"  {key:22s} {value:>12.2f}")
        return

    # Folds must start late enough that no fold is scored on the strategy's
    # cold start, but a fold start below `minimum_history_days` would simply be
    # flat. 280 gives the factor model a usable window without eating the data.
    folds = generate_folds(
        n_days=n_days,
        min_history_days=max(base_config.minimum_history_days(), 280),
        test_size=args.test_size,
        step=args.step,
        holdout_days=args.holdout_days,
    )
    print(f"{len(folds)} folds, last {args.holdout_days} days held out")

    if args.weights:
        table = sweep_ensemble_weights(prices_by_day, base_config, folds)
        print(table.round(2).to_string())
        return

    if args.axis:
        table = sweep_one_axis(
            prices_by_day, base_config, args.axis, SWEEP_AXES[args.axis], folds
        )
        print(table.round(3).to_string())
        return

    tables = coordinate_sweep(
        prices_by_day, base_config, SWEEP_AXES, folds, adopt_best=not args.no_adopt
    )
    print("\n" + "=" * 78)
    print(describe_config(tables["__final_config__"]))  # type: ignore[arg-type]
    print("\nNow run:  python -m strategy_lab.hyperparameter_sweep --holdout")


if __name__ == "__main__":
    main()
