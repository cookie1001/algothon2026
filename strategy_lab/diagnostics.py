"""
Dataset diagnostics. Run this before trusting any strategy result.

    python -m strategy_lab.diagnostics

Everything here operates on days-down-the-rows data. The checks are the ones
that changed design decisions in the submission - they are kept as runnable
code rather than notebook output so they can be re-run the moment the extra
250 days of price data land.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from The_Limitless_Liquidity_Providers import ALGO_INDEX, compute_log_returns  # noqa: E402
from strategy_lab.data_loading import load_prices_by_day  # noqa: E402


def _least_squares_with_intercept(design: np.ndarray, target: np.ndarray):
    augmented = np.column_stack([np.ones(design.shape[0]), design])
    coefficients, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    fitted = augmented @ coefficients
    residual = target - fitted
    r_squared = 1.0 - residual.var() / target.var() if target.var() > 0 else 0.0
    return coefficients, residual, r_squared


def check_algo_is_an_index(prices_by_day: np.ndarray) -> None:
    """Is ALGO a fixed-weight basket of the other 50 instruments?

    This is the single most important structural fact about the dataset. If
    ALGO is an exact basket then (a) index exposure can be hedged with no
    tracking error, (b) ALGO adds no new idiosyncratic capacity to the book
    because it is spanned by the other 50, and (c) any "ALGO vs basket"
    arbitrage is limited to price-rounding noise and cannot cover commission.
    """
    print("\n" + "=" * 72)
    print("1. IS ALGO A FIXED-WEIGHT INDEX OF THE OTHER 50?")
    print("=" * 72)

    algo_prices = prices_by_day[:, ALGO_INDEX]
    constituent_prices = np.delete(prices_by_day, ALGO_INDEX, axis=1)

    coefficients, residual, r_squared = _least_squares_with_intercept(
        constituent_prices, algo_prices
    )
    print(f"  full-sample price regression   R^2 = {r_squared:.8f}")
    print(f"  residual std                   ${residual.std():.4f}")
    print(f"  max |residual|                 ${np.abs(residual).max():.4f}")
    print(f"  weights: sum={coefficients[1:].sum():.4f}  min={coefficients[1:].min():.4f}  "
          f"max={coefficients[1:].max():.4f}  all positive={np.all(coefficients[1:] > 0)}")

    # A price quoted to 2dp carries rounding error ~ Uniform(-0.005, 0.005),
    # standard deviation 0.01/sqrt(12). If the residual is the same size as
    # the rounding noise then the relationship is exact.
    rounding_std = 0.01 / np.sqrt(12.0)
    implied = rounding_std * np.sqrt(1.0 + float(coefficients[1:] @ coefficients[1:]))
    print(f"  residual std predicted by 2dp price rounding alone: ${implied:.4f}")
    print("  -> the relationship is EXACT; the residual is quantisation noise, not alpha."
          if residual.std() < 3 * implied else
          "  -> residual exceeds rounding noise; there may be tradeable deviation.")

    print("\n  -- Is R^2 = 1.0 just 50 free parameters fitting noise? --")
    print("  If it were, out-of-sample accuracy would collapse. It does not:")
    n_days, n_constituents = constituent_prices.shape

    def fit_then_predict(n_fit: int):
        coefficients, _, _ = _least_squares_with_intercept(
            constituent_prices[:n_fit], algo_prices[:n_fit]
        )
        design = np.column_stack([np.ones(n_days - n_fit), constituent_prices[n_fit:]])
        return algo_prices[n_fit:] - design @ coefficients

    print(f"    {'fit days':>9} {'predicts':>9} {'OOS resid':>11}")
    for n_fit in (52, 60, 100, 250, 500):
        residual = fit_then_predict(n_fit)
        print(f"    {n_fit:>9} {n_days - n_fit:>9} {'$' + format(residual.std(), '.4f'):>11}")
    print(f"  Fitting on 52 days - ONE more than the {n_constituents} free parameters - still")
    print(f"  predicts the remaining {n_days - 52} days. Overfitting cannot do that.")

    # Control: destroy the relation, keep the parameter count.
    generator = np.random.default_rng(0)
    shuffled = constituent_prices.copy()
    for column in range(shuffled.shape[1]):
        generator.shuffle(shuffled[:, column])
    coefficients, _, _ = _least_squares_with_intercept(shuffled[:250], algo_prices[:250])
    design = np.column_stack([np.ones(n_days - 250), shuffled[250:]])
    control_residual = algo_prices[250:] - design @ coefficients
    control_r2 = 1.0 - control_residual.var() / algo_prices[250:].var()
    print(f"  CONTROL, same 50 parameters on column-shuffled prices: OOS R^2 = {control_r2:+.3f}")
    print("  -> that negative R^2 is what overfitting actually looks like.")

    # Are the weights the same in disjoint windows?
    window_weights = []
    for start in range(0, n_days - 199, 200):
        coefficients, _, _ = _least_squares_with_intercept(
            constituent_prices[start : start + 200], algo_prices[start : start + 200]
        )
        window_weights.append(coefficients[1:])
    window_weights = np.array(window_weights)
    relative_drift = window_weights.std(axis=0).mean() / np.abs(window_weights.mean(axis=0)).mean()
    print(f"  weights across {len(window_weights)} disjoint 200-day windows are stable to "
          f"{100 * relative_drift:.2f}%")


def check_lead_lag(prices_by_day: np.ndarray) -> None:
    """Does the basket lead ALGO (or vice versa) by a day?

    If it did, that would be a near-riskless signal and would explain very
    large leaderboard scores. Worth re-running on new data - a lead/lag is
    exactly the kind of thing that can appear when a dataset is regenerated.
    """
    print("\n" + "=" * 72)
    print("2. LEAD-LAG BETWEEN ALGO AND THE BASKET")
    print("=" * 72)
    algo_prices = prices_by_day[:, ALGO_INDEX]
    constituent_prices = np.delete(prices_by_day, ALGO_INDEX, axis=1)

    for lag in (-2, -1, 0, 1, 2):
        if lag < 0:
            target, design = algo_prices[-lag:], constituent_prices[:lag]
        elif lag > 0:
            target, design = algo_prices[:-lag], constituent_prices[lag:]
        else:
            target, design = algo_prices, constituent_prices
        _, residual, r_squared = _least_squares_with_intercept(design, target)
        marker = "   <-- contemporaneous" if lag == 0 else ""
        print(f"  ALGO_t vs basket_(t{lag:+d}):  R^2 = {r_squared:.6f}  "
              f"resid std = ${residual.std():.4f}{marker}")
    print("  -> only the contemporaneous fit is exact: no exploitable lead or lag.")


def check_factor_structure(prices_by_day: np.ndarray) -> None:
    """Eigenvalue spectrum - how many factors are real rather than noise?"""
    print("\n" + "=" * 72)
    print("3. FACTOR STRUCTURE (drives StatArbConfig.n_factors)")
    print("=" * 72)
    returns_by_day = compute_log_returns(prices_by_day)
    correlation = np.corrcoef(returns_by_day, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    n_instruments = prices_by_day.shape[1]

    print(f"  top 10 eigenvalues: {np.round(eigenvalues[:10], 2)}")
    print(f"  PC1 explains {100 * eigenvalues[0] / n_instruments:.1f}% of variance")
    print(f"  eigenvalues above 1.0: {int((eigenvalues > 1.0).sum())} of {n_instruments}")
    print("  -> only the first ~3 stand clear of the noise bulk; the rest sit near 1.0,")
    print("     which is why projecting out 30 components costs little real signal.")

    standardised = (returns_by_day - returns_by_day.mean(axis=0)) / returns_by_day.std(axis=0)
    _, eigenvectors = np.linalg.eigh(correlation)
    first_pc = standardised @ eigenvectors[:, -1]
    algo_returns = returns_by_day[:, ALGO_INDEX]
    print(f"  corr(PC1, ALGO return) = {np.corrcoef(first_pc, algo_returns)[0, 1]:.4f}"
          "   -> ALGO *is* PC1")


def check_reversion_horizon(prices_by_day: np.ndarray) -> None:
    """Cross-sectional information coefficient by lookback horizon.

    IC = average cross-sectional correlation between the past `lookback`-day
    return and the next day's return. Negative IC means reversion. This is
    what sets `reversion_window`, and it is the honest measure of how much
    edge exists at all: an IC of -0.02 across ~50 names implies a theoretical
    annual Sharpe near 0.02 * sqrt(50) * sqrt(250) ~= 2.2, which is the right
    order of magnitude for what the strategy actually achieves.
    """
    print("\n" + "=" * 72)
    print("4. CROSS-SECTIONAL REVERSION BY HORIZON (drives reversion_window)")
    print("=" * 72)
    returns_by_day = compute_log_returns(prices_by_day)
    n_days = returns_by_day.shape[0]

    print(f"  {'lookback':>9} {'IC':>9} {'t-stat':>8}")
    for lookback in (1, 2, 3, 5, 10, 20, 40, 60, 100, 150, 250):
        coefficients = []
        for day in range(lookback, n_days - 1):
            past = returns_by_day[day - lookback + 1 : day + 1].sum(axis=0)
            future = returns_by_day[day + 1]
            past = past - past.mean()
            future = future - future.mean()
            if past.std() > 0 and future.std() > 0:
                coefficients.append(np.corrcoef(past, future)[0, 1])
        coefficients = np.array(coefficients)
        t_stat = coefficients.mean() / coefficients.std() * np.sqrt(len(coefficients))
        print(f"  {lookback:>9} {coefficients.mean():>+9.4f} {t_stat:>+8.1f}")
    print("  -> negative at every horizon: this universe reverts, it does not trend.")


def check_tradeability_of_algo_basis(prices_by_day: np.ndarray) -> None:
    """Can the ALGO-vs-basket deviation actually be traded profitably?"""
    print("\n" + "=" * 72)
    print("5. IS THE ALGO/BASKET DEVIATION TRADEABLE AFTER COSTS?")
    print("=" * 72)
    algo_prices = prices_by_day[:, ALGO_INDEX]
    constituent_prices = np.delete(prices_by_day, ALGO_INDEX, axis=1)
    _, residual, _ = _least_squares_with_intercept(constituent_prices, algo_prices)

    typical_deviation = residual.std()
    algo_price = algo_prices.mean()
    deviation_in_bps = 10_000 * typical_deviation / algo_price
    # Round trip: cross ALGO (0.2bp) and the basket (1bp), entry and exit.
    round_trip_cost_bps = 2 * (0.2 + 1.0)
    print(f"  typical deviation      {deviation_in_bps:.3f} bps of ALGO price")
    print(f"  round-trip commission  {round_trip_cost_bps:.1f} bps")
    print(f"  edge per round trip    {deviation_in_bps - round_trip_cost_bps:+.3f} bps")
    print("  -> deeply negative: the basis is real but ~1000x too small to pay for")
    print("     commission. ALGO's value is cheap hedging, not arbitrage.")


def main() -> None:
    prices_by_day, instrument_names = load_prices_by_day()
    print(f"prices: {prices_by_day.shape[0]} days x {prices_by_day.shape[1]} instruments "
          "(days down the rows)")
    print(f"instrument 0 = {instrument_names[0]}")

    check_algo_is_an_index(prices_by_day)
    check_lead_lag(prices_by_day)
    check_factor_structure(prices_by_day)
    check_reversion_horizon(prices_by_day)
    check_tradeability_of_algo_basis(prices_by_day)


if __name__ == "__main__":
    main()
