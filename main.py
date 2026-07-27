"""
main.py - Algothon strategy entry point

ONLY getMyPosition() is called by the evaluator. Keep it thin: it should just
gather signals, combine them, size the resulting positions, and return.
All actual logic lives in the helper functions below (or split these into
separate modules like signals.py / regime.py / risk.py and import them here
if your submission format allows multiple files - check the rules/starter repo).

ASSUMPTIONS TO VERIFY once you have the real task sheet / data:
  - Input shape to getMyPosition(): assumed (nInst, nDays), i.e. rows = instruments,
    columns = days (this matched the 2024 starter code convention).
  - Which column index is ALGO: placeholder ALGO_IDX = 0, change once known.
  - Return type: assumed integer number of units/shares per instrument.
  - Position limits are enforced by the evaluator anyway, but we clip here too
    as a safety net so our own backtests are representative.
"""

import numpy as np

N_INST = 51
ALGO_IDX = 0  # PLACEHOLDER - update once you know which column is ALGO


def compute_position_limits(n_inst, algo_idx=ALGO_IDX, algo_limit=100_000, other_limit=10_000):
    limits = np.full(n_inst, float(other_limit))
    limits[algo_idx] = float(algo_limit)
    return limits


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def zscore_mean_reversion_signal(prices, lookback=20):
    """
    Per-instrument mean-reversion signal: negative z-score of the latest price
    vs. its rolling mean, so the signal is positive (buy) when price is below
    its recent mean and negative (sell) when above.
    """
    window = prices[:, -lookback:]
    mu = window.mean(axis=1)
    sigma = window.std(axis=1)
    sigma[sigma == 0] = 1e-9
    z = (prices[:, -1] - mu) / sigma
    return -z


def regime_dampening(prices, lookback=60):
    """
    Very simple regime detector: lag-1 autocorrelation of returns over a
    rolling window. Negative autocorrelation -> mean-reverting regime (trust
    the signal fully). Positive autocorrelation -> trending regime (dampen
    the mean-reversion signal, since it will fight the trend).

    This is a starter heuristic - worth upgrading to a Hurst exponent, a
    variance-ratio test, or an HMM once you have real data to tune against.
    """
    n_inst = prices.shape[0]
    window = prices[:, -(lookback + 1):]
    rets = np.diff(window, axis=1)
    dampening = np.ones(n_inst)
    for i in range(n_inst):
        r = rets[i]
        if len(r) > 2 and np.std(r[:-1]) > 0 and np.std(r[1:]) > 0:
            ac = np.corrcoef(r[:-1], r[1:])[0, 1]
        else:
            ac = 0.0
        # only dampen for positive autocorrelation (trending); leave
        # mean-reverting/neutral regimes untouched
        dampening[i] = np.clip(1 - max(ac, 0.0) * 3, 0.1, 1.0)
    return dampening


# ---------------------------------------------------------------------------
# Position sizing / risk
# ---------------------------------------------------------------------------

def size_positions(signal, dampening, prices_today, position_limits, capital_alloc_frac=0.5):
    """
    Turns a raw signal into target dollar positions, then converts to units.
    Signal is clipped to +/-3 std and normalized to [-1, 1] to avoid huge
    outlier bets, then scaled by regime dampening and a fraction of each
    instrument's dollar position limit.
    """
    sig_clipped = np.clip(signal, -3, 3) / 3.0
    dollar_alloc = capital_alloc_frac * position_limits
    dollar_pos = sig_clipped * dampening * dollar_alloc
    return dollar_pos / prices_today


def clip_to_limits(units, prices_today, position_limits):
    dollar_pos = units * prices_today
    over = np.abs(dollar_pos) > position_limits
    if np.any(over):
        units = units.copy()
        units[over] = np.sign(units[over]) * (position_limits[over] / prices_today[over])
    return units


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def getMyPosition(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    n_inst, n_days = prcSoFar.shape

    lookback_needed = 61  # regime_dampening needs 60 days of returns
    if n_days < lookback_needed:
        return np.zeros(n_inst, dtype=int)

    position_limits = compute_position_limits(n_inst)

    signal = zscore_mean_reversion_signal(prcSoFar, lookback=20)
    dampening = regime_dampening(prcSoFar, lookback=60)

    units = size_positions(signal, dampening, prcSoFar[:, -1], position_limits)
    units = clip_to_limits(units, prcSoFar[:, -1], position_limits)

    return np.floor(units).astype(int)
