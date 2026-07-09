"""
backtester.py - Algothon backtester matching the scoring rules you were given.

  Commission:      0.2 bps on traded notional for ALGO, 1 bps for everything else
  Position limits: $100,000 for ALGO, $10,000 for everything else
  Score:           mean(PL) * SR^2/(SR^2+1)   if mean(PL) > 0
                    mean(PL)                   if mean(PL) <= 0

ASSUMPTIONS TO VERIFY against the real task sheet once you have it:
  - Sharpe ratio: assumed mean(dailyPL)/std(dailyPL), annualized by sqrt(252).
    Toggle `annualize` in score_from_pl() if the real rules use a different
    convention (e.g. un-annualized, or a different trading-day count).
  - Price array orientation: (nInst, nDays), matching what getMyPosition expects.
  - Commission is charged on traded notional (|position change| * price).
  - Positions returned by getMyPosition are in units/shares, limits are in dollars.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data - use this NOW, before real data is released, to validate
# the harness and sanity-check strategy behaviour under known conditions.
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_inst=51, n_days=750, seed=0):
    """
    Builds a mix of trending (GBM), mean-reverting (OU), and regime-switching
    price series, so you can check your strategy actually reacts the way you
    expect to each kind of behaviour before real data exists.
    """
    rng = np.random.default_rng(seed)
    prices = np.zeros((n_inst, n_days))

    for i in range(n_inst):
        kind = rng.choice(["gbm", "meanrev", "regime"])
        p0 = rng.uniform(20, 200)

        if kind == "gbm":
            rets = rng.normal(0.0002, 0.01, n_days)
            prices[i] = p0 * np.exp(np.cumsum(rets))

        elif kind == "meanrev":
            theta, mu, sigma = 0.05, p0, p0 * 0.02
            p = p0
            path = np.empty(n_days)
            for t in range(n_days):
                p += theta * (mu - p) + rng.normal(0, sigma)
                path[t] = p
            prices[i] = path

        else:  # regime-switching: alternating trending / mean-reverting blocks
            path = np.empty(n_days)
            path[0] = p0
            block = rng.integers(60, 150)
            trending = rng.random() < 0.5
            for t in range(1, n_days):
                if t % block == 0:
                    trending = not trending
                if trending:
                    path[t] = path[t - 1] * (1 + rng.normal(0.0005, 0.01))
                else:
                    path[t] = path[t - 1] + 0.05 * (p0 - path[t - 1]) + rng.normal(0, p0 * 0.01)
            prices[i] = path

    return prices


# ---------------------------------------------------------------------------
# Rules configuration
# ---------------------------------------------------------------------------

def compute_commission_rates(n_inst, algo_idx=0, algo_bps=0.2, other_bps=1.0):
    rates = np.full(n_inst, other_bps / 10_000)
    rates[algo_idx] = algo_bps / 10_000
    return rates


def compute_position_limits(n_inst, algo_idx=0, algo_limit=100_000, other_limit=10_000):
    limits = np.full(n_inst, float(other_limit))
    limits[algo_idx] = float(algo_limit)
    return limits


def score_from_pl(daily_pl, annualize=True, periods_per_year=252):
    mean_pl = daily_pl.mean()
    std_pl = daily_pl.std(ddof=0)
    sr = 0.0 if std_pl == 0 else mean_pl / std_pl
    if annualize and std_pl != 0:
        sr *= np.sqrt(periods_per_year)

    score = mean_pl * (sr ** 2 / (sr ** 2 + 1)) if mean_pl > 0 else mean_pl

    return {"mean_pl": mean_pl, "std_pl": std_pl, "sharpe": sr, "score": score}


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_backtest(prices, get_position_fn, algo_idx=0, start_day=61, verbose=False):
    """
    prices: (nInst, nDays) array
    get_position_fn: your getMyPosition(prcSoFar) -> array of nInst target units
    """
    n_inst, n_days = prices.shape
    commission_rates = compute_commission_rates(n_inst, algo_idx)
    position_limits = compute_position_limits(n_inst, algo_idx)

    cash = 0.0
    pos = np.zeros(n_inst)
    prev_equity = 0.0
    equity_curve = []
    daily_pl = []
    total_commission = 0.0

    for t in range(start_day, n_days):
        price_today = prices[:, t]
        price_hist = prices[:, : t + 1]

        new_pos = np.array(get_position_fn(price_hist), dtype=float)

        # enforce dollar position limits (safety net - your strategy should
        # already respect these, but the evaluator will too)
        dollar_pos = new_pos * price_today
        over = np.abs(dollar_pos) > position_limits
        if np.any(over):
            new_pos[over] = np.sign(new_pos[over]) * (position_limits[over] / price_today[over])
            new_pos = np.floor(new_pos)

        d_pos = new_pos - pos
        traded_dollars = np.abs(d_pos) * price_today
        commission = np.sum(traded_dollars * commission_rates)
        total_commission += commission

        cash -= np.sum(d_pos * price_today) + commission
        pos = new_pos

        equity = cash + np.sum(pos * price_today)
        daily_pl.append(equity - prev_equity)
        equity_curve.append(equity)
        prev_equity = equity

        if verbose:
            print(f"day {t}: PL={daily_pl[-1]:.2f}  equity={equity:.2f}  commission={commission:.2f}")

    daily_pl = np.array(daily_pl)
    results = score_from_pl(daily_pl)
    results["equity_curve"] = np.array(equity_curve)
    results["daily_pl_series"] = daily_pl
    results["total_commission"] = total_commission
    return results


if __name__ == "__main__":
    from The_Limitless_Liquidity_Providers import getMyPosition  # your strategy

    prices = generate_synthetic_data(n_inst=51, n_days=750, seed=42)
    results = run_backtest(prices, getMyPosition, algo_idx=0, start_day=61)

    print(f"Mean PL:         {results['mean_pl']:.2f}")
    print(f"Std PL:          {results['std_pl']:.2f}")
    print(f"Sharpe (ann.):   {results['sharpe']:.3f}")
    print(f"Total commission:{results['total_commission']:.2f}")
    print(f"Score:           {results['score']:.2f}")
