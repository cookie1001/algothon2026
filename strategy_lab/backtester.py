"""
Backtester that reproduces `eval.py` exactly, on days-down-the-rows data.

Every accounting detail below was matched line-for-line against eval.py's
`calcPL`, because a backtester that disagrees with the scorer is worse than no
backtester at all. Specifically:

  * position limits are converted to integer share caps with
    `int(dollar_limit / price)` (truncation), then positions are clipped to
    +/- that cap and cast to int - exactly as eval.py does;
  * commission on day t is charged against day t's cash *on day t+1*, which is
    eval.py's `cash -= curPrices.dot(deltaPos) + comm` using the *previous*
    day's `comm`. This one-day lag is a quirk of the reference implementation
    and is reproduced here rather than "fixed";
  * daily PL is the change in (cash + mark-to-market position value);
  * on the final day of the window eval.py does NOT call getPosition - it
    holds the previous position and uses that day purely to mark the book.
    `hold_on_final_day` reproduces that;
  * the Sharpe ratio is annualised with sqrt(250), matching eval.py, not 252;
  * the standard deviation uses ddof=0, matching numpy's default in eval.py.

Verified: with these settings `run_evaluation_window(prices, PortfolioConfig())`
reproduces `python eval.py` to the cent.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__))))

from The_Limitless_Liquidity_Providers import (  # noqa: E402
    ALGO_INDEX,
    TRADING_DAYS_PER_YEAR,
    build_commission_rates,
    build_dollar_position_limits,
)

PositionFunction = Callable[[np.ndarray], np.ndarray]
"""Takes (n_days, n_instruments) prices with days down the rows, returns a
length-n_instruments array of target share counts for the last row."""


@dataclass
class BacktestResult:
    mean_pl: float
    std_pl: float
    sharpe: float
    score: float
    total_commission: float
    mean_daily_commission: float
    mean_gross_dollar_exposure: float
    mean_net_dollar_exposure: float
    mean_daily_turnover: float
    worst_drawdown: float
    daily_pl: np.ndarray
    equity_curve: np.ndarray
    n_days: int

    def summary_row(self) -> Dict[str, float]:
        return {
            "score": self.score,
            "mean_pl": self.mean_pl,
            "sharpe": self.sharpe,
            "std_pl": self.std_pl,
            "worst_drawdown": self.worst_drawdown,
            "mean_commission": self.mean_daily_commission,
            "mean_gross_exposure": self.mean_gross_dollar_exposure,
            "mean_net_exposure": self.mean_net_dollar_exposure,
            "mean_turnover": self.mean_daily_turnover,
            "n_days": self.n_days,
        }


def compute_score(daily_pl: np.ndarray, score_param: float = 1.0) -> Dict[str, float]:
    """eval.py's scoring function: mean(PL) * SR^2/(SR^2 + param^2)."""
    mean_pl = float(np.mean(daily_pl))
    std_pl = float(np.std(daily_pl))
    if mean_pl <= 0 or std_pl < 1e-10:
        return {"mean_pl": mean_pl, "std_pl": std_pl, "sharpe": 0.0, "score": mean_pl}
    sharpe = np.sqrt(TRADING_DAYS_PER_YEAR) * mean_pl / std_pl
    fraction = sharpe**2 / (sharpe**2 + score_param**2)
    return {
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "sharpe": float(sharpe),
        "score": float(mean_pl * fraction),
    }


def run_backtest(
    prices_by_day: np.ndarray,
    position_function: PositionFunction,
    start_day: int,
    end_day: Optional[int] = None,
    hold_on_final_day: bool = True,
) -> BacktestResult:
    """Run one continuous backtest over the price-day range [start_day, end_day).

    `position_function` is called once per day, in order, with
    `prices_by_day[: day + 1]` - history up to and including that day, days
    down the rows. No resets happen inside; if you want independent folds,
    reset the strategy between calls.

    Two boundary conventions, both copied from eval.py:
      * `start_day` is a WARM-UP day. The opening position is established at
        that day's close and its PL is not scored, so the first scored day is
        a clean day-over-day change rather than "cost of getting invested".
      * the last day of the window is a MARK-ONLY day (`hold_on_final_day`):
        positions are held, not recomputed.

    Scored days are therefore `end_day - start_day - 1`.
    """
    n_days_total, n_instruments = prices_by_day.shape
    end_day = n_days_total if end_day is None else end_day

    commission_rates = build_commission_rates(n_instruments)
    dollar_position_limits = build_dollar_position_limits(n_instruments)

    cash = 0.0
    current_positions = np.zeros(n_instruments)
    previous_equity = 0.0
    carried_commission = 0.0  # eval.py charges the prior day's commission

    daily_pl = []
    equity_curve = []
    gross_exposures = []
    net_exposures = []
    turnovers = []
    total_commission = 0.0

    for day in range(start_day, end_day):
        prices_today = prices_by_day[day]
        history = prices_by_day[: day + 1]

        if hold_on_final_day and day == end_day - 1:
            new_positions = current_positions.astype(int)
        else:
            requested = np.asarray(position_function(history), dtype=float)
            share_limits = (dollar_position_limits / prices_today).astype(int)
            new_positions = np.clip(requested, -share_limits, share_limits).astype(int)

        position_change = new_positions - current_positions
        cash -= prices_today.dot(position_change) + carried_commission

        traded_dollars = prices_today * np.abs(position_change)
        carried_commission = float(np.sum(traded_dollars * commission_rates))
        total_commission += carried_commission

        current_positions = new_positions.astype(float)
        position_value = current_positions.dot(prices_today)
        equity = cash + position_value

        if day > start_day:  # start_day is warm-up only, not scored
            daily_pl.append(equity - previous_equity)
            equity_curve.append(equity)
            dollar_positions = current_positions * prices_today
            gross_exposures.append(float(np.sum(np.abs(dollar_positions))))
            net_exposures.append(float(np.sum(dollar_positions)))
            turnovers.append(float(np.sum(traded_dollars)))
        previous_equity = equity

    daily_pl = np.asarray(daily_pl)
    equity_curve = np.asarray(equity_curve)
    n_days = len(daily_pl)

    if n_days == 0:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, daily_pl, equity_curve, 0)

    stats = compute_score(daily_pl)
    running_peak = np.maximum.accumulate(equity_curve)
    worst_drawdown = float(np.min(equity_curve - running_peak))

    return BacktestResult(
        mean_pl=stats["mean_pl"],
        std_pl=stats["std_pl"],
        sharpe=stats["sharpe"],
        score=stats["score"],
        total_commission=total_commission,
        mean_daily_commission=total_commission / n_days,
        mean_gross_dollar_exposure=float(np.mean(gross_exposures)),
        mean_net_dollar_exposure=float(np.mean(net_exposures)),
        mean_daily_turnover=float(np.mean(turnovers)),
        worst_drawdown=worst_drawdown,
        daily_pl=daily_pl,
        equity_curve=equity_curve,
        n_days=n_days,
    )


def measure_index_beta(
    prices_by_day: np.ndarray,
    position_function: PositionFunction,
    start_day: int,
    end_day: Optional[int] = None,
) -> Dict[str, float]:
    """Realised beta of the book's daily PL to ALGO's daily return.

    Use this to *verify* the hedge overlay rather than assume it: a hedged book
    should come out with a beta indistinguishable from zero, and "assumed
    neutral" has a habit of not being neutral.
    """
    n_days_total, n_instruments = prices_by_day.shape
    end_day = n_days_total if end_day is None else end_day

    result = run_backtest(prices_by_day, position_function, start_day, end_day)
    algo_returns = np.diff(np.log(prices_by_day[start_day - 1 : end_day, ALGO_INDEX]))
    algo_returns = algo_returns[: result.n_days]
    pl = result.daily_pl[: len(algo_returns)]

    if len(pl) < 3 or np.std(algo_returns) < 1e-12:
        return {"beta": 0.0, "correlation": 0.0, "r_squared": 0.0}

    beta = float(np.cov(pl, algo_returns)[0, 1] / np.var(algo_returns))
    correlation = float(np.corrcoef(pl, algo_returns)[0, 1])
    return {"beta": beta, "correlation": correlation, "r_squared": correlation**2}
