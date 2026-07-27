"""
The_Limitless_Liquidity_Providers.py - Algothon 2026 submission.

`eval.py` imports `getMyPosition` from here and nothing else.

-----------------------------------------------------------------------------
DEPENDENCIES
-----------------------------------------------------------------------------
Everything used here is in the grading sandbox's package set
(requirements-dev.txt), plus hmmlearn, which must be declared in the
submission's own requirements.txt:

    numpy, pandas, scipy, scikit-learn, statsmodels   - already in the sandbox
    hmmlearn                                          - declare in requirements.txt

ONE PERFORMANCE TRAP, because it cost us 8 minutes a sweep once already:
`statsmodels.tsa.stattools.adfuller` defaults to `autolag='AIC'`, which fits
~13 separate OLS models per call. Always pass `maxlag=<n>, autolag=None`.
Measured: 1.700 ms/call with the default, 0.198 ms/call with it pinned - an
8.6x difference, on a function called once per instrument per day.

-----------------------------------------------------------------------------
SHAPE CONVENTION
-----------------------------------------------------------------------------
`eval.py` hands us prices as (n_instruments, n_days). Every internal function
in this file instead uses **days down the rows, instruments across the
columns** - `(n_days, n_instruments)`, the same orientation as the raw
`prices.txt`. `getMyPosition` performs the single transpose at the boundary;
nothing downstream ever transposes again. Any array named `*_by_day` is
(n_days, n_instruments); any `*_per_instrument` is a flat length-n_instruments
vector for the current day.

-----------------------------------------------------------------------------
ARCHITECTURE
-----------------------------------------------------------------------------
    prices (n_days, n_inst)
        |
        +-- StatisticalArbitrageStrategy --+
        +-- MeanReversionStrategy ---------+--> RegimeAwareEnsemble --+
        +-- MomentumStrategy -------------+        ^                  |
                                                    |                 v
                         GaussianHmmRegimeDetector -+          IndexHedgeOverlay
                                                                      |
                                                                      v
                                                         size -> clip to limits
                                                                      |
                                                                      v
                                                           integer share counts

Each strategy returns a *signal*, not a position. See `SignalStrategy` for the
contract.

-----------------------------------------------------------------------------
COMPETITION RULES ENCODED HERE
-----------------------------------------------------------------------------
    position limit   ALGO (index 0): $100,000     every other instrument: $10,000
    commission       ALGO (index 0): 0.2 bps      every other instrument: 1 bps
    score            mean(PL) * SR^2/(SR^2+1)  when mean(PL) > 0, else mean(PL)

Because the score is *linear* in mean(PL) but the Sharpe term is
scale-invariant, running as close to the position limits as the signal
justifies is strictly better than running small. Hence the aggressive
`SizingConfig.signal_gain`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# hmmlearn and statsmodels are imported lazily inside the functions that use
# them, so an install problem degrades one component instead of preventing the
# module from importing at all - which would zero the whole submission.

# =============================================================================
# Market constants
# =============================================================================

ALGO_INDEX = 0  # ALGO is column 0 of prices.txt (verified against the header row)

ALGO_DOLLAR_POSITION_LIMIT = 100_000.0
DEFAULT_DOLLAR_POSITION_LIMIT = 10_000.0

ALGO_COMMISSION_RATE = 0.2 / 10_000.0  # 0.2 basis points
DEFAULT_COMMISSION_RATE = 1.0 / 10_000.0  # 1 basis point

TRADING_DAYS_PER_YEAR = 250  # matches eval.py's sqrt(250) Sharpe annualisation


def build_dollar_position_limits(n_instruments: int) -> np.ndarray:
    """Per-instrument dollar position limits, ALGO special-cased."""
    limits = np.full(n_instruments, DEFAULT_DOLLAR_POSITION_LIMIT)
    limits[ALGO_INDEX] = ALGO_DOLLAR_POSITION_LIMIT
    return limits


def build_commission_rates(n_instruments: int) -> np.ndarray:
    """Per-instrument commission rates on traded notional, ALGO special-cased."""
    rates = np.full(n_instruments, DEFAULT_COMMISSION_RATE)
    rates[ALGO_INDEX] = ALGO_COMMISSION_RATE
    return rates


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StatArbConfig:
    """Hyperparameters for the PCA-residual statistical arbitrage strategy."""

    volatility_span: int = 60
    """EWMA span (days) used to standardise returns before any factor work."""

    factor_model: str = "pca"
    """'pca'           - PCA on all non-ALGO instruments (classic).
       'algo_plus_pca' - ALGO used directly as factor 1 (it *is* PC1 here),
                         then PCA with n_factors-1 components on the
                         ALGO-residuals. Better at low n_factors; plain PCA
                         won on validation at the n_factors we ship."""

    factor_fit_window: int = 252
    """Trailing days used to estimate factor loadings."""

    n_factors: int = 30
    """Factors stripped out before looking for reversion. 30 of 50 looks
    aggressive, but only the first three eigenvalues (11.9, 2.8, 1.9) stand
    clear of the noise bulk - everything from the 8th on sits at ~1.0. See the
    n_factors x reversion_window grid in strategy_lab/README.md."""

    reversion_window: int = 20
    """Days of cumulative residual fed to the OU/AR(1) fit - the horizon of the
    reversion being traded. Centre of the 15-25 plateau."""

    zscore_method: str = "cross_sectional"
    """'cross_sectional' - standardise the cumulative residual across
                           instruments today. Robust, no per-name fit,
                           naturally dollar-neutral.
       'ou'              - standardise each instrument by its own fitted OU
                           equilibrium. Textbook, but noisier."""

    require_stationary_residual: bool = False
    adf_pvalue_threshold: float = 0.05
    """Gate trades on an Augmented Dickey-Fuller test of the residual.
    Only consulted when zscore_method == 'ou'."""

    min_half_life: float = 1.0
    max_half_life: float = 25.0
    """Only trade residuals whose fitted half-life sits in this band."""

    entry_zscore: float = 0.0
    exit_zscore: float = 0.0
    """Separate thresholds create hysteresis and cut churn near zero."""

    max_abs_zscore: float = 3.0
    """Winsorise so one blown-up residual cannot dominate the book."""


@dataclass
class MeanReversionConfig:
    """Hyperparameters for the per-instrument mean-reversion strategy.

    This is the highest-scoring single strategy on this dataset - short-horizon
    reversion in the ALGO-residual. Its PL is almost uncorrelated with
    stat-arb's (+0.07), which is what makes the ensemble worth having.
    """

    lookback_window: int = 5
    volatility_span: int = 60
    entry_zscore: float = 1.0
    exit_zscore: float = 0.5
    max_abs_zscore: float = 3.0
    beta_hedge_against_algo: bool = True
    """Test reversion on the ALGO-residual rather than the raw price, so a
    market-wide move is not mistaken for an idiosyncratic dislocation. Worth
    ~180 points of pooled score at this lookback - by far the biggest single
    decision in this strategy."""
    beta_estimation_window: int = 120


@dataclass
class MomentumConfig:
    """Hyperparameters for the momentum strategy.

    HEALTH WARNING: this universe reverts at every horizon tested (3 to 250
    days, cross-sectional IC negative throughout, t as low as -3.8). Plain
    momentum loses money in every configuration we validated, from -136 to
    roughly 0. The trend gate below exists to restrict it to the minority of
    instruments that genuinely trend; see strategy_lab/README.md for what it
    is and is not worth.
    """

    lookback_windows: tuple = (60, 120)
    """Blend several horizons. Long horizons validated least-badly - short
    ones sit right on top of the strongest reversion."""

    volatility_span: int = 60
    skip_recent_days: int = 1
    """Skip the most recent day(s) to avoid short-horizon reversal."""

    cross_sectional: bool = True
    """True  -> rank across instruments, long top / short bottom
                (market-neutral by construction).
       False -> per-instrument time-series momentum (carries net index beta,
                which the hedge overlay downstream then has to remove)."""

    trend_gate_enabled: bool = True
    min_variance_ratio: float = 1.1
    """Only trade instruments whose variance ratio exceeds this. VR > 1 means
    returns compound (trending); VR < 1 means they offset (reverting). This is
    the Hurst/ADX-style filter - it is what stops momentum from trading the
    reverting majority of this universe, and it is the difference between a
    pooled score of -136 (ungated, short lookbacks) and roughly 0 (gated).
    Do not turn it off."""
    variance_ratio_window: int = 120
    variance_ratio_lag: int = 5

    max_abs_zscore: float = 3.0


@dataclass
class RegimeConfig:
    """Hyperparameters for the hmmlearn Gaussian HMM regime detector."""

    n_regimes: int = 2
    feature_window: int = 250
    refit_interval_days: int = 25
    """Refitting Baum-Welch daily is wasteful and makes the regime label
    jitter. Refit on this cadence; run the cheap forward pass in between."""
    n_em_iterations: int = 40
    random_seed: int = 0


@dataclass
class EnsembleConfig:
    """How strategy signals are blended."""

    baseline_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "stat_arb": 1.0,
            "mean_reversion": 0.5,
            "momentum": 0.05,
        }
    )
    """Weight each strategy gets before the regime adjustment.

    Validated on 21 walk-forward folds, not guessed. The three PL streams are
    almost perfectly uncorrelated with each other (+0.001, +0.051, +0.028), so
    blending genuinely raises Sharpe rather than just averaging.

    Walk-forward pooled score by stat_arb:mean_reversion ratio (momentum 0,
    hedge on): 1:0.25 -> 103, 1:0.5 -> 120, 1:1 -> 98, 0.5:1 -> 84. The 1:0.5
    peak also has the best fold consistency (86-90% profitable folds).

    momentum is held at 0.05 because it is a *measured drag* even after the
    trend gate: 1:0.5:0 -> 120, :0.05 -> 111, :0.10 -> 92, :0.25 -> 42. It is
    non-zero so the strategy stays live, wired in and visible in the score.
    Raise it the moment a rebuild validates positive standalone; see
    MomentumConfig and strategy_lab/README.md."""

    minimum_weight_floor: float = 0.05
    """Never let the regime call zero out a strategy with a non-zero baseline -
    regime detection is itself noisy."""

    calm_regime_tilt: Dict[str, float] = field(
        default_factory=lambda: {
            "stat_arb": 1.0,
            "mean_reversion": 1.3,
            "momentum": 0.7,
        }
    )
    """Multiplier on the baseline weight, scaled by P(calm regime). Mean
    reversion works when the market is quiet; momentum works when it trends.
    Stat-arb is deliberately flat - it is a relative-value bet, so the
    market-wide vol regime is the wrong conditioning variable (its own quality
    metric is used instead, see `stat_arb_quality_scaling`)."""

    stressed_regime_tilt: Dict[str, float] = field(
        default_factory=lambda: {
            "stat_arb": 1.0,
            "mean_reversion": 0.7,
            "momentum": 1.3,
        }
    )

    stat_arb_quality_scaling: bool = True
    """Scale stat-arb's weight by the fraction of residuals currently passing
    its half-life screen, rather than by the vol regime."""


@dataclass
class HedgeConfig:
    """Index-exposure hedge overlay, applied after the ensemble."""

    enabled: bool = True
    """ON, and it earns its place. Stat-arb alone is factor-neutral by
    construction and needs no hedge - with stat-arb as the only live strategy
    the overlay actually *cost* score. But mean-reversion and momentum both
    carry real index exposure, so the combined book does too: at the shipped
    weights the hedge is worth +47 pooled score (120 hedged vs 73 unhedged).

    Validate with `strategy_lab.backtester.measure_index_beta` after any
    weight change - do not assume it worked."""

    target_hedge_ratio: float = 1.0
    """Fraction of measured net index exposure to offset. A partial hedge is
    the usual defence against over-trading ALGO on estimation noise, but that
    logic does not apply here: ALGO is an *exact* basket of the other 50, so
    the exposure being offset is measured, not estimated. Full neutralisation
    validated best (1.0 -> 120, 0.8 -> 116, 0.5 -> 60)."""

    basket_estimation_window: int = 252
    ridge_penalty: float = 1e-4
    """Ridge rather than OLS: the 50 constituent prices are highly collinear."""

    min_hedge_dollars: float = 2_000.0
    """Do not bother trading ALGO for exposures smaller than this."""


@dataclass
class SizingConfig:
    """Signal -> dollar position -> integer share count."""

    signal_gain: float = 8.0
    """Multiplier applied to the combined signal before saturation. The signal
    is a cross-sectional z-score, so gain=1 puts a 1-sigma name at the full
    limit; gain=8 saturates almost everything, which validated best because the
    score is linear in mean(PL) and the position limits, not risk, are the
    binding constraint. Broad plateau from 5 to 15 - not a knife edge."""

    capital_utilisation: float = 1.0
    """Fraction of each instrument's dollar limit available to the strategy."""

    rebalance_threshold_dollars: float = 1000.0
    """No-trade band: skip a rebalance unless the target moves more than this.
    Score effect is inside the noise; it is set because it cuts turnover for
    free, which is cheap insurance if the real cost schedule differs."""

    position_smoothing_days: float = 1.0
    """Exponential smoothing of target positions. 1.0 = no smoothing."""


@dataclass
class PortfolioConfig:
    """Top-level container - the single object the sweep tooling mutates."""

    stat_arb: StatArbConfig = field(default_factory=StatArbConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)

    def minimum_history_days(self) -> int:
        """Days of price history needed before the engine will trade.

        Kept deliberately low. `eval.py` scores the final `numTestDays` days,
        so the very first scored day only has `n_days - numTestDays` of
        history: at numTestDays=500 on a 1000-day file that is 500 days. Any
        requirement above that silently sits flat through the start of the
        scored window. pandas' `ewm` is bias-corrected from the first
        observation, so no volatility burn-in is needed.
        """
        return max(
            self.stat_arb.reversion_window + 30,
            self.mean_reversion.lookback_window + self.mean_reversion.beta_estimation_window + 5,
            self.momentum.variance_ratio_window + 5,
            60,
        ) + 2


# =============================================================================
# Shared numerical helpers (days down the rows throughout)
# =============================================================================


def compute_log_returns(prices_by_day: np.ndarray) -> np.ndarray:
    """(n_days, n_inst) prices -> (n_days - 1, n_inst) log returns."""
    return np.diff(np.log(prices_by_day), axis=0)


def volatility_standardised_returns(
    returns_by_day: np.ndarray, span: int
) -> np.ndarray:
    """Divide each return by its trailing EWMA volatility.

    Uses `pandas.DataFrame.ewm(...).std()`, which is bias-corrected from the
    first observation (`adjust=True`). That matters beyond brevity: a
    hand-rolled zero-initialised recursion needs a few hundred rows of burn-in
    before it is trustworthy, which would push `minimum_history_days` past the
    500 days available at the start of a 500-day scoring window.
    """
    frame = pd.DataFrame(returns_by_day)
    volatility = frame.ewm(span=span, min_periods=2).std()
    volatility = volatility.bfill().fillna(1.0).clip(lower=1e-10)
    return (frame / volatility).to_numpy()


def cross_sectional_zscore(
    values_per_instrument: np.ndarray, valid_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Standardise across instruments. Entries outside `valid_mask` become 0."""
    result = np.zeros_like(values_per_instrument, dtype=float)
    mask = np.ones_like(values_per_instrument, dtype=bool) if valid_mask is None else valid_mask
    if mask.sum() < 2:
        return result
    subset = values_per_instrument[mask]
    if np.nanstd(subset) < 1e-12:
        return result
    result[mask] = np.nan_to_num(stats.zscore(subset, nan_policy="omit"))
    return result


def fit_ar1(series_by_day: np.ndarray) -> Dict[str, np.ndarray]:
    """AR(1) / Dickey-Fuller fit of every column, via `scipy.stats.linregress`.

    Regresses the Dickey-Fuller form  delta_x_t = c + gamma * x_{t-1} + noise,
    which is algebraically the AR(1)  x_t = a + b * x_{t-1}  with b = 1 + gamma.
    `linregress` returns the slope and its standard error in one call, so the
    DF t-statistic is `gamma / stderr` with no extra work.

    For a formal p-value use `augmented_dickey_fuller_pvalues` below - but note
    that this t-statistic is the same quantity, and comparing it against the
    usual -2.89 critical value is enough for a screen.

    Returns dict of length-n_cols arrays: ar1_coefficient, half_life,
    dickey_fuller_tstat, zscore (vs the fitted OU equilibrium).
    """
    n_cols = series_by_day.shape[1]
    lagged, delta = series_by_day[:-1], np.diff(series_by_day, axis=0)

    empty = {
        "ar1_coefficient": np.full(n_cols, np.nan),
        "half_life": np.full(n_cols, np.nan),
        "dickey_fuller_tstat": np.full(n_cols, np.nan),
        "zscore": np.zeros(n_cols),
    }
    if lagged.shape[0] < 5:
        return empty

    ar1_coefficient = np.full(n_cols, np.nan)
    half_life = np.full(n_cols, np.nan)
    tstat = np.full(n_cols, np.nan)
    zscore = np.zeros(n_cols)

    for column in range(n_cols):
        x, dy = lagged[:, column], delta[:, column]
        if np.std(x) < 1e-12:
            continue
        fit = stats.linregress(x, dy)
        gamma, intercept = fit.slope, fit.intercept
        b = 1.0 + gamma
        ar1_coefficient[column] = b
        if fit.stderr > 1e-14:
            tstat[column] = gamma / fit.stderr
        if not (0.0 < b < 1.0):
            continue  # explosive, unit-root or oscillating: no half-life
        half_life[column] = -np.log(2.0) / np.log(b)

        # OU equilibrium implied by the fit.
        residual_variance = np.var(dy - (intercept + gamma * x), ddof=2)
        equilibrium_mean = -intercept / gamma
        equilibrium_variance = residual_variance / (1.0 - b**2)
        if equilibrium_variance > 1e-14:
            zscore[column] = (series_by_day[-1, column] - equilibrium_mean) / np.sqrt(
                equilibrium_variance
            )

    return {
        "ar1_coefficient": ar1_coefficient,
        "half_life": half_life,
        "dickey_fuller_tstat": tstat,
        "zscore": np.nan_to_num(zscore, nan=0.0, posinf=0.0, neginf=0.0),
    }


def augmented_dickey_fuller_pvalues(series_by_day: np.ndarray, maxlag: int = 1) -> np.ndarray:
    """ADF p-value per column, via statsmodels.

    ALWAYS pass `maxlag` and `autolag=None`. The default `autolag='AIC'` fits
    ~13 OLS models per call: 1.700 ms/call against 0.198 ms/call pinned. At 50
    instruments x 500 days x a sweep of configurations that difference is the
    gap between seconds and minutes.
    """
    from statsmodels.tsa.stattools import adfuller

    n_cols = series_by_day.shape[1]
    pvalues = np.ones(n_cols)
    for column in range(n_cols):
        series = series_by_day[:, column]
        if len(series) < maxlag + 5 or np.std(series) < 1e-12:
            continue
        try:
            pvalues[column] = adfuller(series, maxlag=maxlag, autolag=None, regression="c")[1]
        except (ValueError, np.linalg.LinAlgError):
            pvalues[column] = 1.0
    return pvalues


def variance_ratio(returns_by_day: np.ndarray, lag: int) -> np.ndarray:
    """Lo-MacKinlay variance ratio per column: Var(lag-day return) / (lag * Var(1-day)).

    VR > 1 means returns compound in the same direction (trending); VR < 1
    means they offset (reverting); VR = 1 is a random walk. Cheaper and less
    fragile than a full Hurst exponent, and it answers the same question:
    should momentum be allowed to trade this instrument at all?
    """
    n_rows = returns_by_day.shape[0]
    if n_rows < 2 * lag + 2:
        return np.ones(returns_by_day.shape[1])
    single_period_variance = returns_by_day.var(axis=0)
    usable = (n_rows // lag) * lag
    aggregated = returns_by_day[-usable:].reshape(-1, lag, returns_by_day.shape[1]).sum(axis=1)
    multi_period_variance = aggregated.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = multi_period_variance / (lag * single_period_variance)
    return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)


# =============================================================================
# Strategy interface
# =============================================================================


class SignalStrategy:
    """Base class every strategy must implement.

    ------------------------------------------------------------------------
    CONTRACT - read this before writing a strategy
    ------------------------------------------------------------------------
    generate_signal(prices_by_day) -> np.ndarray, shape (n_instruments,)

    INPUT
      `prices_by_day` is (n_days, n_instruments) float - **days down the
      rows**, instruments across the columns. Column `ALGO_INDEX` (0) is ALGO.
      The last row is today. You may only use what is in this array.

    OUTPUT - all four are required:
      1. Length `n_instruments`, dtype float, finite everywhere (no NaN/inf).
         The ensemble rejects a signal that violates this and substitutes
         zeros for that strategy - so a bug shows up as "no contribution",
         not a crash. Check your own output.
      2. SIGN is direction (positive = want to be long), MAGNITUDE is
         conviction. Do NOT return dollar amounts or share counts - sizing,
         limit-clipping and hedging all happen downstream in PortfolioEngine.
      3. Scale is up to you. The ensemble z-scores every signal across
         instruments before weighting, so a raw percentage return and a
         z-score end up on the same footing. Only *relative* magnitudes
         within your own signal matter.
      4. Entry `ALGO_INDEX` must be 0.0 unless your strategy deliberately
         wants outright index exposure. ALGO is an exact fixed-weight basket
         of the other 50, so a non-zero value there is a pure market bet.

    STATE
      Called once per day, in order, with history growing by one row. You may
      keep state on `self`, but implement `reset()` to clear it - the
      validation harness starts every fold flat.

    COST
      Called ~500 times by eval.py and thousands of times per sweep. Keep it
      vectorised. If you use `statsmodels.adfuller`, pass `maxlag` and
      `autolag=None` (see `augmented_dickey_fuller_pvalues`).
    """

    name: str = "unnamed"

    def generate_signal(self, prices_by_day: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any per-fold state. Override if the strategy is stateful."""

    def quality_score(self) -> float:
        """Optional self-assessment in [0, 1], used by the ensemble to scale
        this strategy's weight. Default 1.0 = 'no opinion'."""
        return 1.0


def _apply_entry_exit_hysteresis(
    signal_per_instrument: np.ndarray,
    open_position_sign: np.ndarray,
    entry_zscore: float,
    exit_zscore: float,
):
    """Separate entry/exit thresholds so positions do not churn near zero.

    A flat instrument needs |signal| >= entry to open; an open one is closed
    only when |signal| drops below exit. Shared by the reversion strategies.
    """
    if entry_zscore <= 0.0 and exit_zscore <= 0.0:
        return signal_per_instrument, np.sign(signal_per_instrument)

    magnitude = np.abs(signal_per_instrument)
    currently_open = open_position_sign != 0.0
    active = ((~currently_open) & (magnitude >= entry_zscore)) | (
        currently_open & (magnitude >= exit_zscore)
    )
    held = np.where(active, signal_per_instrument, 0.0)
    return held, np.sign(held)


# =============================================================================
# Strategy 1 - PCA-residual statistical arbitrage
# =============================================================================


class StatisticalArbitrageStrategy(SignalStrategy):
    """Trades mean reversion in the part of each instrument's return that the
    factor model cannot explain.

    Pipeline, days down the rows throughout:
      1. log returns -> divide by EWMA vol, so every instrument enters the
         factor fit on a comparable scale.
      2. `sklearn.decomposition.PCA` estimates factor loadings on a trailing
         `factor_fit_window`.
      3. Residual returns = standardised returns - PCA reconstruction.
      4. Cumulative residual over `reversion_window` days is the "spread".
      5. Fit AR(1)/OU to the spread; extract a z-score and a half-life.
      6. Signal = -z on residuals passing the screen.
    """

    name = "stat_arb"

    def __init__(self, config: Optional[StatArbConfig] = None):
        self.config = config or StatArbConfig()
        self._open_position_sign = None
        self._last_screen_pass_fraction = 1.0

    def reset(self) -> None:
        self._open_position_sign = None
        self._last_screen_pass_fraction = 1.0

    def quality_score(self) -> float:
        return float(self._last_screen_pass_fraction)

    def _residual_returns(self, standardised_returns_by_day: np.ndarray) -> np.ndarray:
        """(window, n_inst) standardised returns -> (window, n_non_algo) residuals."""
        cfg = self.config
        non_algo = np.delete(standardised_returns_by_day, ALGO_INDEX, axis=1)
        n_samples, n_series = non_algo.shape

        if cfg.factor_model == "algo_plus_pca":
            # Factor 1 is ALGO itself: in this dataset ALGO *is* PC1 (return
            # correlation 0.96, and in price space an exact fixed-weight basket
            # of the other 50), so using it directly is more stable than
            # re-estimating PC1 from a rolling covariance every day.
            algo_column = standardised_returns_by_day[:, ALGO_INDEX]
            algo_variance = float(algo_column @ algo_column)
            if algo_variance > 1e-12:
                betas = (algo_column @ non_algo) / algo_variance
                non_algo = non_algo - np.outer(algo_column, betas)
            remaining = max(cfg.n_factors - 1, 0)
            if remaining == 0:
                return non_algo
            n_components = min(remaining, n_series, n_samples)
        else:
            n_components = min(cfg.n_factors, n_series, n_samples)

        model = PCA(n_components=n_components)
        scores = model.fit_transform(non_algo)
        return non_algo - model.inverse_transform(scores)

    def generate_signal(self, prices_by_day: np.ndarray) -> np.ndarray:
        cfg = self.config
        n_days, n_instruments = prices_by_day.shape

        if self._open_position_sign is None or self._open_position_sign.shape[0] != n_instruments:
            self._open_position_sign = np.zeros(n_instruments)

        if n_days < cfg.reversion_window + 25:
            return np.zeros(n_instruments)

        returns_by_day = compute_log_returns(prices_by_day)
        standardised = volatility_standardised_returns(returns_by_day, cfg.volatility_span)
        # Use as much of the fit window as history allows - at the start of a
        # 500-day scoring window there may be less than factor_fit_window.
        fit_rows = min(cfg.factor_fit_window, standardised.shape[0])
        standardised = standardised[-fit_rows:]

        residual_returns = self._residual_returns(standardised)
        cumulative_residual = np.cumsum(residual_returns[-cfg.reversion_window :], axis=0)

        fit = fit_ar1(cumulative_residual)
        half_life = fit["half_life"]

        tradeable = np.isfinite(half_life)
        tradeable &= (half_life >= cfg.min_half_life) & (half_life <= cfg.max_half_life)
        if cfg.require_stationary_residual and cfg.zscore_method == "ou":
            pvalues = augmented_dickey_fuller_pvalues(cumulative_residual)
            tradeable &= pvalues < cfg.adf_pvalue_threshold

        self._last_screen_pass_fraction = float(tradeable.mean()) if tradeable.size else 0.0

        if cfg.zscore_method == "cross_sectional":
            # Standardise the final cumulative residual across instruments:
            # robust, no per-name small-sample fit, naturally dollar-neutral.
            zscore_non_algo = cross_sectional_zscore(cumulative_residual[-1])
        else:
            zscore_non_algo = np.where(tradeable, fit["zscore"], 0.0)

        zscore_non_algo = np.clip(
            np.nan_to_num(zscore_non_algo), -cfg.max_abs_zscore, cfg.max_abs_zscore
        )

        signal_per_instrument = np.zeros(n_instruments)
        non_algo_slots = [i for i in range(n_instruments) if i != ALGO_INDEX]
        # Reversion: a stretched-high spread is a short.
        signal_per_instrument[non_algo_slots] = -zscore_non_algo
        if cfg.zscore_method == "ou":
            signal_per_instrument[non_algo_slots] *= tradeable

        held, self._open_position_sign = _apply_entry_exit_hysteresis(
            signal_per_instrument, self._open_position_sign, cfg.entry_zscore, cfg.exit_zscore
        )
        return held


# =============================================================================
# Strategy 2 - per-instrument mean reversion  (TEAMMATE A)
# =============================================================================


class MeanReversionStrategy(SignalStrategy):
    """Short-horizon reversion in the ALGO-residual.

    TEAMMATE A: this is a validated, working strategy, not a placeholder - it
    is the single highest-scoring component in the book. Improve it rather
    than replacing it wholesale, and re-validate against
    `strategy_lab.walk_forward` before changing the shipped config.

    Suggested build-out, in rough priority order:
      1. Screen per-instrument for reversion on a *rolling* window (ADF via
         `augmented_dickey_fuller_pvalues`, or the `variance_ratio` helper),
         re-run periodically so the qualifying list changes over time. Do not
         screen once on the full sample - that leaks the future into every
         fold.
      2. Set `lookback_window` per instrument from the fitted AR(1) half-life
         (`fit_ar1`) instead of one global number.
      3. The ALGO-beta residualisation is worth ~180 points of pooled score at
         lookback 5. Anything that improves the beta estimate (shrinkage,
         a longer window, a robust regression) is worth trying.
      4. Keep the separate entry/exit thresholds - the strategy is very
         sensitive to them.
    """

    name = "mean_reversion"

    def __init__(self, config: Optional[MeanReversionConfig] = None):
        self.config = config or MeanReversionConfig()
        self._open_position_sign = None

    def reset(self) -> None:
        self._open_position_sign = None

    def generate_signal(self, prices_by_day: np.ndarray) -> np.ndarray:
        cfg = self.config
        n_days, n_instruments = prices_by_day.shape

        if self._open_position_sign is None or self._open_position_sign.shape[0] != n_instruments:
            self._open_position_sign = np.zeros(n_instruments)

        if n_days < cfg.lookback_window + 25:
            return np.zeros(n_instruments)

        returns_by_day = compute_log_returns(prices_by_day)

        if cfg.beta_hedge_against_algo:
            # Strip each instrument's exposure to the market index, so a
            # market-wide move is not mistaken for an idiosyncratic
            # dislocation. This is the single biggest decision in the strategy.
            window_rows = min(cfg.beta_estimation_window, returns_by_day.shape[0])
            window = returns_by_day[-window_rows:]
            algo_returns = window[:, ALGO_INDEX]
            algo_variance = float(algo_returns @ algo_returns)
            betas = (
                (algo_returns @ window) / algo_variance
                if algo_variance > 1e-12
                else np.zeros(n_instruments)
            )
            working_returns = returns_by_day - np.outer(returns_by_day[:, ALGO_INDEX], betas)
        else:
            working_returns = returns_by_day

        standardised = volatility_standardised_returns(working_returns, cfg.volatility_span)

        # Cumulative standardised return over the lookback = displacement from
        # the rolling mean in volatility units; /sqrt(window) makes it a
        # t-statistic so the threshold means the same thing at any lookback.
        displacement = standardised[-cfg.lookback_window :].sum(axis=0)
        displacement /= np.sqrt(cfg.lookback_window)

        signal_per_instrument = -np.clip(
            np.nan_to_num(displacement), -cfg.max_abs_zscore, cfg.max_abs_zscore
        )
        signal_per_instrument[ALGO_INDEX] = 0.0

        held, self._open_position_sign = _apply_entry_exit_hysteresis(
            signal_per_instrument, self._open_position_sign, cfg.entry_zscore, cfg.exit_zscore
        )
        return held


# =============================================================================
# Strategy 3 - momentum  (TEAMMATE B)
# =============================================================================


class MomentumStrategy(SignalStrategy):
    """Risk-adjusted momentum, gated to instruments that actually trend.

    TEAMMATE B: read this before spending time here.

    This universe **reverts at every horizon we tested** - cross-sectional
    information coefficient is negative from 3 to 250 days (t as low as -3.8).
    Validated standalone on walk-forward folds, plain momentum scored:

        lookbacks (5,10)     -136        (20,60)    -156
        lookbacks (10,20)    -108        (60,120)    -56
        lookbacks (120,250)  roughly 0

    Cross-sectional and time-series variants were within 1 point of each other
    at every horizon. Momentum is not merely weak here, it is the reversion
    signal with the sign flipped - which is why the shipped weight is 0.10
    rather than 0.

    That makes the trend gate the whole game: `variance_ratio` restricts
    trading to instruments whose returns actually compound (VR > 1) rather
    than offset. If you can find a subset that genuinely trends, momentum
    earns its weight; if you cannot, say so and we drop it to the floor.

    Other things worth trying:
      1. Regime-conditional momentum - trade it only when the HMM says
         stressed. The ensemble already tilts this way; a hard gate may work
         better than a soft tilt.
      2. Report realised net ALGO beta of your signal explicitly with
         `strategy_lab.backtester.measure_index_beta`. The ensemble cannot fix
         what it is not told about. The time-series variant
         (`cross_sectional=False`) carries real index beta by construction -
         that is what the hedge overlay is for.
      3. If your rebuild validates positive standalone, raise
         `EnsembleConfig.baseline_weights['momentum']` and re-run the sweep.
    """

    name = "momentum"

    def __init__(self, config: Optional[MomentumConfig] = None):
        self.config = config or MomentumConfig()
        self._last_gate_pass_fraction = 1.0

    def quality_score(self) -> float:
        return float(self._last_gate_pass_fraction)

    def generate_signal(self, prices_by_day: np.ndarray) -> np.ndarray:
        cfg = self.config
        n_days, n_instruments = prices_by_day.shape

        if n_days < min(cfg.lookback_windows) + cfg.skip_recent_days + 25:
            return np.zeros(n_instruments)

        returns_by_day = compute_log_returns(prices_by_day)
        standardised = volatility_standardised_returns(returns_by_day, cfg.volatility_span)
        if cfg.skip_recent_days > 0:
            standardised = standardised[: -cfg.skip_recent_days]

        blended = np.zeros(n_instruments)
        horizons_used = 0
        for lookback in cfg.lookback_windows:
            if standardised.shape[0] < lookback:
                continue
            # Already in volatility units, so a sum over the window divided by
            # sqrt(window) is a per-horizon t-statistic: risk-adjusted
            # momentum, not raw return.
            blended += standardised[-lookback:].sum(axis=0) / np.sqrt(lookback)
            horizons_used += 1
        if horizons_used == 0:
            return np.zeros(n_instruments)
        blended /= horizons_used

        if cfg.trend_gate_enabled:
            gate_rows = min(cfg.variance_ratio_window, returns_by_day.shape[0])
            ratios = variance_ratio(returns_by_day[-gate_rows:], cfg.variance_ratio_lag)
            trending = ratios > cfg.min_variance_ratio
            trending[ALGO_INDEX] = False
            self._last_gate_pass_fraction = float(trending[1:].mean())
            blended = np.where(trending, blended, 0.0)

        blended[ALGO_INDEX] = 0.0
        if cfg.cross_sectional:
            non_algo_mask = np.ones(n_instruments, dtype=bool)
            non_algo_mask[ALGO_INDEX] = False
            non_algo_mask &= blended != 0.0  # do not let gated-out names shift the mean
            blended = cross_sectional_zscore(blended, non_algo_mask)

        blended = np.clip(np.nan_to_num(blended), -cfg.max_abs_zscore, cfg.max_abs_zscore)
        blended[ALGO_INDEX] = 0.0
        return blended


# =============================================================================
# Regime detection - hmmlearn
# =============================================================================


class GaussianHmmRegimeDetector:
    """Two-state Gaussian HMM over market-state features, via hmmlearn.

    Features (per day, on the trailing window):
      1. ALGO's own return - ALGO is the market index, so this is the cleanest
         market-direction feature available.
      2. |ALGO return| - the volatility feature that separates the states.
      3. Cross-sectional dispersion of returns - how much instruments move
         relative to each other, which is what stat-arb and momentum care
         about.

    NOTE on feature choice: the original plan called for rolling ALGO-basket
    correlation as a stress feature. That is unusable in this dataset - ALGO
    is an exact fixed-weight basket of the other 50, so the correlation is
    pinned at ~1.0 every day and carries no information. Cross-sectional
    dispersion replaces it.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.model = None
        self.calm_state_index = None
        self._days_at_last_fit = None

    def reset(self) -> None:
        self.model = None
        self.calm_state_index = None
        self._days_at_last_fit = None

    def _build_features(self, prices_by_day: np.ndarray) -> np.ndarray:
        window = prices_by_day[-(self.config.feature_window + 1) :]
        returns_by_day = compute_log_returns(window)
        algo_return = returns_by_day[:, ALGO_INDEX]
        dispersion = returns_by_day.std(axis=1)
        features = np.column_stack([algo_return, np.abs(algo_return), dispersion])
        centre = features.mean(axis=0)
        spread = np.maximum(features.std(axis=0), 1e-10)
        return (features - centre) / spread

    def fit(self, prices_by_day: np.ndarray) -> "GaussianHmmRegimeDetector":
        from hmmlearn.hmm import GaussianHMM

        cfg = self.config
        features = self._build_features(prices_by_day)
        model = GaussianHMM(
            n_components=cfg.n_regimes,
            covariance_type="diag",
            n_iter=cfg.n_em_iterations,
            random_state=cfg.random_seed,
        )
        model.fit(features)
        self.model = model
        # The calm state is the one with the lowest mean on the |ALGO return|
        # feature. Labelling by a feature mean (not by fitted variance) keeps
        # the label stable across refits, so the weights do not flip sign
        # just because EM relabelled the states.
        self.calm_state_index = int(np.argmin(model.means_[:, 1]))
        self._days_at_last_fit = prices_by_day.shape[0]
        return self

    def probability_of_calm_regime(self, prices_by_day: np.ndarray) -> float:
        """P(calm | data) for today. Refits on the configured cadence.

        Returns 0.5 ("no opinion") on any failure, so a regime problem
        degrades to equal weighting instead of taking the book down.
        """
        cfg = self.config
        n_days = prices_by_day.shape[0]
        needs_refit = (
            self.model is None
            or self._days_at_last_fit is None
            or n_days < self._days_at_last_fit  # history restarted (new fold)
            or n_days - self._days_at_last_fit >= cfg.refit_interval_days
        )
        try:
            if needs_refit:
                self.fit(prices_by_day)
            posteriors = self.model.predict_proba(self._build_features(prices_by_day))
            probability = float(posteriors[-1, self.calm_state_index])
        except Exception:
            return 0.5
        return probability if np.isfinite(probability) else 0.5


# =============================================================================
# Ensemble
# =============================================================================


class RegimeAwareEnsemble:
    """Blends strategy signals with regime-dependent weights.

    Every strategy's signal is z-scored across instruments before weighting,
    so a raw percentage return and a z-score contribute on the same scale and
    the weights mean what they look like they mean.
    """

    def __init__(
        self,
        strategies: Dict[str, SignalStrategy],
        regime_detector: GaussianHmmRegimeDetector,
        config: Optional[EnsembleConfig] = None,
    ):
        self.strategies = strategies
        self.regime_detector = regime_detector
        self.config = config or EnsembleConfig()
        self.last_weights: Dict[str, float] = {}
        self.last_probability_of_calm: float = 0.5

    def reset(self) -> None:
        for strategy in self.strategies.values():
            strategy.reset()
        self.regime_detector.reset()
        self.last_weights = {}
        self.last_probability_of_calm = 0.5

    def compute_weights(self, prices_by_day: np.ndarray) -> Dict[str, float]:
        cfg = self.config
        active = {
            name: cfg.baseline_weights.get(name, 0.0)
            for name in self.strategies
            if cfg.baseline_weights.get(name, 0.0) > 0.0
        }
        if not active:
            return {name: 0.0 for name in self.strategies}

        # Skip the HMM when only one strategy is live - it cannot change a
        # single normalised weight, and fitting it is the most expensive thing
        # in the pipeline.
        if len(active) == 1:
            self.last_probability_of_calm = 0.5
            weights = {name: 0.0 for name in self.strategies}
            weights[next(iter(active))] = 1.0
            self.last_weights = weights
            return weights

        probability_of_calm = self.regime_detector.probability_of_calm_regime(prices_by_day)
        self.last_probability_of_calm = probability_of_calm

        weights = {}
        for name, baseline in active.items():
            calm_tilt = cfg.calm_regime_tilt.get(name, 1.0)
            stressed_tilt = cfg.stressed_regime_tilt.get(name, 1.0)
            tilt = probability_of_calm * calm_tilt + (1.0 - probability_of_calm) * stressed_tilt
            weight = baseline * tilt

            if name == "stat_arb" and cfg.stat_arb_quality_scaling:
                # Stat-arb is conditioned on its own health (what fraction of
                # residuals currently pass the half-life screen), not on the
                # market vol regime - it is a relative-value bet, so market
                # direction is the wrong conditioning variable.
                quality = self.strategies[name].quality_score()
                weight *= 0.5 + 0.5 * float(np.clip(quality, 0.0, 1.0))

            weights[name] = max(weight, cfg.minimum_weight_floor * baseline)

        total = sum(weights.values())
        normalised = {name: 0.0 for name in self.strategies}
        if total > 0:
            for name, weight in weights.items():
                normalised[name] = weight / total
        self.last_weights = normalised
        return normalised

    def combined_signal(self, prices_by_day: np.ndarray) -> np.ndarray:
        n_instruments = prices_by_day.shape[1]
        weights = self.compute_weights(prices_by_day)

        non_algo_mask = np.ones(n_instruments, dtype=bool)
        non_algo_mask[ALGO_INDEX] = False

        combined = np.zeros(n_instruments)
        for name, strategy in self.strategies.items():
            weight = weights.get(name, 0.0)
            if weight <= 0.0:
                continue
            raw_signal = np.asarray(strategy.generate_signal(prices_by_day), dtype=float)

            # Defend the book against a strategy misbehaving: a wrong-shaped or
            # non-finite signal contributes nothing rather than poisoning the
            # whole combination.
            if raw_signal.shape != (n_instruments,) or not np.all(np.isfinite(raw_signal)):
                continue

            algo_component = raw_signal[ALGO_INDEX]
            standardised = cross_sectional_zscore(raw_signal, non_algo_mask)
            # A deliberate outright index bet passes through unstandardised;
            # only the cross-section is z-scored.
            standardised[ALGO_INDEX] = algo_component
            combined += weight * standardised

        return combined


# =============================================================================
# Index hedge overlay
# =============================================================================


class IndexHedgeOverlay:
    """Removes the book's net exposure to the market index using ALGO.

    ALGO is not merely correlated with the other 50 instruments - it is an
    *exact* fixed-weight basket of them. Regressing ALGO's price on the other
    50 prices gives R^2 = 0.99999988 with a residual standard deviation of
    $0.0032 on a ~$100 price, which is precisely the $0.0032 predicted by the
    prices being rounded to two decimals and nothing else. Fitting on just 52
    days (one more than the 50 free parameters) still predicts the remaining
    948 days to $0.16, and the weights are stable to 0.65% across five
    disjoint windows - so this is a real structural relation, not an
    overfitted regression.

    Consequences that shape this file:
      * The hedge is exact, not statistical: net index exposure can be removed
        with essentially no tracking error.
      * ALGO costs 0.2 bps against 1 bp elsewhere, so hedging through ALGO is
        5x cheaper than trading the 50 constituents.
      * ALGO adds no *new* idiosyncratic capacity - it is spanned by the other
        50, so a position in it is a pure directional bet on the index. A book
        whose edge is purely relative-value gains nothing from touching it,
        which is why this overlay only pays once strategies carrying real
        index exposure are in the mix.
    """

    def __init__(self, config: Optional[HedgeConfig] = None):
        self.config = config or HedgeConfig()
        self._basket_weights_in_shares = None
        self._days_at_last_fit = None

    def reset(self) -> None:
        self._basket_weights_in_shares = None
        self._days_at_last_fit = None

    def estimate_basket_weights(self, prices_by_day: np.ndarray) -> np.ndarray:
        """Ridge-fit ALGO's price on the other 50 prices.

        Returns length-n_instruments share weights with 0 in the ALGO slot.
        Ridge rather than OLS because the constituent prices are highly
        collinear; the penalty is tiny because the relation is near-exact.
        """
        cfg = self.config
        window_rows = min(cfg.basket_estimation_window, prices_by_day.shape[0])
        window = prices_by_day[-window_rows:]
        n_instruments = prices_by_day.shape[1]

        target = window[:, ALGO_INDEX]
        constituents = np.delete(window, ALGO_INDEX, axis=1)

        model = Ridge(alpha=cfg.ridge_penalty, fit_intercept=True)
        model.fit(constituents, target)

        weights = np.zeros(n_instruments)
        weights[[i for i in range(n_instruments) if i != ALGO_INDEX]] = model.coef_
        return weights

    def _refresh_weights_if_needed(self, prices_by_day: np.ndarray) -> np.ndarray:
        n_days = prices_by_day.shape[0]
        stale = (
            self._basket_weights_in_shares is None
            or self._days_at_last_fit is None
            or n_days < self._days_at_last_fit
            or n_days - self._days_at_last_fit >= 25
        )
        if stale:
            try:
                self._basket_weights_in_shares = self.estimate_basket_weights(prices_by_day)
                self._days_at_last_fit = n_days
            except (np.linalg.LinAlgError, ValueError):
                self._basket_weights_in_shares = np.zeros(prices_by_day.shape[1])
        return self._basket_weights_in_shares

    def apply(
        self, target_dollars_per_instrument: np.ndarray, prices_by_day: np.ndarray
    ) -> np.ndarray:
        """Overwrite the ALGO slot with the offsetting index position."""
        cfg = self.config
        if not cfg.enabled:
            return target_dollars_per_instrument

        basket_weights = self._refresh_weights_if_needed(prices_by_day)
        prices_today = prices_by_day[-1]

        # One ALGO unit is `basket_weights` shares of the constituents, so the
        # book's index-equivalent exposure is the least-squares projection of
        # its share vector onto that basket vector. Dividing share-by-share
        # (shares_i / weight_i) would be ill-conditioned wherever a weight is
        # small; the projection is not.
        shares = target_dollars_per_instrument / np.maximum(prices_today, 1e-9)
        shares[ALGO_INDEX] = 0.0
        basket_norm = float(basket_weights @ basket_weights)
        if basket_norm < 1e-12:
            return target_dollars_per_instrument

        index_equivalent_units = float(shares @ basket_weights) / basket_norm
        hedge_dollars = -cfg.target_hedge_ratio * index_equivalent_units * prices_today[ALGO_INDEX]

        if abs(hedge_dollars) < cfg.min_hedge_dollars:
            hedge_dollars = 0.0

        hedged = target_dollars_per_instrument.copy()
        hedged[ALGO_INDEX] = hedge_dollars
        return hedged


# =============================================================================
# Sizing and risk
# =============================================================================


def convert_signal_to_dollar_positions(
    signal_per_instrument: np.ndarray,
    dollar_position_limits: np.ndarray,
    config: SizingConfig,
) -> np.ndarray:
    """Saturating map from combined signal to target dollar positions.

    `clip(signal * gain, -1, 1)` rather than a soft tanh: validation preferred
    the harder saturation, because the score is linear in mean(PL) while the
    Sharpe factor is scale-invariant, so unused position limit is forgone
    score.
    """
    saturated = np.clip(signal_per_instrument * config.signal_gain, -1.0, 1.0)
    return saturated * dollar_position_limits * config.capital_utilisation


def apply_position_limits(
    target_dollars_per_instrument: np.ndarray,
    prices_today: np.ndarray,
    dollar_position_limits: np.ndarray,
) -> np.ndarray:
    """Clip to the dollar limits and convert to integer share counts.

    Rounds *toward zero* so a rounding error can never push a position over
    the limit and get clipped differently by the evaluator.
    """
    clipped = np.clip(
        target_dollars_per_instrument, -dollar_position_limits, dollar_position_limits
    )
    shares = clipped / np.maximum(prices_today, 1e-9)
    return (np.sign(shares) * np.floor(np.abs(shares))).astype(int)


# =============================================================================
# Portfolio engine
# =============================================================================


class PortfolioEngine:
    """Owns the strategies, the ensemble, the hedge and the sizing.

    Stateful across days (strategy hysteresis, HMM refit cache, basket-weight
    cache), and detects a shortened price history as a restart so the
    validation harness can run independent folds against one engine.
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.strategies: Dict[str, SignalStrategy] = {
            "stat_arb": StatisticalArbitrageStrategy(self.config.stat_arb),
            "mean_reversion": MeanReversionStrategy(self.config.mean_reversion),
            "momentum": MomentumStrategy(self.config.momentum),
        }
        self.regime_detector = GaussianHmmRegimeDetector(self.config.regime)
        self.ensemble = RegimeAwareEnsemble(
            self.strategies, self.regime_detector, self.config.ensemble
        )
        self.hedge_overlay = IndexHedgeOverlay(self.config.hedge)

        self._smoothed_target_dollars = None
        self._previous_target_dollars = None
        self._days_seen = 0

    def reset(self) -> None:
        self.ensemble.reset()
        self.hedge_overlay.reset()
        self._smoothed_target_dollars = None
        self._previous_target_dollars = None
        self._days_seen = 0

    def target_positions(self, prices_by_day: np.ndarray) -> np.ndarray:
        """(n_days, n_inst) prices -> integer share counts for today."""
        cfg = self.config
        n_days, n_instruments = prices_by_day.shape

        if n_days < self._days_seen:
            self.reset()  # history restarted: new fold
        self._days_seen = n_days

        if n_days < cfg.minimum_history_days():
            return np.zeros(n_instruments, dtype=int)

        dollar_position_limits = build_dollar_position_limits(n_instruments)
        prices_today = prices_by_day[-1]

        combined_signal = self.ensemble.combined_signal(prices_by_day)
        target_dollars = convert_signal_to_dollar_positions(
            combined_signal, dollar_position_limits, cfg.sizing
        )
        target_dollars = self.hedge_overlay.apply(target_dollars, prices_by_day)

        if cfg.sizing.position_smoothing_days > 1.0:
            if (
                self._smoothed_target_dollars is None
                or self._smoothed_target_dollars.shape != target_dollars.shape
            ):
                self._smoothed_target_dollars = target_dollars.copy()
            else:
                self._smoothed_target_dollars += (
                    target_dollars - self._smoothed_target_dollars
                ) / cfg.sizing.position_smoothing_days
            target_dollars = self._smoothed_target_dollars.copy()

        # No-trade band: hold the previous target where the move is too small
        # to be worth the commission.
        if (
            cfg.sizing.rebalance_threshold_dollars > 0.0
            and self._previous_target_dollars is not None
            and self._previous_target_dollars.shape == target_dollars.shape
        ):
            too_small = (
                np.abs(target_dollars - self._previous_target_dollars)
                < cfg.sizing.rebalance_threshold_dollars
            )
            target_dollars = np.where(too_small, self._previous_target_dollars, target_dollars)
        self._previous_target_dollars = target_dollars.copy()

        return apply_position_limits(target_dollars, prices_today, dollar_position_limits)


# =============================================================================
# Entry point - the only thing eval.py touches
# =============================================================================

_ENGINE = PortfolioEngine()


def getMyPosition(prcSoFar) -> np.ndarray:
    """Algothon entry point.

    `prcSoFar` arrives from eval.py as **(n_instruments, n_days)**. This is the
    single place in the codebase where that orientation exists: it is
    transposed immediately to (n_days, n_instruments) - days down the rows -
    and everything downstream uses that convention.

    Returns integer share counts, length n_instruments.
    """
    prices_by_day = np.asarray(prcSoFar, dtype=float).T

    if prices_by_day.ndim != 2 or prices_by_day.shape[0] < 2:
        n_instruments = prices_by_day.shape[1] if prices_by_day.ndim == 2 else 0
        return np.zeros(n_instruments, dtype=int)

    try:
        return _ENGINE.target_positions(prices_by_day)
    except Exception:
        # A live run must never crash: an exception here would zero the whole
        # submission. Fall back to flat and keep going.
        return np.zeros(prices_by_day.shape[1], dtype=int)


def reset_engine(config: Optional[PortfolioConfig] = None) -> PortfolioEngine:
    """Rebuild the module-level engine, optionally with a different config.

    Used by the sweep / walk-forward tooling in strategy_lab/ to evaluate many
    configurations in one process.
    """
    global _ENGINE
    _ENGINE = PortfolioEngine(config)
    return _ENGINE
