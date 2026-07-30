
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

# hmmlearn and statsmodels are imported lazily inside the functions that use
# them, so an install problem degrades one component instead of preventing the
# module from importing at all - which would zero the whole submission.

# Market constants

ALGO_INDEX = 0  

ALGO_DOLLAR_POSITION_LIMIT = 100_000.0
DEFAULT_DOLLAR_POSITION_LIMIT = 10_000.0

ALGO_COMMISSION_RATE = 0.2 / 10_000.0  # 0.2 basis points
DEFAULT_COMMISSION_RATE = 1.0 / 10_000.0  # 1 basis point

TRADING_DAYS_PER_YEAR = 250  # matches eval.py's sqrt(250) Sharpe annualisation


def build_dollar_position_limits(n_instruments: int) -> np.ndarray:
    limits = np.full(n_instruments, DEFAULT_DOLLAR_POSITION_LIMIT)
    limits[ALGO_INDEX] = ALGO_DOLLAR_POSITION_LIMIT
    return limits


def build_commission_rates(n_instruments: int) -> np.ndarray:
    rates = np.full(n_instruments, DEFAULT_COMMISSION_RATE)
    rates[ALGO_INDEX] = ALGO_COMMISSION_RATE
    return rates

# Configuration

@dataclass
class StatArbConfig:

    volatility_span: int = 60

    factor_model: str = "pca"

    factor_fit_window: int = 252

    n_factors: int = 30

    reversion_window: int = 20

    zscore_method: str = "cross_sectional"

    require_stationary_residual: bool = False
    adf_pvalue_threshold: float = 0.05

    min_half_life: float = 1.0
    max_half_life: float = 25.0

    entry_zscore: float = 0.0
    exit_zscore: float = 0.0

    max_abs_zscore: float = 3.0


@dataclass
class MeanReversionConfig:

    lookback_windows: tuple = (3, 5)

    volatility_span: int = 60
    entry_zscore: float = 1.0
    exit_zscore: float = 0.5
    max_abs_zscore: float = 3.0
    beta_hedge_against_algo: bool = True
    beta_estimation_window: int = 200


@dataclass
class MomentumConfig:

    lookback_windows: tuple = (60, 120)

    volatility_span: int = 60
    skip_recent_days: int = 1

    cross_sectional: bool = True

    trend_gate_enabled: bool = True
    min_variance_ratio: float = 1.1
    variance_ratio_window: int = 120
    variance_ratio_lag: int = 5

    max_abs_zscore: float = 3.0


@dataclass
class RegimeConfig:

    n_regimes: int = 2
    feature_window: int = 250

    volatility_window: int = 15

    refit_interval_days: int = 25

    n_em_iterations: int = 40

    n_restarts: int = 5

    min_covar: float = 1e-3

    random_seed: int = 0


@dataclass
class EnsembleConfig:

    baseline_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "stat_arb": 1.0,
            "mean_reversion": 0.1,
            "momentum": 0.0,
        }
    )

    minimum_weight_floor: float = 0.05

    calm_regime_tilt: Dict[str, float] = field(
        default_factory=lambda: {
            "stat_arb": 1.0,
            "mean_reversion": 1.3,
            "momentum": 0.7,
        }
    )

    stressed_regime_tilt: Dict[str, float] = field(
        default_factory=lambda: {
            "stat_arb": 1.0,
            "mean_reversion": 0.7,
            "momentum": 1.3,
        }
    )

    stat_arb_quality_scaling: bool = True


@dataclass
class HedgeConfig:

    enabled: bool = True

    target_hedge_ratio: float = 1.0

    beta_estimation_window: int = 252

    min_hedge_dollars: float = 2_000.0


@dataclass
class SizingConfig:

    signal_gain: float = 8.0

    capital_utilisation: float = 1.0

    rebalance_threshold_dollars: float = 1000.0

    position_smoothing_days: float = 1.0

    volatility_target_enabled: bool = True

    volatility_lookback_days: int = 10

    volatility_scale_floor: float = 0.3

    volatility_scale_ceiling: float = 1.5


@dataclass
class PortfolioConfig:

    stat_arb: StatArbConfig = field(default_factory=StatArbConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)

    def minimum_history_days(self) -> int:
        return max(
            self.stat_arb.reversion_window + 30,
            max(self.mean_reversion.lookback_windows)
            + self.mean_reversion.beta_estimation_window
            + 5,
            self.momentum.variance_ratio_window + 5,
            60,
        ) + 2



# Shared numerical helpers 
def compute_log_returns(prices_by_day: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices_by_day), axis=0)


def volatility_standardised_returns(
    returns_by_day: np.ndarray, span: int
) -> np.ndarray:
    frame = pd.DataFrame(returns_by_day)
    volatility = frame.ewm(span=span, min_periods=2).std()
    volatility = volatility.bfill().fillna(1.0).clip(lower=1e-10)
    return (frame / volatility).to_numpy()


def cross_sectional_zscore(
    values_per_instrument: np.ndarray, valid_mask: Optional[np.ndarray] = None
) -> np.ndarray:
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


def compute_volatility_scale(
    prices_by_day: np.ndarray,
    lookback_days: int,
    floor: float,
    ceiling: float,
) -> float:
    returns_by_day = compute_log_returns(prices_by_day)
    algo_returns = returns_by_day[:, ALGO_INDEX]
    if len(algo_returns) < lookback_days + 20:
        return 1.0

    recent_vol = algo_returns[-lookback_days:].std()
    typical_vol = algo_returns.std()
    if typical_vol < 1e-12 or recent_vol < 1e-12:
        return 1.0

    return float(np.clip(typical_vol / recent_vol, floor, ceiling))


# Strategy interface

class SignalStrategy:

    name: str = "unnamed"

    def generate_signal(self, prices_by_day: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def quality_score(self) -> float:
        return 1.0


def _apply_entry_exit_hysteresis(
    signal_per_instrument: np.ndarray,
    open_position_sign: np.ndarray,
    entry_zscore: float,
    exit_zscore: float,
):
    if entry_zscore <= 0.0 and exit_zscore <= 0.0:
        return signal_per_instrument, np.sign(signal_per_instrument)

    magnitude = np.abs(signal_per_instrument)
    currently_open = open_position_sign != 0.0
    active = ((~currently_open) & (magnitude >= entry_zscore)) | (
        currently_open & (magnitude >= exit_zscore)
    )
    held = np.where(active, signal_per_instrument, 0.0)
    return held, np.sign(held)


# Strategy 1 - PCA-residual statistical arbitrage

class StatisticalArbitrageStrategy(SignalStrategy):

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


# Strategy 2 - per-instrument mean reversion  

class MeanReversionStrategy(SignalStrategy):

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

        if n_days < max(cfg.lookback_windows) + 25:
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

        # Average the displacement across several horizons rather than betting
        # on one. Cumulative standardised return over a lookback is the
        # displacement from the rolling mean in volatility units; dividing by
        # sqrt(window) turns it into a t-statistic, which is what makes
        # different horizons commensurable enough to average in the first
        # place (and keeps entry/exit thresholds meaning the same thing at
        # every horizon).
        displacement = np.zeros(n_instruments)
        horizons_used = 0
        for lookback in cfg.lookback_windows:
            if standardised.shape[0] < lookback:
                continue
            displacement += standardised[-lookback:].sum(axis=0) / np.sqrt(lookback)
            horizons_used += 1
        if horizons_used == 0:
            return np.zeros(n_instruments)
        displacement /= horizons_used

        signal_per_instrument = -np.clip(
            np.nan_to_num(displacement), -cfg.max_abs_zscore, cfg.max_abs_zscore
        )
        signal_per_instrument[ALGO_INDEX] = 0.0

        held, self._open_position_sign = _apply_entry_exit_hysteresis(
            signal_per_instrument, self._open_position_sign, cfg.entry_zscore, cfg.exit_zscore
        )
        return held



# Strategy 3 - momentum  

class MomentumStrategy(SignalStrategy):

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


# Regime detection - hmmlearn

class GaussianHmmRegimeDetector:

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.model = None
        self.calm_state_index = None
        self._feature_mean = None
        self._feature_std = None
        self._days_at_last_fit = None

    def reset(self) -> None:
        self.model = None
        self.calm_state_index = None
        self._feature_mean = None
        self._feature_std = None
        self._days_at_last_fit = None

    def _build_raw_features(self, prices_by_day: np.ndarray) -> Optional[np.ndarray]:
        cfg = self.config
        window = prices_by_day[-(cfg.feature_window + 1) :]
        returns_by_day = compute_log_returns(window)
        market_return = returns_by_day.mean(axis=1)

        volatility_window = cfg.volatility_window
        if len(market_return) < volatility_window:
            return None

        # Rolling std via a strided index matrix rather than a Python loop.
        # This runs on every single day (not just refit days), and the loop
        # version measured 2.18 ms/call against 0.13 ms here - 16x, for
        # byte-identical output.
        n_windows = len(market_return) - volatility_window + 1
        offsets = np.arange(n_windows)[:, None] + np.arange(volatility_window)[None, :]
        rolling_volatility = market_return[offsets].std(axis=1)

        absolute_return = np.abs(market_return[volatility_window - 1 :])
        return np.column_stack([absolute_return, rolling_volatility])

    def _standardise(self, raw_features: np.ndarray) -> np.ndarray:
        return (raw_features - self._feature_mean) / self._feature_std

    def fit(self, prices_by_day: np.ndarray) -> "GaussianHmmRegimeDetector":
        from hmmlearn.hmm import GaussianHMM

        cfg = self.config
        raw_features = self._build_raw_features(prices_by_day)
        if raw_features is None or len(raw_features) < 30:
            raise ValueError("not enough history to fit the regime HMM")

        # Standardise before fitting. hmmlearn's `min_covar` floor sits far
        # above the raw variance of these features (~1e-5 and below), so
        # fitting unstandardised would clamp both states' covariances to
        # essentially the same value and the two "regimes" would be
        # statistically indistinguishable no matter what EM found.
        feature_mean = raw_features.mean(axis=0)
        feature_std = np.maximum(raw_features.std(axis=0), 1e-10)
        standardised = (raw_features - feature_mean) / feature_std

        # Restart Baum-Welch from several seeds and keep the best converged
        # run - a single attempt can settle into a degenerate optimum, which
        # yields confident but meaningless regime probabilities.
        best_model, best_log_likelihood = None, -np.inf
        for restart in range(max(cfg.n_restarts, 1)):
            candidate = GaussianHMM(
                n_components=cfg.n_regimes,
                covariance_type="diag",
                n_iter=cfg.n_em_iterations,
                tol=1e-4,
                random_state=cfg.random_seed + restart,
                min_covar=cfg.min_covar,
            )
            try:
                candidate.fit(standardised)
                if not candidate.monitor_.converged:
                    continue
                log_likelihood = candidate.score(standardised)
            except Exception:
                continue
            if log_likelihood > best_log_likelihood:
                best_log_likelihood, best_model = log_likelihood, candidate

        if best_model is None:
            raise ValueError("no HMM restart converged")

        self.model = best_model
        self._feature_mean = feature_mean
        self._feature_std = feature_std
        # The calm state is the one with the lowest mean on the rolling
        # volatility feature. Labelling by a feature mean (not by fitted
        # variance) keeps the label stable across refits, so the weights do
        # not flip sign just because EM relabelled the states.
        self.calm_state_index = int(np.argmin(best_model.means_[:, 1]))
        return self

    def probability_of_calm_regime(self, prices_by_day: np.ndarray) -> float:
        cfg = self.config
        n_days = prices_by_day.shape[0]
        needs_refit = (
            self._days_at_last_fit is None
            or n_days < self._days_at_last_fit  # history restarted (new fold)
            or n_days - self._days_at_last_fit >= cfg.refit_interval_days
        )
        if needs_refit:
            # Record the ATTEMPT, not the success. If fitting keeps failing,
            # this backs off to the normal cadence instead of paying for a
            # doomed fit every single day for the rest of the run.
            self._days_at_last_fit = n_days
            try:
                self.fit(prices_by_day)
            except Exception:
                pass  # keep the last good model, if any

        if self.model is None:
            return 0.5  # never successfully fit - genuinely no opinion

        try:
            raw_features = self._build_raw_features(prices_by_day)
            if raw_features is None:
                return 0.5
            posteriors = self.model.predict_proba(self._standardise(raw_features))
            probability = float(posteriors[-1, self.calm_state_index])
        except Exception:
            return 0.5
        return probability if np.isfinite(probability) else 0.5


# Ensemble

class RegimeAwareEnsemble:

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

# Index hedge overlay

class IndexHedgeOverlay:

    def __init__(self, config: Optional[HedgeConfig] = None):
        self.config = config or HedgeConfig()

    def reset(self) -> None:
        pass

    def estimate_return_betas(self, prices_by_day: np.ndarray) -> np.ndarray:
        cfg = self.config
        rows = min(cfg.beta_estimation_window, prices_by_day.shape[0] - 1)
        if rows < 10:
            return np.zeros(prices_by_day.shape[1])

        returns_by_day = compute_log_returns(prices_by_day[-(rows + 1) :])
        algo_returns = returns_by_day[:, ALGO_INDEX]
        algo_variance = float(algo_returns @ algo_returns)
        if algo_variance < 1e-14:
            return np.zeros(prices_by_day.shape[1])

        betas = (algo_returns @ returns_by_day) / algo_variance
        betas[ALGO_INDEX] = 0.0
        return betas

    def apply(
        self, target_dollars_per_instrument: np.ndarray, prices_by_day: np.ndarray
    ) -> np.ndarray:
        cfg = self.config
        if not cfg.enabled:
            return target_dollars_per_instrument

        betas = self.estimate_return_betas(prices_by_day)
        book_dollar_exposure_to_algo = float(target_dollars_per_instrument @ betas)
        hedge_dollars = -cfg.target_hedge_ratio * book_dollar_exposure_to_algo

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
    saturated = np.clip(signal_per_instrument * config.signal_gain, -1.0, 1.0)
    return saturated * dollar_position_limits * config.capital_utilisation


def apply_position_limits(
    target_dollars_per_instrument: np.ndarray,
    prices_today: np.ndarray,
    dollar_position_limits: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        target_dollars_per_instrument, -dollar_position_limits, dollar_position_limits
    )
    shares = clipped / np.maximum(prices_today, 1e-9)
    return (np.sign(shares) * np.floor(np.abs(shares))).astype(int)


# =============================================================================
# Portfolio engine
# =============================================================================


class PortfolioEngine:

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

        if cfg.sizing.volatility_target_enabled:
            scale = compute_volatility_scale(
                prices_by_day,
                cfg.sizing.volatility_lookback_days,
                cfg.sizing.volatility_scale_floor,
                cfg.sizing.volatility_scale_ceiling,
            )
            target_dollars = target_dollars * scale

        # Hedge AFTER scaling: it should offset whatever exposure the (possibly
        # scaled-down) book actually carries, not the pre-scale signal.
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


# Entry point - the only thing eval.py touches

_ENGINE = PortfolioEngine()


def getMyPosition(prcSoFar) -> np.ndarray:
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
    global _ENGINE
    _ENGINE = PortfolioEngine(config)
    return _ENGINE
