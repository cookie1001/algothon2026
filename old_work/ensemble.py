"""
ensemble.py - boilerplate for combining mean-reversion / stat-arb / momentum
strategies with regime-dependent weights.

This is deliberately a SKELETON, not a finished implementation - the actual
strategy internals (stat-arb signal, momentum tuning, feature choice for the
HMM) are left as stubs / TODOs for you to build out.

Architecture:
  1. Each Strategy returns a per-instrument SIGNAL (not a final position).
  2. RegimeDetector turns price history into regime probabilities.
  3. Ensemble maps regime probabilities -> strategy weights, normalizes
     each strategy's signal onto a comparable scale, then combines.
  4. Only the combined signal gets sized into positions and clipped to
     limits (reuse size_positions / clip_to_limits from main.py).
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM  # pip install hmmlearn


# ---------------------------------------------------------------------------
# 1. Strategy interface
# ---------------------------------------------------------------------------

class Strategy:
    """Common interface - every strategy returns a raw signal per instrument.
    Sign = direction, magnitude = conviction. Don't worry about sizing here,
    the ensemble normalizes and sizes after combining."""

    def signal(self, prices):
        raise NotImplementedError


class MeanReversionStrategy(Strategy):
    def signal(self, prices, lookback=20):
        window = prices[:, -lookback:]
        mu = window.mean(axis=1)
        sigma = window.std(axis=1)
        sigma[sigma == 0] = 1e-9
        return -(prices[:, -1] - mu) / sigma


class MomentumStrategy(Strategy):
    def signal(self, prices, lookback=40):
        # TODO: this is the crudest possible version (raw past return).
        # Worth investigating: risk-adjusted momentum (return / vol),
        # multiple lookback horizons blended together, etc.
        return prices[:, -1] / prices[:, -lookback] - 1.0


class StatArbStrategy(Strategy):
    def signal(self, prices):
        # TODO: PCA residual signal or cointegrated-pair spread z-score.
        # Whatever you build, keep the output as one signal value per
        # instrument (residual mean-reversion signal from the PCA factor
        # model works naturally here - the pairs version needs to be
        # aggregated per-instrument if an instrument is in multiple pairs).
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. Regime detection -> probabilities, not a hard label
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Fits a 2-state Gaussian HMM on simple volatility-related features and
    exposes P(low-vol regime) / P(high-vol regime) for the most recent day.

    TODO: the feature choice here is a starting point, not a final answer -
    worth experimenting with what actually separates your two regimes best
    (portfolio-level realized vol, cross-sectional dispersion, average
    absolute return, etc.)
    """

    def __init__(self, n_regimes=2, lookback=250):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = None
        self.low_vol_state = None

    def _features(self, prices):
        rets = np.diff(prices, axis=1) / prices[:, :-1]
        port_ret = rets.mean(axis=0)  # portfolio-level daily return
        return np.column_stack([port_ret, np.abs(port_ret)])

    def fit(self, prices):
        feat = self._features(prices[:, -self.lookback:])
        self.model = GaussianHMM(n_components=self.n_regimes,
                                  covariance_type="diag", n_iter=100)
        self.model.fit(feat)
        variances = self.model.covars_[:, 0, 0]
        self.low_vol_state = int(np.argmin(variances))
        return self

    def regime_probs(self, prices):
        feat = self._features(prices[:, -self.lookback:])
        posteriors = self.model.predict_proba(feat)
        p_low = posteriors[-1, self.low_vol_state]
        return p_low, 1.0 - p_low


# ---------------------------------------------------------------------------
# 3. Ensemble: regime probabilities -> weights -> combined signal
# ---------------------------------------------------------------------------

class Ensemble:
    def __init__(self, strategies: dict, regime_detector: RegimeDetector):
        self.strategies = strategies
        self.regime_detector = regime_detector

    def weights(self, prices):
        p_low, p_high = self.regime_detector.regime_probs(prices)
        # TODO: this is the actual design decision you need to make - where
        # does stat-arb sit? It's not obviously "low vol" or "high vol"
        # since it's about relative pricing, not absolute trend. Options:
        #   - constant baseline weight regardless of regime
        #   - its own regime signal (e.g. spread-specific vol/dispersion)
        #   - tied to low-vol like a second mean-reversion strategy
        w = {
            "meanrev": p_low,
            "momentum": p_high,
            "statarb": 0.5,  # placeholder
        }
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    @staticmethod
    def _normalize(signal):
        # TODO: strategies produce signals on very different scales
        # (a z-score vs. a raw percentage return) - they need to be put on
        # a comparable footing before weighted-summing, otherwise the
        # weights won't mean what you think they mean. Simplest fix: turn
        # every signal into a z-score of itself across instruments.
        mu, sigma = signal.mean(), signal.std()
        return (signal - mu) / sigma if sigma > 0 else np.zeros_like(signal)

    def combined_signal(self, prices):
        w = self.weights(prices)
        combined = np.zeros(prices.shape[0])
        for name, strat in self.strategies.items():
            combined += w[name] * self._normalize(strat.signal(prices))
        return combined
