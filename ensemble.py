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
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ---------------------------------------------------------------------------
# 1. Strategy interface
# ---------------------------------------------------------------------------

class Strategy:
    """Common interface - every strategy returns a raw signal per instrument.
    Sign = direction, magnitude = conviction. Don't worry about sizing here,
    the ensemble normalizes and sizes after combining."""

    def signal(self, prices):
        raise NotImplementedError


class MeanReversion(Strategy):
    """
    Per-instrument mean reversion.

    Pipeline:
      1. (Optional) beta-hedge each instrument against an equal-weighted
         market proxy built from the instrument universe, to isolate the
         idiosyncratic (market-uncorrelated) component before testing for
         mean reversion. Off by default -- turn on with beta_hedge=True.
         NOTE: if StatArb ends up doing its own residual extraction
         (extract_residuals / standardise_returns), talk to whoever owns
         that class about whether it's worth sharing one hedge instead of
         computing it twice -- this implementation is self-contained and
         doesn't depend on StatArb.
      2. Periodically re-screen every instrument's series for mean
         reversion via ADF and/or Hurst, over a trailing lookback window.
         Re-run every `screen_step` days (not just once) so the
         mean-reverting set is allowed to change over time.
      3. For instruments currently flagged mean-reverting, estimate a
         half-life via an OU fit, and use it (scaled by window_mult) as
         the lookback for a z-score -- rather than an arbitrary fixed
         window.
      4. Apply separate entry/exit z-thresholds (hysteresis): enter a
         position when |z| crosses entry_z, hold it until |z| falls back
         inside exit_z. This state persists across days (signal() is
         called once per day in order by eval.py/backtester.py).

    signal() output IS the z-score itself (sign-flipped so the direction
    means "bet on reversion": positive signal = spread is far below its
    mean, i.e. go long; negative = far above, i.e. go short), for whichever
    instruments are currently in a flagged position. Instruments not
    currently in a position (either never triggered entry, or exited)
    get 0. The ensemble is responsible for any further scaling/clipping.
    """

    def __init__(self, beta_hedge=False, beta_window=60,
                 screen_lookback=120, screen_step=20,
                 adf_thresh=0.05, hurst_thresh=0.45, combine="and",
                 entry_z=2.0, exit_z=0.5, stop_z=4.0,
                 hold_mult=3.0, min_hold=10, max_hold=60,
                 window_mult=2.0, min_window=5, max_window=252):
        if combine not in ("and", "or"):
            raise ValueError("combine must be 'and' or 'or'")
        if not (exit_z < entry_z < stop_z):
            raise ValueError("require exit_z < entry_z < stop_z")
        self.beta_hedge = beta_hedge
        self.beta_window = beta_window
        self.screen_lookback = screen_lookback
        self.screen_step = screen_step
        self.adf_thresh = adf_thresh
        self.hurst_thresh = hurst_thresh
        self.combine = combine    # 'and' = both ADF and Hurst must pass (stricter,
                                   # fewer false positives). 'or' = either passing
                                   # is enough (catches more true mean-reversion,
                                   # but more false positives too). See sweep
                                   # results in the team notes before picking --
                                   # this changed the backtest Sharpe a lot on
                                   # real prices.txt, in both directions depending
                                   # on beta_hedge, so it's a real tradeoff, not
                                   # a strictly-better setting.
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z          # |z| this extreme means the spread kept
                                       # moving AGAINST the reversion thesis well
                                       # past entry -- force-exit regardless of
                                       # half-life expectations (protects against
                                       # a genuine regime break, not just noise)
        self.hold_mult = hold_mult    # max holding period = hold_mult * half_life,
        self.min_hold = min_hold      # clipped to [min_hold, max_hold] days --
        self.max_hold = max_hold      # prevents holding a position indefinitely
                                       # if it never reverts and never stops out
        self.window_mult = window_mult
        self.min_window = min_window
        self.max_window = max_window

        # persisted across calls (signal() is called once per day, in order)
        self._last_screen_t = -10 ** 9
        self._mr_mask = None      # bool array, per instrument
        self._half_life = None    # float array, per instrument
        self._state = None        # -1 / 0 / +1 per instrument, hysteresis state
        self._days_held = None    # int array, days since entry, per instrument

        # public diagnostics, refreshed every call -- separate from the
        # trade signal (which is 0 unless a position is actually active).
        # RegimeDetector / teammates can read these for a regime feature
        # like "how many instruments currently look mean-reverting" or
        # "average |z| across the flagged set", without depending on
        # whether MeanReversion has actually entered a trade on them.
        self.last_mr_mask = None       # bool array, per instrument (this call)
        self.last_zscores = None       # float array, NaN where not computed
        self.last_stopped_out = None   # bool array: exited via stop_z today
        self.last_timed_out = None     # bool array: exited via max hold today

    # -- screening helpers ---------------------------------------------

    @staticmethod
    def _hurst_exponent(series: np.ndarray, max_lag: int = 50) -> float:
        """Variance-of-differences Hurst estimate. H < 0.5 => mean reverting."""
        series = series[~np.isnan(series)]
        if len(series) < max_lag * 2:
            return np.nan
        lags = range(2, max_lag)
        tau = np.array([np.std(series[lag:] - series[:-lag]) for lag in lags])
        valid = tau > 0
        if valid.sum() < 2:
            return np.nan
        # tau = std(diff) ~ lag^H, so the log-log slope IS H directly
        return np.polyfit(np.log(list(lags))[valid], np.log(tau[valid]), 1)[0]

    @staticmethod
    def _adf_pvalue(series: pd.Series) -> float:
        s = series.dropna()
        if len(s) < 20:
            return np.nan
        try:
            return adfuller(s, autolag="AIC")[1]
        except Exception:
            return np.nan

    @staticmethod
    def _estimate_half_life(series: pd.Series) -> float:
        """OU fit: delta_t = a + b*series_{t-1}; half-life = -ln(2)/b.
        NaN if the fit doesn't imply mean reversion (b >= 0)."""
        s = series.dropna()
        if len(s) < 20:
            return np.nan
        lagged = s.shift(1).dropna()
        delta = (s - s.shift(1)).dropna()
        lagged, delta = lagged.align(delta, join="inner")
        if len(lagged) < 10:
            return np.nan
        b = OLS(delta.values, add_constant(lagged.values)).fit().params[1]
        return -np.log(2) / b if b < 0 else np.nan

    def _screen(self, series_df: pd.DataFrame):
        """ADF + Hurst over the trailing screen_lookback window, per column.
        Returns (mr_mask: bool array, half_life: float array)."""
        window_data = series_df.tail(self.screen_lookback)
        n = series_df.shape[1]
        mr_mask = np.zeros(n, dtype=bool)
        half_life = np.full(n, np.nan)
        for j, col in enumerate(series_df.columns):
            s = window_data[col]
            p = self._adf_pvalue(s)
            h = self._hurst_exponent(s.values)
            adf_ok = (not np.isnan(p)) and p < self.adf_thresh
            hurst_ok = (not np.isnan(h)) and h < self.hurst_thresh
            is_mr = (adf_ok and hurst_ok) if self.combine == "and" else (adf_ok or hurst_ok)
            mr_mask[j] = is_mr
            if is_mr:
                half_life[j] = self._estimate_half_life(s)
        return mr_mask, half_life

    # -- optional beta hedge ---------------------------------------------

    def _beta_hedge_all(self, log_px: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling-beta hedge against a leave-one-out equal-weighted market
        proxy built from the instrument universe (no external benchmark is
        provided by the eval harness). "Leave-one-out" matters here: with
        only ~51 instruments, each one is ~2% of a naive equal-weighted
        average, which mechanically correlates every instrument with the
        very benchmark it's being hedged against. Excluding instrument i
        from its own proxy avoids that.

        Done in RETURN space and then integrated into a synthetic
        idiosyncratic price series per instrument:

            r_hedged_i,t = r_i,t - beta_i,t * r_market(-i),t
            spread_i,t   = cumsum(r_hedged_i,t)

        IMPORTANT: this must be done in return space, not by subtracting
        beta_t * price_level directly. beta_t is a noisy rolling estimate,
        and log-price levels carry the market's full random-walk
        magnitude -- multiplying a noisy, time-varying beta against a
        large wandering quantity injects spurious non-stationary noise
        into the spread even when the true idiosyncratic process is
        genuinely mean reverting (verified this the hard way on synthetic
        data before landing on the return-space version).
        """
        n = log_px.shape[1]
        total = log_px.sum(axis=1)
        r = log_px.diff()
        hedged_return = {}
        for col in log_px.columns:
            market_i = (total - log_px[col]) / (n - 1)  # leave instrument i out
            r_mkt_i = market_i.diff()
            beta_i = r[col].rolling(self.beta_window).cov(r_mkt_i) / r_mkt_i.rolling(self.beta_window).var()
            hedged_return[col] = r[col] - beta_i * r_mkt_i
        return pd.DataFrame(hedged_return).cumsum()

    # -- main entry point ---------------------------------------------

    def signal(self, prices: np.ndarray) -> np.ndarray:
        nInst, t = prices.shape
        if self._state is None:
            self._state = np.zeros(nInst)
            self._days_held = np.zeros(nInst, dtype=int)

        min_hist = self.screen_lookback
        if self.beta_hedge:
            min_hist = max(min_hist, self.beta_window) + 5
        if t < min_hist:
            self.last_mr_mask = np.zeros(nInst, dtype=bool)
            self.last_zscores = np.full(nInst, np.nan)
            self.last_stopped_out = np.zeros(nInst, dtype=bool)
            self.last_timed_out = np.zeros(nInst, dtype=bool)
            return np.zeros(nInst)

        log_px = pd.DataFrame(np.log(prices).T)  # index=day, columns=instrument
        series_df = self._beta_hedge_all(log_px) if self.beta_hedge else log_px

        if t - self._last_screen_t >= self.screen_step or self._mr_mask is None:
            self._mr_mask, self._half_life = self._screen(series_df)
            self._last_screen_t = t

        signal = np.zeros(nInst)
        zscores = np.full(nInst, np.nan)
        stopped_out = np.zeros(nInst, dtype=bool)
        timed_out = np.zeros(nInst, dtype=bool)
        band = self.entry_z - self.exit_z  # guaranteed > 0 by the constructor check

        for j, col in enumerate(series_df.columns):
            if not self._mr_mask[j]:
                self._state[j] = 0.0
                self._days_held[j] = 0
                continue

            hl = self._half_life[j]
            window = (int(np.clip(self.window_mult * hl, self.min_window, self.max_window))
                      if not np.isnan(hl) else self.max_window)
            s = series_df[col].dropna()
            if len(s) < window + 1:
                continue
            recent = s.tail(window)
            std = recent.std()
            if std == 0 or np.isnan(std):
                continue
            z = (s.iloc[-1] - recent.mean()) / std
            zscores[j] = z

            if self._state[j] == 0:
                if z <= -self.entry_z:
                    self._state[j] = 1
                    self._days_held[j] = 0
                elif z >= self.entry_z:
                    self._state[j] = -1
                    self._days_held[j] = 0
            else:
                self._days_held[j] += 1

                # stop-loss: the spread kept moving AGAINST the reversion
                # thesis well past entry -- treat as evidence the thesis
                # may be broken (regime shift), not just noise, and exit
                # regardless of the half-life-based window/exit logic.
                if abs(z) >= self.stop_z:
                    stopped_out[j] = True
                    self._state[j] = 0
                    self._days_held[j] = 0

                # max holding period: scale with the estimated half-life
                # (a position expected to revert in ~5 days shouldn't
                # still be open after 200), but clip to sane absolute
                # bounds. Prevents holding forever if price drifts
                # sideways near the entry threshold without ever
                # completing a reversion OR triggering the stop.
                else:
                    max_days = int(np.clip(self.hold_mult * hl, self.min_hold, self.max_hold)) \
                        if not np.isnan(hl) else self.max_hold
                    if self._days_held[j] >= max_days:
                        timed_out[j] = True
                        self._state[j] = 0
                        self._days_held[j] = 0
                    # normal exit: reverted back inside the exit band
                    elif self._state[j] == 1 and z >= -self.exit_z:
                        self._state[j] = 0
                        self._days_held[j] = 0
                    elif self._state[j] == -1 and z <= self.exit_z:
                        self._state[j] = 0
                        self._days_held[j] = 0

            if self._state[j] != 0:
                # conviction: 0 right at the exit band, ramps to 1 by the
                # time |z| reaches entry_z, held at 1 beyond that (up to
                # stop_z, where the position is closed above instead) --
                # bounded and tied to your actual thresholds, rather than
                # scaling directly (and unboundedly) with raw z.
                conviction = np.clip((abs(z) - self.exit_z) / band, 0.0, 1.0)
                signal[j] = self._state[j] * conviction

        self.last_mr_mask = self._mr_mask.copy()
        self.last_zscores = zscores
        self.last_stopped_out = stopped_out
        self.last_timed_out = timed_out
        return signal


class Momentum():
   None
   
class StatArb():
   def __init__(self):
      None
   
   def standardise_returns():
      None
      
   def extract_residuals():
      None
   
   def analyze_ou_process():
      None
      
   def signal():
      None
   
class RegimeDetector():
   None

class Ensemble():
   None
