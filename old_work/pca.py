import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller


class StatArb:
  def __init__(self, vol_span, k_components, num_assets = 51, algo_idx = 0, fit_window = 252, mr_window = 60,
               max_half_life = 10, min_half_life = 1, z_entry = 1.75, z_exit = 0.75,
               z_stop_multiplier = 1.5, time_stop_multiplier = 3) :   # >>> NEW: two new stop params
    # Initial variables, and hardcoded variables
    self.vol_span = vol_span
    self.k_components = k_components
    self.num_assets = num_assets
    self.algo_idx = algo_idx
    self.fit_window = fit_window
    self.mr_window = mr_window
    self.max_half_life = max_half_life 
    self.min_half_life = min_half_life
    self.z_entry = z_entry
    self.z_exit = z_exit
    self.z_stop_multiplier = z_stop_multiplier          # >>> NEW: hard stop - exit if |z| grows past entry |z| * this
    self.time_stop_multiplier = time_stop_multiplier    # >>> NEW: time stop - exit if held > this * half_life at entry
    self.signals = np.zeros(self.num_assets)
    self.days_held = np.zeros(self.num_assets)          # >>> NEW: consecutive days each position has been open
    self.entry_z = np.zeros(self.num_assets)            # >>> NEW: z-score at the moment a position was opened
    self.entry_half_life = np.zeros(self.num_assets)    # >>> NEW: half_life at the moment a position was opened
    
  def _standardise_returns(self, returns_df):
    rolling_std = returns_df.ewm(span = self.vol_span).std()
    rolling_std = rolling_std.replace(0, np.nan).fillna(method = 'bfill')
    std_returns = returns_df / rolling_std
    
    return std_returns
      
  def _extract_residuals(self, std_returns):
    # Train PCA
    pca = PCA(n_components=self.k_components)
    factors = pca.fit_transform(std_returns)
    
    # Construct systematic returns
    systematic_returns = pd.DataFrame(
      pca.inverse_transform(factors),
      index = std_returns.index,
      columns = std_returns.columns
    )
    
    daily_residuals = std_returns - systematic_returns
    
    return daily_residuals.cumsum()
  
  def _analyse_ou_process(self, series):
    # Use only the last 60 days
    series = series.iloc[-self.mr_window:].dropna()
    
    # Ensure minimum length 
    if len(series)< max(20, self.mr_window // 2):
      return None # not enough data to trust the fit
    
    y = series.values[1:]
    x = series.values[:-1]
    
    # Fit AR(1): X_t = a + b * X_{t-1} + zeta
    b, a = np.polyfit(x, y, 1) # linear regression on residuals
    
    # Discard explosive or random walk processes
    if b <= 0 or b >= 1:
      return None # explosive / unit-root / oscillating - discard
    
    half_life = -np.log(2) / np.log(b)
    
    # Stationarity test (adfuller)
    adf_stat , p_value, _, _, _, _ = adfuller(series)
    is_stationary = p_value < 0.05
    
    # Z-score calculation
    zeta = y - (a + b * x)
    zeta_var = np.var(zeta)
    eq_var = zeta_var / (1 - b**2)
    eq_mean = a / (1 - b)
    
    current_z = (series.iloc[-1] - eq_mean) / np.sqrt(eq_var)
    
    return {
      'half_life': half_life,
      'p_value': p_value,
      'is_stationary': is_stationary,
      'eq_mean': eq_mean,
      'eq_std': np.sqrt(eq_var),
      'current_z': current_z
    }

  def _close_position(self, global_i):   # >>> NEW: shared helper so every exit path resets state identically
    self.signals[global_i] = 0
    self.days_held[global_i] = 0
    self.entry_z[global_i] = 0
    self.entry_half_life[global_i] = 0
    
  def signal(self, prices):
    """ 
      prices: (nInst, nDays) array. Returns a length-nInst signal array - 0 for ALGO (excluded from factor universe), 
      0 for instruments with insufficient data or a residual that fails the stationarity / half-life screen,
      0 for instruments whose open position has reverted back inside the z_exit band, hit its hard stop-loss,
      exceeded its time stop, or whose residual has broken regime (all used as exit triggers), and a continuous
      conviction score (sign-preserving, scaled by distance past z_entry) for everything else. self.signals
      persists between calls, so it doubles as the current position vector: a nonzero entry means that asset
      currently has an open position.
    """     
    
    n_inst, n_days = prices.shape
    
    if n_days < self.fit_window + 1:
      return self.signals # guard for early days
    
    other_idx = [i for i in range(n_inst) if i != self.algo_idx] # return other indicies
    window = prices[other_idx][:, -self.fit_window:]
    
    # Preserve all asset names in data set
    col_labels = [str(i) for i in range(len(other_idx))]
    returns_df = pd.DataFrame(window.T, columns = col_labels).pct_change().dropna()
    
    # Standardise returns and get residuals
    std_returns = self._standardise_returns(returns_df)
    cum_residuals = self._extract_residuals(std_returns)
    
    # Get stationarity statistics of residuals
    for local_i, global_i in enumerate(other_idx):
      col = str(local_i)
      in_position = self.signals[global_i] != 0

      if col not in cum_residuals.columns:
        if in_position:
          self._close_position(global_i) # exit: lost the ability to price this residual at all
        continue
      
      stats = self._analyse_ou_process(cum_residuals[col])
      if stats is None:
        if in_position:
          self._close_position(global_i) # exit: OU fit broke down (explosive / unit-root / too little data)
        continue

      if not (stats["is_stationary"]
              and self.min_half_life <= stats["half_life"] <= self.max_half_life):
        if in_position:
          self._close_position(global_i) # exit: residual no longer passes the stationarity / half-life screen
        continue
      
      # Get z scores
      z = stats["current_z"]

      if in_position:
        self.days_held[global_i] += 1 # >>> NEW: one more day survived in this trade

        # >>> NEW: hard stop-loss - spread has diverged further past where we entered, cut it
        if abs(z) > abs(self.entry_z[global_i]) * self.z_stop_multiplier:
          self._close_position(global_i)
          continue

        # >>> NEW: time stop - held too long relative to this residual's half-life at entry, cut it
        max_hold_days = np.ceil(self.time_stop_multiplier * self.entry_half_life[global_i])
        if self.days_held[global_i] > max_hold_days:
          self._close_position(global_i)
          continue

        # Mean-reversion exit: spread has reverted back inside the exit band
        if abs(z) < self.z_exit:
          self._close_position(global_i)
          continue

        # Still open - hold the position and refresh conviction
        self.signals[global_i] = -np.sign(z) * (abs(z) - self.z_entry)
      else:
        # Entry: only open a new position once |z| clears z_entry
        if abs(z) < self.z_entry:
          continue
        self.signals[global_i] = -np.sign(z) * (abs(z) - self.z_entry)
        self.entry_z[global_i] = z                            # >>> NEW: remember entry z for the stop-loss check
        self.entry_half_life[global_i] = stats["half_life"]    # >>> NEW: remember entry half-life for the time stop
        self.days_held[global_i] = 0                           # >>> NEW: reset holding-period counter
      
    # Return signals
    return self.signals
  
  
