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



# ---------------------------------------------------------------------------
# 1. Strategy interface
# ---------------------------------------------------------------------------

class Strategy:
    """Common interface - every strategy returns a raw signal per instrument.
    Sign = direction, magnitude = conviction. Don't worry about sizing here,
    the ensemble normalizes and sizes after combining."""

    def signal(self, prices):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. Regime detection -> probabilities, not a hard label
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. Ensemble: regime probabilities -> weights -> combined signal
# ---------------------------------------------------------------------------

