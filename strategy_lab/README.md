# strategy_lab

Research and validation toolkit for the Algothon 2026 submission.

The submission itself is the single file `../The_Limitless_Liquidity_Providers.py`.
Nothing in this folder ships — it exists to validate what does.

```
python -m strategy_lab.diagnostics            # dataset structure checks
python -m strategy_lab.hyperparameter_sweep   # walk-forward coordinate sweep
python -m strategy_lab.hyperparameter_sweep --weights        # ensemble weights
python -m strategy_lab.hyperparameter_sweep --axis stat_arb.n_factors
python -m strategy_lab.hyperparameter_sweep --holdout        # single-use holdout
python eval.py                                # the official scorer
```

**Dependencies.** The submission uses `numpy`, `pandas`, `scipy`,
`scikit-learn`, `statsmodels` (all in the grading sandbox) plus **`hmmlearn`**,
which is not. The submission's own `requirements.txt` must therefore contain
exactly one line: `hmmlearn`. Do not submit a copy of `requirements-dev.txt`.

**One performance trap.** `statsmodels.tsa.stattools.adfuller` defaults to
`autolag='AIC'`, which fits ~13 OLS models per call — 1.700 ms/call versus
0.198 ms/call with `maxlag=n, autolag=None`. That default was 84% of the
runtime in the original strategy. Always pin both; see
`augmented_dickey_fuller_pvalues`.

---

## Shape convention

**Days down the rows, instruments across the columns — `(n_days, n_instruments)` — everywhere.**

`prices.txt` is already stored that way. The *only* transpose in the entire
codebase is the first line of `getMyPosition`, which flips the evaluator's
`(n_instruments, n_days)` input once at the boundary. Nothing downstream ever
transposes again, and no function in `strategy_lab/` takes instrument-major
data.

Naming follows the convention: `*_by_day` is a 2-D `(n_days, n_instruments)`
array; `*_per_instrument` is a flat length-`n_instruments` vector for today.

---

## Results

Full three-strategy ensemble, hedge overlay on:

| Window | Days | Score | Mean PL | Sharpe |
|---|---|---|---|---|
| Walk-forward, 21 overlapping folds, pooled | 21 × 59 | **117** | $158 | 1.68 |
| Holdout (never touched by any fold) | 150 | **194** | $229 | 2.36 |
| `eval.py`, `numTestDays=250` | 250 | **190** | $227 | 2.27 |
| `eval.py`, `numTestDays=500` ← competition | 500 | **133** | $174 | 1.79 |

**The competition scores 500 days, so 133 is the number that matters.** It is
lower than the 250-day figure because the extra 250 days reach back into a
weaker stretch of the series, not because anything is wrong — 86% of folds are
profitable.

`strategy_lab.backtester` reproduces `eval.py` to the cent — verified at both
250 and 500 test days, including the warm-up day, the mark-only final day, and
the one-day commission lag in the reference implementation. If those two
numbers ever disagree, trust `eval.py` and fix the backtester.

The pooled walk-forward number (117) is the honest forward-looking estimate.
**Plan around 117, not 194.**

`PortfolioConfig.minimum_history_days()` is 132. This matters more than it
looks: `eval.py` gives the first scored day only `n_days - numTestDays` of
history, which at `numTestDays=500` on a 1000-day file is 500 days. Any
requirement above that would leave the book silently flat through the start of
the scored window. Keep it well under 500.

---

## What the data actually is

Run `python -m strategy_lab.diagnostics` for the live version. The findings
that shaped every design decision:

**1. ALGO is an exact fixed-weight index of the other 50 instruments.**
Regressing ALGO's *price* on the other 50 prices gives R² = 0.99999988 with a
residual standard deviation of $0.0032 — which is precisely the $0.0032
predicted by the prices being rounded to two decimals, and nothing else. All 50
weights are positive and sum to 2.90.

*An R² of 1.0 from 50 free parameters deserves suspicion, so it was tested
properly.* Fitting on **52 days — one more than the 50 parameters — still
predicts the remaining 948 days to $0.157**, and accuracy improves monotonically
with more fit data ($0.011 at 250 days, $0.005 at 500). Overfitting cannot
generalise from 52 points to 948. The control settles it: the identical
regression on column-shuffled prices gives out-of-sample R² of **−0.077**, which
is what 50 parameters fitting noise actually looks like. The weights are stable
to **0.65%** across five disjoint 200-day windows. This is a real structural
relation, not a fit artefact.

Three consequences:
- Index exposure can be hedged with essentially zero tracking error.
- ALGO adds **no new idiosyncratic capacity**. It is spanned exactly by the
  other 50, so a position in ALGO is a pure directional bet on the whole
  market. A book whose edge is relative-value gains nothing by touching it.
- The ALGO-vs-basket basis is ~0.33 bps against a 2.4 bp round-trip
  commission. It is real, and it is roughly 1000× too small to trade.

**2. No lead or lag.** Only the contemporaneous relation is exact; shifting the
basket by ±1 day drops R² to 0.989. There is no free signal here. Worth
re-running when the extra 250 days land — a lead/lag is exactly the kind of
artefact that can appear when a dataset is regenerated.

**3. The universe reverts, it does not trend.** Cross-sectional information
coefficient is negative at every horizon from 3 to 250 days, strongest around
20–100 days (IC ≈ −0.02, t ≈ −3). This is the entire edge, and it is modest:
IC of 0.02 across ~50 names implies a ceiling near
`0.02 × √50 × √250 ≈ 2.2` annualised Sharpe. The strategy realises 1.7–3.3,
which is the right order of magnitude — a useful reality check on any claim
of a much higher Sharpe from this data.

**4. Only ~3 real factors.** Eigenvalues are 11.9, 2.8, 1.9, then everything
from the 8th on sits at ~1.0 — the Marchenko-Pastur noise bulk. PC1 correlates
0.96 with ALGO's return, so ALGO *is* the market factor.

**5. Nothing else.** No periodicity, no volatility clustering (lag-1
autocorrelation of squared returns ≈ −0.008), no per-instrument autocorrelation
worth trading.

---

## Why some leaderboard scores are ~1000

A score of 1000 means mean PL of roughly $1,100/day at a high Sharpe. Total
deployable capital is $600k (50 × $10k + $100k for ALGO), so that is ~0.19% per
day on *gross* notional, sustained. Working backwards from the measured
statistics of this book — mean PL $174/day at a daily PL standard deviation of
$1,537 — reaching $1,100/day at the same risk would require an annualised
Sharpe near 11. Nothing in the data supports that: the cross-sectional IC caps
a fully-diversified book at roughly Sharpe 2–3.

So the plausible explanations, in order:

1. **Overfitting to the public window.** `prices.txt` is public and `eval.py`
   scores its last 250 days. Anyone can tune directly against the exact number
   the leaderboard reports. Fitting hard against a public window is easy and
   produces enormous in-sample scores that collapse on fresh data. This is
   almost certainly the bulk of it, and it is why the config here was locked on
   walk-forward folds and checked once on an untouched holdout — a process that
   *cost* score on the public window versus an in-sample pick, and is worth it.
2. **The leaderboard's `prices.txt` may differ from ours.** If the graded stage
   uses more days, or a regenerated series, scores are not comparable at all.
3. **Genuinely more capital deployed.** Score is *linear* in mean PL, so a team
   running at the position limits scores several times one running at half
   size with the identical signal. This is real and worth capturing — it is why
   `signal_gain` is 8.0 and the book runs ~$535k gross. It is worth a factor of
   ~2–4, not ~6.
4. **A structural edge we have not found.** Possible, but the checks above rule
   out the usual suspects (index arbitrage, lead/lag, periodicity).

Practical advice: do not chase the leaderboard by tuning on the public 250
days. When the extra 250 days land, re-run the sweep and compare the *new*
holdout against 117 — that comparison tells you whether the edge is real. A
large drop on new data is the signature of everything in category 1.

---

## Hyperparameters: what to vary

The original sweep varied `vol_span` alone across 2–30, and every result was
negative. Two separate problems:

**The signal was being used as a position.** `StatArb(...).signal` was passed
straight to the backtester as `get_position_fn`, so a conviction score of 1.5
became *1.5 shares* — about $50 of notional. Mean absolute net exposure came
out at $4.87. At that size commission ($0.056/day) was ~136% of gross PL
($0.041/day), so the sign of the result was determined entirely by costs. The
edge was there; the sizing layer was missing. `PortfolioEngine` now owns the
signal → dollars → shares conversion, and no strategy returns positions.

**`vol_span` is the least important knob.** It only rescales returns before the
factor fit. Across 10–90 the pooled score does move (−68 to 104), but it moves
*because* it interacts with everything else — on its own it was never going to
turn a −0.015 result positive.

Ranked by measured impact on pooled walk-forward score:

| # | Parameter | Why it matters | Chosen |
|---|---|---|---|
| 1 | `sizing.signal_gain` | Score is **linear** in mean PL and the Sharpe term is scale-invariant, so unused position limit is forgone score outright. | **8.0** |
| 2 | `ensemble.baseline_weights` | See below — the three PL streams are near-uncorrelated, so the blend matters a lot. | **1 : 0.5 : 0.05** |
| 3 | `hedge.enabled` / `target_hedge_ratio` | Worth **+47** once mean-reversion is weighted. | **True / 1.0** |
| 4 | `mean_reversion.lookback_window` + `entry_zscore` | Very sensitive; the two interact. | **5 / 1.0** |
| 5 | `stat_arb.n_factors` | How much systematic risk is stripped before looking for reversion. | **30** |
| 6 | `stat_arb.reversion_window` | Horizon of the reversion traded; interacts with `n_factors`. | **20** |
| 7 | `momentum.min_variance_ratio` | The trend gate. Difference between −136 and break-even. | **1.1** |
| 8 | `stat_arb.volatility_span` | Vol standardisation before the factor fit. | **60** |
| 9 | `stat_arb.factor_fit_window` | Loading estimation window. | **252** |
| 10 | `sizing.rebalance_threshold_dollars` | Commission control; effect inside the noise. | **1000** |

### Ensemble weights

Swept separately with `--weights`: they are a simplex, not independent axes, so
a coordinate sweep over three numbers that get normalised anyway would mostly
re-discover the same ratio.

The three PL streams are **almost perfectly uncorrelated** — stat-arb/mean-rev
+0.001, stat-arb/momentum +0.051, mean-rev/momentum +0.028 — so blending
genuinely raises Sharpe rather than just averaging.

```
SA : MR : MO       pooled   Sharpe   % profitable folds
1  : 0.5 : 0        120.0    1.71    0.86
1  : 0.5 : 0.05     116.6    1.68    0.86   <-- shipped
1  : 0.5 : 0.1      103.7    1.56    0.81
1  : 1   : 1        103.2    1.48    0.71
1  : 1   : 0        102.8    1.48    0.76
1  : 0.25: 0        102.1    1.56    0.95
1  : 0   : 0         95.5    1.51    0.76   (stat-arb alone)
0  : 1   : 0         69.4    1.14    0.67   (mean-reversion alone)
0  : 0   : 1          0.0    0.06    0.62   (momentum alone, gated)
```

### The hedge overlay earns its place — now

With stat-arb as the only live strategy the hedge *cost* score: stat-arb is
factor-neutral by construction, so the overlay only added turnover and noise.
Once mean-reversion carries a real weight the book carries real index exposure,
and the hedge is worth **+47 pooled score** (120 hedged vs 73 unhedged).

Full neutralisation (`target_hedge_ratio=1.0`) beat partial (0.8 → 116,
0.5 → 60). The usual argument for a partial hedge — avoid over-trading on
estimation noise — does not apply here, because ALGO is an *exact* basket, so
the exposure being offset is measured rather than estimated.

Verified, not assumed: realised index beta of the book is R² = **0.0002**
hedged versus **0.0016** unhedged.

### Momentum is a drag, and here is the number

This universe reverts at every horizon. Validated standalone:

```
ungated, lookbacks (5,10)   -136     (20,60)   -156
ungated, lookbacks (10,20)  -108     (60,120)   -56
gated VR>1.1, (60,120)         0.0  <-- shipped
```

The `variance_ratio` trend gate takes momentum from −136 to break-even by
restricting it to instruments whose returns actually compound. That is a real
improvement, and it is what makes a non-zero weight affordable: at 0.05 it costs
**~3 points** of pooled score (120 → 117) rather than the ~50 it would have cost
ungated.

It ships at 0.05 so the strategy stays live, wired in and visible in the score.
If teammate B's rebuild validates positive standalone, raise it and re-run
`--weights`.

### Ranking criterion

Configurations are ranked on **`pooled_score`**: every fold's daily PL series
concatenated, then scored once. A per-fold score is a Sharpe estimate on ~59
observations, and fold-to-fold standard deviation here is ~140–260 points —
ranking on the mean or median of fold scores mostly ranks luck. Pooling uses
every observation in one estimate and matches what `eval.py` measures (a single
continuous run). Per-fold statistics are still reported to show *consistency*
(`fraction_profitable_folds`, `worst_fold_score`).

### Two axes that do nothing

`max_half_life` and `require_stationary_residual` produce byte-identical
results under the shipped config, because `zscore_method='cross_sectional'`
does not consult the screen. The sweep detects zero spread across an axis and
declines to adopt a winner rather than picking one arbitrarily. Those two
parameters only bite when `zscore_method='ou'`.

---

## Why the code was slow (8 minutes), and what fixed it

It was the code, not the machine. Profiling the original at 98 ms/day:

```
0.921s total
  0.845s  _analyse_ou_process
    0.771s  statsmodels adfuller          <-- 84% of total runtime
      0.634s  _autolag                    <-- 3,237 OLS fits for 249 adfuller calls
```

**`adfuller` defaults to `autolag='AIC'`**, which fits ~13 separate OLS models
*per instrument per day*. At 50 instruments × 540 days × 9 configs that is
roughly 3 million model fits — hence 8 minutes.

The fix is not to abandon statsmodels, it is to pin the lag order:

| call | ms/call | ms/day (50 instruments) |
|---|---|---|
| `adfuller(x)` | 1.700 | 85.0 |
| `adfuller(x, maxlag=1, autolag=None)` | 0.204 | 10.2 |

That is an **8.6× speedup from two keyword arguments**, and it is wrapped in
`augmented_dickey_fuller_pvalues` so nobody has to remember. The AR(1) fit and
half-life come from `scipy.stats.linregress` in `fit_ar1`, which returns the
slope and its standard error in one call — so the Dickey–Fuller t-statistic is
free, and the full ADF p-value is only computed when the screen actually needs
it (`zscore_method='ou'`).

Measured end to end: **98 ms/day → 21 ms/day**, with all three strategies, the
HMM and the hedge overlay live — versus the original figure for the stat-arb
component alone. A full 500-day `eval.py` run takes 11 seconds.

Other decisions that keep it fast:
- **`pandas.ewm(span=…).std()`** instead of a hand-rolled recursion. Beyond
  brevity this fixed a real problem: `ewm` is bias-corrected from the first
  observation (`adjust=True`), whereas a zero-initialised recursion needs
  hundreds of rows of burn-in — which had pushed `minimum_history_days` to 514
  and would have left the book flat at the start of a 500-day scoring window.
- **Cached refits.** `hmmlearn`'s Baum–Welch costs ~29 ms, so the HMM refits
  every 25 days and runs only the cheap `predict_proba` in between (~1 ms/day
  amortised). Basket weights refit on the same cadence. Both detect a shortened
  price history as a fold restart and refit immediately.
- **`sklearn.decomposition.PCA`** for the factor model (1.1 ms/call). Slower
  than a raw `numpy.linalg.eigh` (0.38 ms) but not enough to matter, and much
  clearer.

---

## What teammates need to deliver

**Both strategies now carry real weight and are live in the score** —
mean-reversion at 0.5 and momentum at 0.05. Neither is a stub any more:

- **Mean reversion (teammate A)** is genuinely good — short-horizon reversion in
  the ALGO-residual, pooled 69 standalone, and its near-zero correlation with
  stat-arb is what lifts the blend to 120. The ALGO-beta residualisation is the
  single biggest decision in it: worth ~180 points at lookback 5. *Improve this,
  do not replace it wholesale.*
- **Momentum (teammate B)** is break-even at best after the trend gate, for the
  structural reason in finding 3 above. The honest task here is to find whether
  *any* subset of this universe trends. If not, say so and we drop it to the
  floor — that is a legitimate result, not a failure.

Replace or extend the body of `generate_signal` in your class. Do not change its
signature, and do not touch `PortfolioEngine`. Re-run
`python -m strategy_lab.hyperparameter_sweep --weights` after any change — the
shipped weights were validated against the current strategies and are not
guaranteed to stay optimal.

### The contract

```python
def generate_signal(self, prices_by_day: np.ndarray) -> np.ndarray:
```

**Input** — `(n_days, n_instruments)`, days down the rows. Column 0 is ALGO.
The last row is today. Everything you may use is in this array.

**Output** — all four are required:

1. Length `n_instruments`, dtype float, **finite everywhere**. The ensemble
   rejects a wrong-shaped or non-finite signal and substitutes zeros for your
   strategy, so a bug in one strategy cannot poison the book — but it also
   means a silently broken signal shows up as "no contribution", not a crash.
   Check your own output.
2. **Sign is direction** (positive = want to be long), **magnitude is
   conviction**. Do **not** return dollar amounts or share counts. Sizing,
   limit-clipping and hedging all happen downstream.
3. Scale is up to you — the ensemble z-scores every signal across instruments
   before weighting, so a raw percentage return and a z-score end up on the
   same footing. Only *relative* magnitudes within your own signal matter.
4. `signal[0]` (ALGO) must be `0.0` unless you deliberately want outright index
   exposure. See finding 1 above for why that is a pure market bet.

**State** — called once per day, in order, with history growing by one row.
You may keep state on `self`, but you must implement `reset()` to clear it:
the validation harness starts every fold flat.

**Cost** — called ~250 times by `eval.py` and thousands of times per sweep.
Keep it vectorised. Use `fit_ar1_and_dickey_fuller` from the submission file
for half-lives and stationarity tests rather than `statsmodels.adfuller` — same
numbers, ~400× faster, and it handles all instruments in one call.

### Validating your strategy in isolation

```python
from The_Limitless_Liquidity_Providers import PortfolioConfig
from strategy_lab.data_loading import load_prices_by_day
from strategy_lab.walk_forward import generate_folds, evaluate_config_on_folds, aggregate_fold_results

prices_by_day, _ = load_prices_by_day()

config = PortfolioConfig()
config.ensemble.baseline_weights = {"stat_arb": 0.0, "mean_reversion": 1.0, "momentum": 0.0}

folds = generate_folds(len(prices_by_day), config.minimum_history_days(),
                       test_size=60, step=25, holdout_days=150)
print(aggregate_fold_results(evaluate_config_on_folds(prices_by_day, config, folds)))
```

Judge your rebuild against the standalone numbers already on the board:
stat-arb **95**, mean-reversion **69**, momentum **0**. A strategy that is
negative alone will not be rescued by the ensemble — though note that a
*low-scoring but uncorrelated* strategy can still earn weight, which is exactly
why mean-reversion at 69 lifts the book to 120. Check the correlation of your
daily PL against the others before concluding a weak score means no value.

Also report your realised index beta:

```python
from strategy_lab.backtester import measure_index_beta
from strategy_lab.walk_forward import make_position_function
position_function, engine = make_position_function(config)
print(measure_index_beta(prices_by_day, position_function, 750, 1000))
```

The shipped book comes out at R² = **0.0002** hedged against **0.0016**
unhedged. If your strategy adds meaningful beta the overlay will absorb it, but
re-measure after any change rather than assuming — an overlay that silently
stops working looks exactly like one that is working.

---

## Regime detection

`GaussianHmmRegimeDetector` wraps **`hmmlearn.hmm.GaussianHMM`** — a 2-state
diagonal-covariance Gaussian HMM. Features are ALGO's return, |ALGO return|,
and cross-sectional return dispersion.

The original plan called for rolling ALGO-basket correlation as a stress
feature. That is unusable here: ALGO is an *exact* basket of the other 50, so
the correlation is pinned near 1.0 every day and carries no information.
Dispersion replaces it.

Two implementation details worth knowing:

- **The calm state is labelled by the lowest mean on the |ALGO return|
  feature**, not by fitted variance. EM is free to relabel states between
  refits; labelling by a feature mean keeps the label stable, so the regime
  weights do not silently flip sign after a refit.
- **Every failure path returns P(calm) = 0.5**, i.e. "no opinion", so a regime
  problem degrades to the baseline weighting instead of taking the book down.

Stat-arb's weight is scaled by its **own quality metric** (the fraction of
residuals currently passing the half-life screen) rather than by the market vol
regime — it is a relative-value bet, so market direction is the wrong
conditioning variable for it. Mean reversion and momentum are the ones tilted by
the regime.

The ensemble skips the HMM entirely when only one strategy has non-zero weight,
since it cannot change a single normalised weight.

---

## Remaining build order

1. ~~Stat-arb built, walk-forward validated, holdout checked.~~ Done.
2. ~~All three strategies live with validated weights; hedge overlay on and
   verified to reduce realised index beta.~~ Done.
3. ~~Confirmed against `eval.py` at `numTestDays=500`.~~ Done — score 133.
4. Teammates improve mean reversion and momentum against the contract above,
   validated in isolation on the same folds.
5. Re-run `--weights` after any strategy change, then re-measure index beta.
6. When the extra 250 days land: re-run `diagnostics` first (confirm the ALGO
   index relation and the absence of lead/lag still hold), then the full sweep,
   then one holdout check.
7. Submit `The_Limitless_Liquidity_Providers.py` plus a `requirements.txt`
   containing the single line `hmmlearn`. `getMyPosition` catches every
   exception and returns a flat book, so a run-time failure costs a day rather
   than the competition — but that is a safety net, not a substitute for step 6.
