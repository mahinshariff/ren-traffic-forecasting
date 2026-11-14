# Methods and design decisions

This document records *why* the pipeline is built the way it is, so that the
numbers in the paper can be traced back to concrete choices.

## 1. Coverage-based window selection

A NetFlow collection from a production backbone is never uniform: routers join
and leave the collection, exporters restart, and the first and last days are
partial. A fixed chronological train/validation/test split applied to the
whole collection can therefore place the test block in a period where a
router exported almost nothing — in which case the evaluation measures the
outage, not the forecaster.

The `coverage` stage counts flows per router per calendar day and marks a day
*good* when both its flow count and its packet count are at least
`data.good_day_fraction` (default 0.2) of that router's median daily values. The study window is the longest run
of consecutive days on which **all** selected routers are good. The fourth
router is chosen automatically as the lowest-volume router that is good on at
least 95 % of the window days and is not so sparse that minute-level series
are mostly empty (`data.auto_select`). All thresholds are in the config, and
the stage report lists the day-by-day flow counts so that the choice can be
audited.

## 1b. An independent fourth router

The automatic rule above selects the lowest-volume router with complete
coverage. On this collection that is Phoenix, whose traffic is almost
identical to El Paso's (hourly-volume correlation 0.98, 90 % overlap of the
most popular address prefixes on the same day): the two routers are adjacent
on the backbone and see the same flows. Phoenix is therefore used only in the
cross-router commonality analysis (`data.extra_routers`), where an adjacent
pair is informative, and the next candidate, Baton Rouge, is the fourth router
of the forecasting study (`data.fourth_router`).

## 2. Supervised windows are built before the split

Examples are created over the entire resampled series: an example with origin
*t* has inputs `y[t-L+1..t]` and targets `y[t+1..t+H]`, where *L* is the
lookback in steps and *H* the number of steps in 24 h. Each example is then
assigned to train / validation / test by the timestamp of its **first target**
(70 / 15 / 15 % of the window by time). The scaler (standardisation, optionally
after `log1p`) is fitted on the training-period values only.

Compared with splitting the series first and then building examples inside
each block, this keeps every test-period target even at coarse windows (a
7-day test block at a 1-day window still yields 7 test targets instead of
zero) while never letting a target from the test period influence training.

**Log transform.** Packet volumes are modelled as `log1p(y)` before
standardisation (`sweep.transform: log1p`). Backbone volumes are heavy-tailed
and non-stationary (in this collection the first two weeks of October carry
three to five times the November volume on the Dallas router); on the raw
scale the mean-squared-error loss is dominated by the high-volume regime and
the models generalise poorly to the lower regime in which the test block
falls. In log space the loss is relative, which matches the percentage
metrics used for evaluation. All metrics are computed after inverting the
transform.

**Loss on the evaluated lead times.** Each model has a multi-step head that
covers 24 h, but the loss is computed only on the output steps that are
evaluated (`sweep.loss_on_evaluated_leads: true`). With the loss spread over
all 24 h of outputs (288 steps at a 5-minute window) the many far-horizon
terms dominate and short-lead accuracy suffers; restricting the loss brought
the 1-hour-lead accuracy of the learned models to the level of the
persistence baseline in our validation runs.

**Calendar covariates (optional, off by default).** With
`sweep.calendar_features: true`, every lookback step carries sine/cosine
encodings of the hour of day and day of week in addition to the scaled
volume, and the encodings of the first forecast step are provided as
known-future inputs. On this collection (four weeks of training data) the
covariates did not help consistently — they improved some routers at long
lead times and hurt others — so the reported experiments use the volume
history alone.

## 3. Lookback (lag) and lead-time rules

* The lookback is specified in hours (`sweep.lags_hours`, default 24, 48 and
  168 h) and converted to steps per window. A pair (window, lag) is skipped
  when it yields fewer than `min_lookback_steps` (4) steps — a 24 h lookback at
  a 1-day window is a single value and cannot represent a daily cycle — or
  more than `max_lookback_steps` (576, i.e. 48 h at 5 min) to bound sequence
  length for the recurrent models.
* Lead times (`sweep.horizons_hours`, default 1, 6, 12 and 24 h) are evaluated
  only when they are a whole number of steps of at least one window. This
  avoids reporting identical numbers for different nominal lead times.
* One model per (window, lag, model, seed) is trained with a multi-step output
  head covering 24 h; the lead-time metrics are read off the corresponding
  output step.

## 4. Models and training

* **Hybrid GRU-LSTM**: GRU(64) → dropout 0.25 → LSTM(256) → dropout 0.2 →
  dense(64, ReLU) → linear(H) (the last hidden state is concatenated with the
  known-future covariates when calendar features are enabled). **GRU** and
  **LSTM** ablations keep the same shape with a single cell type.
* **N-HiTS** (`renfc/models/nhits.py`): three stacks, each with max-pooling of
  the target lookback (kernels 8, 4, 1), a two-layer 512-unit MLP, a backcast
  of length *L* and a forecast produced by linear interpolation of `H / r`
  knots (*r* = 24, 12, 1), combined with doubly residual stacking (calendar
  covariates, when enabled, are appended to each block's MLP input).
* Adam (lr 1e-3), MSE loss, batch 128, up to 100 epochs, early stopping on
  validation loss with patience 10, gradient-norm clipping at 1, best
  validation weights restored. Three seeds per configuration; the figure
  stage reports mean ± standard deviation.
* **Baselines**: last value and seasonal naive (same time one day earlier),
  evaluated on exactly the same test origins and lead times.

## 5. Metrics and sanity checks

MAE, RMSE, sMAPE, MAPE, the 99th percentile of the absolute percentage error
(P99 APE; a percentage so that routers with very different loads are
comparable), R² and MSLE. Percentage errors use only points with a positive
actual value; `pct_support` records that fraction. Every run additionally
records the ratio of mean forecast to mean actual, the correlation between
forecast and actual and the fraction of negative forecasts; a run is flagged
when the ratio leaves [0.5, 2], the correlation is non-positive or more than
5 % of forecasts are negative.

## 6. Traffic-distribution statistics

* **Flow sizes**: packets per sampled flow, per calendar week of the window
  and overall. Maximum-likelihood power-law fits with automatic lower cut-off
  (Clauset, Shalizi & Newman, *SIAM Review* 2009; `powerlaw` package) and
  log-likelihood-ratio comparisons against log-normal and exponential
  alternatives. CCDFs are computed from exact log-binned histograms of all
  flows.
* **Self-similarity**: variance–time plots of the 1-minute packet series
  (variance of block means vs block length) and the Hurst exponent
  *H = 1 + β/2* from the log–log slope β (Leland, Taqqu, Willinger & Wilson,
  *IEEE/ACM ToN* 1994).
* **Window statistics**: coefficient of variation, lag-1 autocorrelation and
  STL seasonal strength (Wang, Smith & Hyndman 2006) of the series at every
  aggregation window of the sweep.
* **Window rule**: for a lead time *h*, among the aggregation windows no
  longer than *h*, choose the window at which the volume series has the
  largest autocorrelation magnitude at lag *h* (its square is the variance a
  linear predictor explains from that lag; the sign is negative at a 12-hour
  lead because of the daily cycle). The magnitude increases with aggregation
  while noise is averaged out and decreases once the window is too coarse to
  resolve the cycle. The rule needs no trained model and is
  compared, per router and lead time, with the window that minimises the
  headline error in the sweep.

## 7. Cross-router commonality

Addresses in the collection are pseudonymised per day. The `commonality`
stage therefore (a) tests whether pseudonyms are consistent across routers on
the same day by comparing the overlap of the most popular /16 prefixes with
the overlap expected for random prefix sets, and (b) always computes
pseudonym-independent temporal similarities: per-day Pearson correlation of
hourly volumes, maximum cross-correlation within ±6 h, cosine similarity and
dynamic-time-warping distance of the mean diurnal profiles. When (a) shows
consistent pseudonyms, prefix-share similarities (cosine, Jensen–Shannon
divergence, top-N Jaccard) are reported as well. Backbone hop counts between
router pairs are supplied in the config (`topology.hops`) and joined to the
similarity tables.
