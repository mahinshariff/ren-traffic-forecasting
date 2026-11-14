# Traffic Forecasting on the Internet2 Backbone: Aggregation Windows, Lead and Lag Times

Code and results for

> M. A. U. Shariff, V. S. S. L. Karanam, and B. Ramamurthy, *Empirical Analysis of Traffic
> Forecasting Sensitivity to Lead, Lag, and Aggregation Windows in Research and Education
> Networks*, submitted to IEEE Transactions on Network Science and Engineering.

The study measures how the three temporal windows that frame a forecasting problem (the
aggregation window, the lookback or lag window, and the lead time) shape forecast accuracy on
NetFlow traces from four core routers of the Internet2 backbone, characterises the traffic
(heavy-tailed flow sizes, self-similarity, burstiness and seasonality per aggregation level),
derives a training-free rule for the aggregation window, and measures cross-router commonality
against the backbone topology.

## Repository layout

```
src/renfc/        the analysis pipeline (Python package `renfc`)
configs/          study configuration (default.yaml) and a small test configuration (smoke.yaml)
scripts/          synthetic-data generator, end-to-end smoke test, run-summary collector
slurm/            batch scripts for a SLURM cluster (one GPU per router for the sweep)
tests/            unit tests (pytest)
docs/METHODS.md   design decisions behind every stage
results/          outputs of the study (see "Results of the study" below)
sample_data/      synthetic NetFlow-like Parquet files to exercise the pipeline without the private data
```

## The pipeline

Six stages, each a sub-command of `python -m renfc`, driven by one YAML file:

| Stage | What it does | Main outputs |
|---|---|---|
| `coverage` | scans every raw router file, flags days with incomplete export, picks the study window on which all selected routers are complete | `coverage/daily_grid.csv`, `selection.json`, coverage figure |
| `etl` | one streaming pass per router: 1-minute packet series, weekly flow-size samples and histograms, per-day address-prefix shares | `series/`, `flows/`, `prefixes/` |
| `sweep` | forecasting experiments over aggregation window × lookback × model × seed, evaluated at several lead times, with seasonal-naive and last-value baselines; resumable | `results/sweep_results.csv`, `predictions/` |
| `distribution` | power-law fits of flow sizes, variance–time / Hurst exponents, burstiness and seasonal strength per window, autocorrelation-at-lead window rule | `distribution/*.csv` |
| `commonality` | prefix-share and temporal similarity between routers, anonymisation-consistency check, hop counts from the configuration | `commonality/*.csv` |
| `figures` | the figures and LaTeX tables of the paper | `figures/`, `tables/` |

Models: hybrid GRU–LSTM, GRU, LSTM (PyTorch) and N-HiTS (multi-rate sampling, hierarchical
interpolation, doubly residual stacks), trained on log-transformed volumes with the loss restricted
to the evaluated lead times. `docs/METHODS.md` explains the reasoning behind each choice.

## Quick start on synthetic data (CPU, about 3 minutes)

```bash
git clone https://github.com/mahinshariff/ren-traffic-forecasting.git
cd ren-traffic-forecasting
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q

export RENFC_RAW_DIR=$PWD/sample_data/raw RENFC_OUT_DIR=/tmp/renfc_out
python -m renfc all -c configs/smoke.yaml
ls /tmp/renfc_out/reports /tmp/renfc_out/figures /tmp/renfc_out/tables
```

`sample_data/raw/` holds six synthetic routers (21 days) with the schema of the Internet2
collection and the properties the pipeline is built to handle: diurnal and weekly cycles,
heavy-tailed flow sizes, a router whose export stops early, a router with an outage, a very
sparse router, and partial first and last days. The coverage stage selects the window and the
fourth router automatically; the smoke configuration uses small models and a few epochs so the
whole run takes minutes. `bash scripts/smoke_test.sh` does the same from scratch, regenerating
the synthetic data first.

## Reproducing the study

The raw NetFlow records were collected from Internet2 core routers under a data use agreement and
are not distributed. The pipeline expects one Parquet file (or dataset directory) per router,
`<router>.parquet`, with at least the columns `t_first` (flow start time), `in_packets`,
`src4_addr` and `dst4_addr`; column names, paths, routers, windows, lookbacks, lead times,
models, seeds and figure styling are all set in `configs/default.yaml`. On the cluster used for
the study the data live under

```
/mnt/nrdstor/ramamurthy/mhnarfth/Datasets/internet2_old_processed/raw_parquet/<router>.parquet
```

and the outputs under `$WORK/ren-traffic-forecasting/outputs` (both are `paths` entries of the
configuration). The full run on SLURM:

```bash
bash slurm/env_setup.sh                                   # conda env "renfc" with CUDA PyTorch
j1=$(sbatch --parsable slurm/01_coverage.sbatch)
j2=$(sbatch --parsable --dependency=afterok:$j1 slurm/02_etl.sbatch)
j3=$(sbatch --parsable --dependency=afterok:$j2 slurm/03_sweep.sbatch)   # array job, one GPU per router
j4=$(sbatch --parsable --dependency=afterok:$j3 slurm/04_analysis.sbatch)
bash scripts/collect_run_summary.sh                       # one text file with every report and log tail
```

The sweep (4 routers × 10 windows × 3 lookbacks × 4 models × 3 seeds, 1,200 training runs)
takes about 45 minutes of wall time on four GPUs; a time-out can be resubmitted and finished runs
are skipped. The figures in the paper were rendered with `figures.suptitles: false` and
`figures.panel_layout: row`.

## Results of the study (`results/`)

`results/` is the output directory of the pipeline for the study, with its sub-folders as
produced by the stages above:

```
results/coverage/        daily flow and packet counts of all ten routers, the good-day grid and the
                         selected window (selection.json)
results/reports/         one Markdown + JSON report per analysis stage, with the tables printed at run time
results/series/          1-minute packet series per router (dallas, atlanta, elpaso, batonrouge, phoenix)
results/flows/           weekly flow-size samples and histograms per router
results/prefixes/        per-day /16 address-prefix shares per router (pseudonymised addresses)
results/results/         sweep_results.csv — every run × lead time (1,280 runs, 3,984 rows: MAE, RMSE,
                         sMAPE, MAPE, P99 APE, R², MSLE, training time/epochs, sanity flags), plus the
                         per-router files it was merged from
results/predictions/     test-set predictions of every model and seed at the 6 h and 12 h windows
                         (used for the qualitative figure)
results/distribution/    power-law fits, CCDF points, variance–time curves and Hurst exponents,
                         statistics vs window, autocorrelation-at-lead window recommendation
results/commonality/     prefix-share similarity (daily and summary), anonymisation consistency,
                         temporal similarity (daily and per pair), hourly volumes per router
results/figures/         all thirteen figures as rendered by the pipeline (grid layout, PDF + PNG)
results/figures_paper/   the same figures in the layout used in the manuscript
                         (figures.suptitles: false, figures.panel_layout: row)
results/tables/          the LaTeX tables generated from the CSVs above
```

### Where each figure and table of the paper comes from

| In the paper | File in `results/figures_paper/` or `results/tables/` | Produced from |
|---|---|---|
| Fig. 1 data coverage | `fig_coverage.pdf` | `results/coverage/daily_grid.csv` |
| Fig. 2 topology sketch | drawn in the manuscript | hop counts in `configs/default.yaml` (`topology.hops`) |
| Fig. 3 hourly volumes | `fig_hourly_volumes.pdf` | `results/series/`, `results/commonality/hourly_volumes.csv` |
| Fig. 4 flow-size CCDF | `fig_flowsize_ccdf.pdf` | `results/distribution/flowsize_ccdf.csv`, `powerlaw_fits.csv` |
| Fig. 5 variance–time | `fig_variance_time.pdf` | `results/distribution/variance_time.csv`, `hurst.csv` |
| Fig. 6 statistics vs window | `fig_stats_vs_window.pdf` | `results/distribution/stats_vs_window.csv` |
| Figs. 7–9 window / lead / lag sensitivity | `fig_sweep_window.pdf`, `fig_sweep_horizon.pdf`, `fig_sweep_lag.pdf` | `results/results/sweep_results.csv` |
| Fig. 10 qualitative forecasts | `fig_qualitative.pdf` | `results/predictions/` |
| Fig. 11 commonality heat maps | `fig_commonality_heatmaps.pdf` | `results/commonality/*.csv` |
| Table I dataset | `tab_dataset.tex` | `results/coverage/router_summary.csv`, `results/series/` |
| Table II heavy-tail statistics | `tab_distribution.tex` | `results/distribution/powerlaw_fits.csv`, `hurst.csv` |
| Table III experimental grid | typeset in the manuscript | `configs/default.yaml` |
| Table IV best configuration per router and lead | `tab_best_window.tex` | `results/results/sweep_results.csv` |
| Table V lookback effect | `tab_lag.tex` | `results/results/sweep_results.csv` |
| Table VI model comparison | `tab_models.tex` | `results/results/sweep_results.csv` |
| Table VII window rule vs sweep optimum | `tab_window_rule.tex` | `results/distribution/window_recommendation.csv` |
| Table VIII router-pair commonality | `tab_commonality.tex` | `results/commonality/temporal_similarity_pairs.csv`, `prefix_similarity_summary.csv` |

Every number quoted in the paper can be traced to `results/reports/*.md` (the stage reports print
the same tables) or to the CSV named above. Three generated figures are not in the manuscript:
`fig_model_ablation`, `fig_diurnal_profiles` and `fig_commonality_daily` (in `results/figures/`
and `results/figures_paper/`).

## Acknowledgements

We thank the Internet2 consortium for providing the network telemetry, and the Holland
Computing Center of the University of Nebraska for computing resources. This work was supported
in part by the National Science Foundation under Grants OAC-2322369 and CNS-2413851, and by the
U.S. Department of Energy under Grant DE-SC0024648.

## License

MIT, see `LICENSE`.
