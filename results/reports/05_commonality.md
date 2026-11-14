# Stage 5 – Cross-router commonality



## Anonymisation consistency check

- **Median within-router day-over-day top-N Jaccard / random expectation:** 11.1
- **Median cross-router same-day top-N Jaccard / random expectation:** 5.6
- **Highest mean same-day top-N Jaccard between two routers:** 0.79 (elpaso-phoenix)
- **Prefix-based cross-router comparison informative:** True
Interpretation: a within-router ratio near 1 means pseudonyms change every day; a cross-router ratio near 1 means pseudonyms differ between routers, in which case only the temporal-pattern similarity below should be used in the paper. A pair of adjacent routers with a high same-day overlap is direct evidence that pseudonyms are shared across routers within a day.


## Prefix-share similarity (mean over days)

| direction | pair | jaccard_topN | cosine_flows | cosine_packets | jsd_flows | jsd_packets |
|---|---|---|---|---|---|---|
| dst | atlanta-batonrouge | 0.185 | 0.263 | 0.375 | 0.619 | 0.558 |
| dst | atlanta-elpaso | 0.269 | 0.375 | 0.627 | 0.493 | 0.420 |
| dst | atlanta-phoenix | 0.263 | 0.366 | 0.561 | 0.507 | 0.442 |
| dst | batonrouge-phoenix | 0.138 | 0.370 | 0.388 | 0.520 | 0.505 |
| dst | dallas-atlanta | 0.100 | 0.140 | 0.262 | 0.817 | 0.746 |
| dst | dallas-batonrouge | 0.080 | 0.175 | 0.209 | 0.795 | 0.754 |
| dst | dallas-elpaso | 0.102 | 0.211 | 0.319 | 0.784 | 0.694 |
| dst | dallas-phoenix | 0.097 | 0.191 | 0.304 | 0.800 | 0.704 |
| dst | elpaso-batonrouge | 0.132 | 0.356 | 0.377 | 0.536 | 0.512 |
| dst | elpaso-phoenix | 0.677 | 0.965 | 0.976 | 0.034 | 0.014 |
| src | atlanta-batonrouge | 0.136 | 0.201 | 0.337 | 0.705 | 0.672 |
| src | atlanta-elpaso | 0.267 | 0.413 | 0.762 | 0.526 | 0.468 |
| src | atlanta-phoenix | 0.262 | 0.397 | 0.711 | 0.533 | 0.489 |
| src | batonrouge-phoenix | 0.270 | 0.232 | 0.333 | 0.579 | 0.572 |
| src | dallas-atlanta | 0.286 | 0.373 | 0.316 | 0.528 | 0.583 |
| src | dallas-batonrouge | 0.086 | 0.144 | 0.184 | 0.810 | 0.765 |
| src | dallas-elpaso | 0.248 | 0.235 | 0.285 | 0.648 | 0.639 |
| src | dallas-phoenix | 0.251 | 0.243 | 0.280 | 0.641 | 0.639 |
| src | elpaso-batonrouge | 0.273 | 0.222 | 0.331 | 0.579 | 0.572 |
| src | elpaso-phoenix | 0.899 | 0.968 | 0.984 | 0.025 | 0.014 |


## Temporal-pattern similarity per router pair

| pair | xcorr_max | xcorr_lag_hours | pearson_lag0 | diurnal_cosine | diurnal_dtw | hops | prefix_cosine_src | prefix_jaccard_src |
|---|---|---|---|---|---|---|---|---|
| dallas-atlanta | 0.617 | 0 | 0.617 | 0.967 | 5.687 | 5 | 0.373 | 0.286 |
| dallas-elpaso | 0.653 | 0 | 0.653 | 0.987 | 3.495 | 4 | 0.235 | 0.248 |
| dallas-batonrouge | 0.729 | 0 | 0.729 | 0.997 | 1.228 | 2 | 0.144 | 0.086 |
| dallas-phoenix | 0.661 | 0 | 0.661 | 0.981 | 4.208 | 5 | 0.243 | 0.251 |
| atlanta-elpaso | 0.663 | 0 | 0.663 | 0.983 | 3.022 | 8 | 0.413 | 0.267 |
| atlanta-batonrouge | 0.618 | 0 | 0.618 | 0.965 | 5.914 | 3 | 0.201 | 0.136 |
| atlanta-phoenix | 0.666 | 0 | 0.666 | 0.986 | 2.499 | 9 | 0.397 | 0.262 |
| elpaso-batonrouge | 0.691 | 0 | 0.691 | 0.983 | 3.442 | 6 | 0.222 | 0.273 |
| elpaso-phoenix | 0.984 | 0 | 0.984 | 0.998 | 1.119 | 1 | 0.968 | 0.899 |
| batonrouge-phoenix | 0.707 | 0 | 0.707 | 0.977 | 4.413 | 7 | 0.232 | 0.270 |


## Per-day Pearson correlation of hourly volumes (mean ± std)

| pair | mean | std | count |
|---|---|---|---|
| atlanta-batonrouge | 0.474 | 0.376 | 43 |
| atlanta-elpaso | 0.733 | 0.220 | 43 |
| atlanta-phoenix | 0.716 | 0.218 | 43 |
| batonrouge-phoenix | 0.457 | 0.315 | 43 |
| dallas-atlanta | 0.700 | 0.306 | 43 |
| dallas-batonrouge | 0.414 | 0.406 | 43 |
| dallas-elpaso | 0.577 | 0.305 | 43 |
| dallas-phoenix | 0.578 | 0.306 | 43 |
| elpaso-batonrouge | 0.464 | 0.330 | 43 |
| elpaso-phoenix | 0.969 | 0.063 | 43 |

