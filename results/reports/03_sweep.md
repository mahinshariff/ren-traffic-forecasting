# Stage 3 – Forecasting sweep


- **Rows in results CSV:** 3984
- **Runs:** 1280
- **Routers:** atlanta, batonrouge, dallas, elpaso
- **Sanity failures (rows):** 917

## Mean sMAPE at 12 h lead time (over seeds) – router × window, best lag per model

| router | window | gru_only | hybrid_gru_lstm | lstm_only | naive_last | nhits | snaive_24h |
|---|---|---|---|---|---|---|---|
| atlanta | 5min | 73.6 | 73.1 | 69.9 | 91.3 | 52.9 | 53.4 |
| atlanta | 15min | 67.3 | 62.8 | 63.1 | 85.5 | 47.9 | 48.1 |
| atlanta | 30min | 63.3 | 58.2 | 60.0 | 82.0 | 40.7 | 44.6 |
| atlanta | 45min | 55.6 | 53.4 | 52.8 | 81.2 | 39.5 | 42.2 |
| atlanta | 1h | 51.0 | 50.1 | 47.7 | 78.9 | 36.6 | 40.7 |
| atlanta | 2h | 41.6 | 38.7 | 36.2 | 73.7 | 33.1 | 39.6 |
| atlanta | 4h | 36.6 | 39.7 | 40.1 | 66.5 | 28.3 | 38.1 |
| atlanta | 6h | 34.4 | 35.2 | 35.1 | 56.3 | 22.3 | 33.5 |
| atlanta | 12h | 35.4 | 43.8 | 43.9 | 51.7 | 15.3 | 29.9 |
| batonrouge | 5min | 59.1 | 59.5 | 59.6 | 77.4 | 65.1 | 68.0 |
| batonrouge | 15min | 49.8 | 50.0 | 50.2 | 67.3 | 51.9 | 57.1 |
| batonrouge | 30min | 46.8 | 47.0 | 47.8 | 63.9 | 42.7 | 52.1 |
| batonrouge | 45min | 43.6 | 43.2 | 43.4 | 60.7 | 41.1 | 47.9 |
| batonrouge | 1h | 44.0 | 43.8 | 43.0 | 60.1 | 43.0 | 46.3 |
| batonrouge | 2h | 39.2 | 40.8 | 39.2 | 57.6 | 40.0 | 44.3 |
| batonrouge | 4h | 32.7 | 36.0 | 37.1 | 51.2 | 32.3 | 37.9 |
| batonrouge | 6h | 32.6 | 35.1 | 29.3 | 38.9 | 31.9 | 31.8 |
| batonrouge | 12h | 30.7 | 31.7 | 33.3 | 39.8 | 32.4 | 26.9 |
| dallas | 5min | 29.8 | 32.3 | 30.2 | 42.6 | 25.2 | 31.6 |
| dallas | 15min | 24.8 | 24.4 | 27.2 | 37.1 | 19.6 | 23.2 |
| dallas | 30min | 21.4 | 20.6 | 20.3 | 35.9 | 14.8 | 21.1 |
| dallas | 45min | 21.9 | 24.3 | 25.0 | 35.9 | 16.0 | 20.2 |
| dallas | 1h | 17.6 | 19.3 | 22.3 | 35.4 | 13.4 | 19.6 |
| dallas | 2h | 21.7 | 24.4 | 23.1 | 34.4 | 15.8 | 18.2 |
| dallas | 4h | 22.5 | 21.7 | 24.3 | 28.1 | 15.5 | 17.3 |
| dallas | 6h | 18.9 | 19.2 | 19.6 | 24.5 | 11.8 | 17.3 |
| dallas | 12h | 20.0 | 20.0 | 20.0 | 23.5 | 10.7 | 14.3 |
| elpaso | 5min | 86.4 | 86.4 | 84.6 | 98.0 | 84.2 | 68.8 |
| elpaso | 15min | 69.0 | 79.9 | 72.0 | 84.8 | 66.8 | 61.3 |
| elpaso | 30min | 68.9 | 64.8 | 67.4 | 76.5 | 55.8 | 53.2 |
| elpaso | 45min | 58.0 | 58.3 | 56.7 | 73.7 | 55.0 | 45.7 |
| elpaso | 1h | 58.8 | 59.9 | 57.5 | 69.1 | 49.9 | 44.2 |
| elpaso | 2h | 45.4 | 50.3 | 49.9 | 60.9 | 46.6 | 40.9 |
| elpaso | 4h | 41.9 | 44.3 | 43.6 | 55.1 | 41.0 | 36.5 |
| elpaso | 6h | 36.6 | 37.8 | 39.5 | 38.1 | 29.5 | 31.8 |
| elpaso | 12h | 37.3 | 38.6 | 38.9 | 32.7 | 34.8 | 30.7 |


## Best learned configuration per router (min mean sMAPE at headline lead time) vs seasonal naive

| router | window | lag_hours | model | sMAPE | snaive_24h_same_window |
|---|---|---|---|---|---|
| atlanta | 12h | 168 | nhits | 15.32 | 29.89 |
| batonrouge | 6h | 24 | lstm_only | 29.32 | 31.76 |
| dallas | 12h | 168 | nhits | 10.72 | 14.34 |
| elpaso | 6h | 24 | nhits | 29.5 | 31.77 |


## Median metrics by model (all rows)

| model | sMAPE | MAPE | R2 | MSLE | P99_APE | train_time_sec |
|---|---|---|---|---|---|---|
| gru_only | 42.35 | 56.03 | -0.04069 | 0.3018 | 227.6 | 2.475 |
| hybrid_gru_lstm | 43.31 | 59.31 | -0.05237 | 0.3158 | 239.9 | 2.006 |
| lstm_only | 44.02 | 59.68 | -0.06541 | 0.326 | 243.4 | 1.984 |
| naive_last | 43.87 | 51.93 | -0.5498 | 0.4011 | 257.4 | 0 |
| nhits | 40.57 | 49.16 | 0.1824 | 0.2837 | 215.6 | 0.5606 |
| snaive_24h | 41.93 | 46.9 | -0.1123 | 0.3719 | 244.2 | 0 |


## Rows failing sanity checks

| run_id | horizon_hours | bias_ratio | corr | neg_frac | n_test |
|---|---|---|---|---|---|
| atlanta__5min__lag24__naive_last__seed0 | 12 | 1.002 | -0.1607 | 0 | 1570 |
| atlanta__5min__lag24__hybrid_gru_lstm__seed1 | 6 | 0.8235 | -0.07 | 0 | 1570 |
| atlanta__5min__lag24__hybrid_gru_lstm__seed1 | 12 | 0.6772 | -0.1808 | 0 | 1570 |
| atlanta__5min__lag24__hybrid_gru_lstm__seed2 | 12 | 1.02 | -0.1115 | 0 | 1570 |
| atlanta__5min__lag24__gru_only__seed0 | 12 | 1.136 | -0.007099 | 0 | 1570 |
| atlanta__5min__lag24__gru_only__seed1 | 6 | 0.8919 | -0.09718 | 0 | 1570 |
| atlanta__5min__lag24__lstm_only__seed1 | 12 | 1.077 | -0.07352 | 0 | 1570 |
| atlanta__5min__lag48__hybrid_gru_lstm__seed0 | 12 | 0.87 | -0.04765 | 0 | 1570 |
| atlanta__5min__lag48__hybrid_gru_lstm__seed1 | 6 | 0.79 | -0.06397 | 0 | 1570 |
| atlanta__5min__lag48__hybrid_gru_lstm__seed1 | 12 | 0.8143 | -0.03174 | 0 | 1570 |
| atlanta__5min__lag48__hybrid_gru_lstm__seed2 | 6 | 0.7648 | -0.1312 | 0 | 1570 |
| atlanta__5min__lag48__gru_only__seed1 | 6 | 1.143 | -0.2164 | 0 | 1570 |
| atlanta__5min__lag48__gru_only__seed2 | 6 | 0.6506 | -0.1113 | 0 | 1570 |
| atlanta__5min__lag48__lstm_only__seed0 | 6 | 1.207 | -0.1361 | 0 | 1570 |
| atlanta__15min__lag24__naive_last__seed0 | 12 | 1 | -0.2421 | 0 | 524 |
| atlanta__15min__lag24__lstm_only__seed1 | 12 | 1.236 | -0.01143 | 0 | 524 |
| atlanta__15min__lag24__lstm_only__seed2 | 6 | 1.04 | -0.1772 | 0 | 524 |
| atlanta__15min__lag48__hybrid_gru_lstm__seed1 | 6 | 1.074 | -0.002517 | 0 | 524 |
| atlanta__15min__lag48__gru_only__seed0 | 6 | 0.7482 | -0.2402 | 0 | 524 |
| atlanta__15min__lag48__gru_only__seed0 | 12 | 1.062 | -0.01055 | 0 | 524 |
| atlanta__15min__lag48__gru_only__seed1 | 6 | 1.198 | -0.3232 | 0 | 524 |
| atlanta__15min__lag48__gru_only__seed2 | 6 | 1.402 | -0.1932 | 0 | 524 |
| atlanta__30min__lag24__naive_last__seed0 | 12 | 1 | -0.2849 | 0 | 262 |
| atlanta__30min__lag48__gru_only__seed0 | 6 | 0.9908 | -0.05804 | 0 | 262 |
| atlanta__30min__lag48__gru_only__seed2 | 6 | 1.031 | -0.03054 | 0 | 262 |
| atlanta__30min__lag168__hybrid_gru_lstm__seed0 | 6 | 1.665 | -0.1986 | 0 | 262 |
| atlanta__30min__lag168__hybrid_gru_lstm__seed0 | 12 | 2.034 | -0.01909 | 0 | 262 |
| atlanta__30min__lag168__hybrid_gru_lstm__seed1 | 6 | 1.073 | -0.2405 | 0 | 262 |
| atlanta__30min__lag168__gru_only__seed0 | 6 | 1.215 | -0.1805 | 0 | 262 |
| atlanta__30min__lag168__gru_only__seed2 | 6 | 1.246 | -0.0613 | 0 | 262 |
| atlanta__30min__lag168__lstm_only__seed0 | 6 | 1.193 | -0.3459 | 0 | 262 |
| atlanta__30min__lag168__lstm_only__seed0 | 12 | 1.758 | -0.07205 | 0 | 262 |
| atlanta__30min__lag168__lstm_only__seed1 | 6 | 1.101 | -0.07744 | 0 | 262 |
| atlanta__30min__lag168__lstm_only__seed2 | 6 | 1.336 | -0.1468 | 0 | 262 |
| atlanta__45min__lag24__naive_last__seed0 | 12 | 0.9991 | -0.285 | 0 | 175 |
| atlanta__45min__lag168__gru_only__seed0 | 6 | 1.181 | -0.04664 | 0 | 175 |
| atlanta__1h__lag24__naive_last__seed0 | 12 | 1 | -0.3054 | 0 | 131 |
| atlanta__2h__lag24__naive_last__seed0 | 12 | 1 | -0.3006 | 0 | 66 |
| atlanta__2h__lag168__lstm_only__seed0 | 24 | 0.9512 | -0.06432 | 0 | 66 |
| atlanta__2h__lag168__lstm_only__seed1 | 24 | 0.9826 | -0.09395 | 0 | 66 |

_(877 more rows not shown)_

