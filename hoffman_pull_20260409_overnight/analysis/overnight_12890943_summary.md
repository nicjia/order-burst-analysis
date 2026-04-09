# Overnight Backtest Summary (Job 12890943)

## Execution Health
- Job out files: 4
- Completed markers: 4/4
- Tracebacks: 0
- NaN errors: 0
- Shell line errors: 0

## Run-Level Metrics
| ticker   | target   | patch                 |   dropped_rows |   total_valid_bursts |   total_trades |   longs |   shorts |       cum_pnl_raw |   sharpe |   signals_evaluated |
|:---------|:---------|:----------------------|---------------:|---------------------:|---------------:|--------:|---------:|------------------:|---------:|--------------------:|
| JPM      | reg_clcl | nan-guard-v4-20260409 |           1237 |               627222 |         409968 |  236727 |   173241 |      -1.59036e+07 |    -2.5  |              627222 |
| JPM      | reg_clop | nan-guard-v4-20260409 |             39 |                 4560 |             22 |      22 |        0 | -313475           |    -0.58 |                4560 |
| MS       | reg_clcl | nan-guard-v4-20260409 |             47 |                32965 |           2302 |    2278 |       24 | -187958           |    -0.76 |               32965 |
| MS       | reg_clop | nan-guard-v4-20260409 |             32 |                25097 |           5822 |    5808 |       14 | -297268           |    -1.19 |               25097 |
| NVDA     | reg_clcl | nan-guard-v4-20260409 |             42 |                15602 |            701 |     286 |      415 |       1.18711e+08 |     1.58 |               15602 |
| NVDA     | reg_clop | nan-guard-v4-20260409 |             65 |                24958 |           1857 |    1314 |      543 |       1.81748e+08 |     1.17 |               24958 |
| TSLA     | reg_clcl | nan-guard-v4-20260409 |              9 |                 3173 |              0 |       0 |        0 |       0           |     0    |                3173 |
| TSLA     | reg_clop | nan-guard-v4-20260409 |              6 |                 1513 |              0 |       0 |        0 |       0           |     0    |                1513 |

## Aggregate by Target
| target   |   runs |   trades |   cum_pnl_raw |   mean_sharpe |   dropped_rows |
|:---------|-------:|---------:|--------------:|--------------:|---------------:|
| reg_clcl |      4 |   412971 |   1.02619e+08 |         -0.42 |           1335 |
| reg_clop |      4 |     7701 |   1.81137e+08 |         -0.15 |            142 |

## Aggregate by Ticker
| ticker   |   runs |   trades |       cum_pnl_raw |   mean_sharpe |   dropped_rows |
|:---------|-------:|---------:|------------------:|--------------:|---------------:|
| JPM      |      2 |   409990 |      -1.62171e+07 |        -1.54  |           1276 |
| MS       |      2 |     8124 | -485226           |        -0.975 |             79 |
| NVDA     |      2 |     2558 |       3.00459e+08 |         1.375 |            107 |
| TSLA     |      2 |        0 |       0           |         0     |             15 |
