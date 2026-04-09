# Overnight Deep Diagnostics (12890943)

## Key Questions
- Why NVDA can show very large PnL with modest trade count
- Why TSLA has zero trades
- Why JPM reg_clop has very few trades
- How many bursts exist and what percent trigger trades

## Burst Coverage (base files)
| ticker   |   raw_bursts_total |
|:---------|-------------------:|
| JPM      |            1715393 |
| MS       |            1306410 |
| NVDA     |            4632789 |
| TSLA     |            5263150 |

## Run Coverage + Trade Rates
| ticker   | target   |   raw_bursts_total |   filtered_bursts |   filter_keep_rate |   valid_bursts_scanned |   signals_eval |   trades |   trade_rate_vs_valid_bursts |   trade_rate_vs_signals |      cum_pnl_raw |    sharpe |
|:---------|:---------|-------------------:|------------------:|-------------------:|-----------------------:|---------------:|---------:|-----------------------------:|------------------------:|-----------------:|----------:|
| JPM      | reg_clcl |            1715393 |            659699 |           0.384576 |                 627222 |         627222 |   409968 |                     0.653625 |                0.653625 | -15903610.560000 | -2.500000 |
| JPM      | reg_clop |            1715393 |              4729 |           0.002757 |                   4560 |           4560 |       22 |                     0.004825 |                0.004825 |   -313474.700000 | -0.580000 |
| MS       | reg_clcl |            1306410 |             34218 |           0.026192 |                  32965 |          32965 |     2302 |                     0.069832 |                0.069832 |   -187958.100000 | -0.760000 |
| MS       | reg_clop |            1306410 |             26019 |           0.019916 |                  25097 |          25097 |     5822 |                     0.231980 |                0.231980 |   -297268.070000 | -1.190000 |
| NVDA     | reg_clcl |            4632789 |             16194 |           0.003496 |                  15602 |          15602 |      701 |                     0.044930 |                0.044930 | 118711049.880000 |  1.580000 |
| NVDA     | reg_clop |            4632789 |             25922 |           0.005595 |                  24958 |          24958 |     1857 |                     0.074405 |                0.074405 | 181747889.840000 |  1.170000 |
| TSLA     | reg_clcl |            5263150 |              3333 |           0.000633 |                   3173 |           3173 |        0 |                     0.000000 |                0.000000 |         0.000000 |  0.000000 |
| TSLA     | reg_clop |            5263150 |              1585 |           0.000301 |                   1513 |           1513 |        0 |                     0.000000 |                0.000000 |         0.000000 |  0.000000 |

## Gate vs Prediction Scale
| ticker   | target   |   gate_median |   pred_move_median |                                                                                                                                                                                                  pred_move_abs_max |   gate_abs_max |   long_pass_rate |   short_pass_rate |
|:---------|:---------|--------------:|-------------------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|---------------:|-----------------:|------------------:|
| JPM      | reg_clcl |      0.020000 |           0.003438 | 65745559372648870143677130798179941743447886873420914570950421436742039152069069618824626754241858584279876081886664445744746651590922077286638570487878653585233499641867359588823977851951983421238018048.000000 |       1.410000 |         0.377421 |          0.276204 |
| JPM      | reg_clop |      0.030000 |           0.000148 |                                                                                                                                                                                                      380125.159769 |       0.720000 |         0.004825 |          0.000000 |
| MS       | reg_clcl |      0.020000 |           0.002016 |                                                                                                                                                                                                270106468142.540894 |       0.710000 |         0.069104 |          0.000728 |
| MS       | reg_clop |      0.020000 |           0.004969 |                                                                                                                                                                                                  4396718869.484076 |       0.710000 |         0.231422 |          0.000558 |
| NVDA     | reg_clcl |      0.100000 |          -0.000006 |                                                                                                                                                                                                         inf        |       3.870000 |         0.018331 |          0.026599 |
| NVDA     | reg_clop |      0.100000 |           0.000158 |                                                                                                                                                                                                         inf        |       3.870000 |         0.052648 |          0.021757 |
| TSLA     | reg_clcl |      0.040000 |           0.000000 |                                                                                                                                                                                                           0.000605 |       1.750000 |         0.000000 |          0.000000 |
| TSLA     | reg_clop |      0.040000 |           0.000006 |                                                                                                                                                                                                           0.000050 |       0.400000 |         0.000000 |          0.000000 |

## Position Size / PnL per Trade Diagnostics
| ticker   | target   |   trades_csv_rows |      qty_mean |       qty_p95 |        qty_max |      net_mean |        net_p95 |         net_max |         net_min |
|:---------|:---------|------------------:|--------------:|--------------:|---------------:|--------------:|---------------:|----------------:|----------------:|
| JPM      | reg_clcl |            409692 |    225.639422 |    599.000000 |   28144.000000 |    -38.818455 |     290.000000 |    64003.940000 |   -69999.940000 |
| JPM      | reg_clop |                22 |   6540.909091 |  16495.250000 |   21688.000000 | -14248.850000 |   29207.954000 |    44883.070000 |   -69889.530000 |
| MS       | reg_clcl |              2256 |    889.275709 |   1577.250000 |   52141.000000 |    -83.314761 |     684.045000 |    37306.120000 |   -34272.000000 |
| MS       | reg_clop |              5734 |    828.773805 |   1352.000000 |   52141.000000 |    -51.843054 |     620.338500 |    37306.120000 |   -15249.170000 |
| NVDA     | reg_clcl |               684 | 145977.944444 | 619673.900000 | 3453958.000000 | 173554.166491 | 1377829.368000 | 20613178.750000 | -4732250.880000 |
| NVDA     | reg_clop |              1816 |  75180.247797 | 295321.000000 | 3453958.000000 | 100081.437137 |  787601.915000 | 20613178.750000 | -6547077.720000 |
| TSLA     | reg_clcl |                 0 |    nan        |    nan        |     nan        |    nan        |     nan        |      nan        |      nan        |
| TSLA     | reg_clop |                 0 |    nan        |    nan        |     nan        |    nan        |     nan        |      nan        |      nan        |

## Direct Findings
- NVDA `reg_clcl`: trades=701, cum_pnl_raw=118711049.88. Mean qty=145977.94, p95 qty=619673.90, max qty=3453958.00. Large size (volume-linked qty with position_size_mult=1.0) amplifies per-trade PnL.
- TSLA `reg_clcl`: trades=0, trade_rate_vs_signals=0.000000. Pred abs max=0.000605 vs gate median=0.040000; cost-aware gate rejected all signals in this parameter regime.
- TSLA `reg_clop`: trades=0, trade_rate_vs_signals=0.000000. Pred abs max=0.000050 vs gate median=0.040000; cost-aware gate rejected all signals in this parameter regime.
- JPM `reg_clop`: trades=22 out of signals=4560 (rate=0.004825). Gate/prediction scale mismatch heavily suppresses entries.
- NaN rows are from non-finite targets/features after overnight label construction around missing/unaligned print windows; these are now dropped defensively before fit.
