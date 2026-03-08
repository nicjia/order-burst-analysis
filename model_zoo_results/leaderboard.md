# Model Zoo — Permanence Prediction Leaderboard

Generated: 2026-03-07T09:59:25.451858

## Classification (target: φ > 1)

| Rank | Model | Target | AUC | Accuracy | F1 | Brier | Time |
|------|-------|--------|-----|----------|----|-------|------|
| 1 | LightGBM | cls_5m | 0.8553 | 0.8197 | 0.8892 | 0.1211 | 268s |
| 2 | LightGBM (Optuna) | cls_5m | 0.8551 | 0.8197 | 0.8893 | 0.1211 | 5859s |
| 3 | Random Forest | cls_5m | 0.8551 | 0.8199 | 0.8896 | 0.1212 | 1309s |
| 4 | Extra Trees | cls_5m | 0.8532 | 0.8102 | 0.8886 | 0.1252 | 464s |
| 5 | AdaBoost | cls_5m | 0.8532 | 0.8185 | 0.8892 | 0.2250 | 9130s |
| 6 | LogReg (L2) | cls_5m | 0.8526 | 0.8182 | 0.8875 | 0.1225 | 266s |
| 7 | LogReg (ElasticNet) | cls_5m | 0.8523 | 0.8180 | 0.8879 | 0.1227 | 466s |
| 8 | LogReg (L1) | cls_5m | 0.8522 | 0.8180 | 0.8879 | 0.1227 | 872s |
| 9 | XGBoost | cls_5m | 0.8514 | 0.8200 | 0.8893 | 0.1221 | 102s |
| 10 | HistGradientBoosting | cls_5m | 0.8480 | 0.8134 | 0.8843 | 0.1244 | 476s |
| 11 | XGBoost (Optuna) | cls_5m | 0.8464 | 0.8129 | 0.8841 | 0.1253 | 208s |
| 12 | Linear SVM (SGD) | cls_5m | 0.8460 | 0.8158 | 0.8871 | 0.1250 | 118s |
| 13 | Ridge Classifier | cls_5m | 0.8439 | 0.8134 | 0.8857 | 0.1265 | 93s |
| 14 | Random Forest | cls_10m | 0.8298 | 0.8057 | 0.8822 | 0.1304 | 1051s |
| 15 | LightGBM | cls_10m | 0.8293 | 0.8049 | 0.8822 | 0.1305 | 243s |
| 16 | Extra Trees | cls_10m | 0.8283 | 0.7976 | 0.8823 | 0.1335 | 356s |
| 17 | LightGBM (Optuna) | cls_10m | 0.8282 | 0.8048 | 0.8824 | 0.1308 | 7141s |
| 18 | AdaBoost | cls_10m | 0.8280 | 0.8042 | 0.8818 | 0.2282 | 9317s |
| 19 | LogReg (L2) | cls_10m | 0.8279 | 0.8043 | 0.8803 | 0.1315 | 302s |
| 20 | LogReg (L1) | cls_10m | 0.8277 | 0.8042 | 0.8808 | 0.1318 | 402s |
| 21 | LogReg (ElasticNet) | cls_10m | 0.8277 | 0.8042 | 0.8808 | 0.1318 | 566s |
| 22 | Ridge Classifier | cls_10m | 0.8200 | 0.8006 | 0.8788 | 0.1350 | 135s |
| 23 | HistGradientBoosting | cls_10m | 0.8183 | 0.7989 | 0.8771 | 0.1348 | 614s |
| 24 | XGBoost | cls_10m | 0.8169 | 0.8054 | 0.8818 | 0.1340 | 96s |
| 25 | XGBoost (Optuna) | cls_10m | 0.8156 | 0.7972 | 0.8759 | 0.1361 | 222s |
| 26 | Linear SVM (SGD) | cls_10m | 0.8139 | 0.7977 | 0.8775 | 0.1374 | 112s |
| 27 | Random Forest | cls_3m | 0.7883 | 0.7563 | 0.8415 | 0.1602 | 1025s |
| 28 | LightGBM | cls_3m | 0.7877 | 0.7552 | 0.8416 | 0.1604 | 257s |
| 29 | LightGBM (Optuna) | cls_3m | 0.7870 | 0.7553 | 0.8413 | 0.1607 | 5469s |
| 30 | AdaBoost | cls_3m | 0.7865 | 0.7549 | 0.8396 | 0.2366 | 7144s |
| 31 | Extra Trees | cls_3m | 0.7861 | 0.7485 | 0.8443 | 0.1630 | 147s |
| 32 | LogReg (L1) | cls_3m | 0.7858 | 0.7538 | 0.8387 | 0.1616 | 464s |
| 33 | LogReg (ElasticNet) | cls_3m | 0.7858 | 0.7538 | 0.8387 | 0.1616 | 545s |
| 34 | LogReg (L2) | cls_3m | 0.7858 | 0.7538 | 0.8382 | 0.1615 | 185s |
| 35 | XGBoost | cls_3m | 0.7849 | 0.7563 | 0.8413 | 0.1619 | 92s |
| 36 | Ridge Classifier | cls_3m | 0.7810 | 0.7513 | 0.8388 | 0.1635 | 94s |
| 37 | HistGradientBoosting | cls_3m | 0.7795 | 0.7494 | 0.8343 | 0.1636 | 459s |
| 38 | XGBoost (Optuna) | cls_3m | 0.7777 | 0.7486 | 0.8337 | 0.1645 | 212s |
| 39 | Linear SVM (SGD) | cls_3m | 0.7755 | 0.7485 | 0.8388 | 0.1654 | 123s |
| 40 | LightGBM (Optuna) | cls_1m | 0.6614 | 0.6211 | 0.6677 | 0.2282 | 4373s |
| 41 | Random Forest | cls_1m | 0.6606 | 0.6204 | 0.6688 | 0.2283 | 2020s |
| 42 | AdaBoost | cls_1m | 0.6599 | 0.6199 | 0.6620 | 0.2477 | 10286s |
| 43 | LightGBM | cls_1m | 0.6595 | 0.6200 | 0.6683 | 0.2287 | 218s |
| 44 | LogReg (L1) | cls_1m | 0.6589 | 0.6204 | 0.6643 | 0.2292 | 716s |
| 45 | LogReg (ElasticNet) | cls_1m | 0.6589 | 0.6203 | 0.6643 | 0.2292 | 376s |
| 46 | LogReg (L2) | cls_1m | 0.6588 | 0.6203 | 0.6640 | 0.2292 | 134s |
| 47 | XGBoost | cls_1m | 0.6586 | 0.6202 | 0.6656 | 0.2294 | 88s |
| 48 | Extra Trees | cls_1m | 0.6585 | 0.6188 | 0.6763 | 0.2293 | 676s |
| 49 | Ridge Classifier | cls_1m | 0.6573 | 0.6194 | 0.6710 | 0.2296 | 105s |
| 50 | HistGradientBoosting | cls_1m | 0.6518 | 0.6144 | 0.6561 | 0.2313 | 452s |
| 51 | Linear SVM (SGD) | cls_1m | 0.6517 | 0.6150 | 0.6719 | 0.2309 | 173s |
| 52 | XGBoost (Optuna) | cls_1m | 0.6475 | 0.6117 | 0.6538 | 0.2332 | 188s |
| 53 | KNN (k=50) | cls_1m | 0.6383 | 0.6038 | 0.6580 | 0.2341 | 148s |
| 54 | LightGBM (Optuna) | cls_close | 0.5581 | 0.5273 | 0.6676 | 0.2497 | 2514s |
| 55 | XGBoost | cls_close | 0.5509 | 0.5426 | 0.6274 | 0.2589 | 72s |
| 56 | LogReg (ElasticNet) | cls_close | 0.5448 | 0.5330 | 0.6472 | 0.2499 | 881s |
| 57 | LogReg (L1) | cls_close | 0.5448 | 0.5330 | 0.6472 | 0.2499 | 4103s |
| 58 | LogReg (L2) | cls_close | 0.5437 | 0.5322 | 0.6462 | 0.2501 | 245s |
| 59 | LightGBM | cls_close | 0.5333 | 0.5229 | 0.6680 | 0.2575 | 112s |
| 60 | Ridge Classifier | cls_close | 0.5316 | 0.5272 | 0.6574 | 0.2501 | 102s |
| 61 | Extra Trees | cls_close | 0.5255 | 0.5249 | 0.6461 | 0.2516 | 510s |
| 62 | Linear SVM (SGD) | cls_close | 0.5170 | 0.5232 | 0.6641 | 0.2503 | 170s |
| 63 | KNN (k=50) | cls_close | 0.5145 | 0.5170 | 0.5925 | 0.2593 | 119s |
| 64 | AdaBoost | cls_close | 0.5033 | 0.5172 | 0.6409 | 0.2496 | 9336s |
| 65 | Random Forest | cls_close | 0.4980 | 0.5101 | 0.6125 | 0.2688 | 3342s |
| 66 | HistGradientBoosting | cls_close | 0.4788 | 0.4928 | 0.5718 | 0.3657 | 422s |
| 67 | XGBoost (Optuna) | cls_close | 0.4770 | 0.4912 | 0.5684 | 0.3783 | 200s |
