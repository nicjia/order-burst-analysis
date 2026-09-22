# Result definitions and code paths

The main hidden-execution panel contains 474 NASDAQ names over 2023–2024. The original requested universe and the usable measurement sample are different counts.

| Measurement | Result and interpretation | Record |
|---|---|---|
| Aggressive hidden executions | +2.09 bps at three minutes under the baseline quote-signing and burst convention | [Verified results §1.1](VERIFIED_RESULTS.md) |
| Conservative unclustered comparison | +0.621 bps on outside-prequote prints; construction and signing change the level | [Verified results](VERIFIED_RESULTS.md), §1.2 |
| Intraday persistence | +2.023 bps at three minutes and +2.014 to close under the reported gross-markout convention | [Verified results](VERIFIED_RESULTS.md), §1.10 |
| Hidden flow alongside visible OFI | Joint hidden coefficient +0.198, t=22.9, in the stated short-horizon regression | [Verified results](VERIFIED_RESULTS.md), §1.11 |
| Tick-regime reversal | Suggestive under corrected day-level inference; the earlier stronger cross-sectional statistic is superseded | [Methods and reproducibility](METHODS_AND_REPRODUCIBILITY.md) |

The [paper](paper.pdf) describes the measurement study. [Provenance](RESULTS_PROVENANCE.md) maps extraction and aggregation runs to the tables.

## Interpretation

An intraday markout that remains positive through the close is not by itself a causal estimate of permanent impact. Trade-sign classifiers disagree on pooled hidden executions; the level is not invariant across classifiers. The results distinguish aggressive and midpoint populations and document how construction affects the measurement.

## Reconstruction and detector changes

`src_py/execution_packets.py` consolidates timestamp-level execution messages before fragment construction. Visible executions supply native aggressor signs. Type-5 Direction is ignored; hidden-only packets are signed only outside the pre-event quote, otherwise left ambiguous. `fragment_reconstruction.py` forms fragments from packets rather than treating each message as a separate child order.

The original C++ detector remains for historical replication. Its nonzero `-k` gate is now rejected because it selected membership using future returns. Future returns may remain output labels, but cannot select the events. Its self-exciting counter is a clustering heuristic, not a fitted Hawkes intensity model. The implementation retains order-book state and intraday buffers; no constant-memory guarantee is claimed for the full pipeline.

Run `make` followed by `python -m unittest discover -s tests -p 'test_*.py'`. Tests use invented messages and check packet grouping, ambiguous signs, quote lookup, and rejection of future-return selection. They do not rerun the historical licensed-data studies.
