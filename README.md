# Hidden liquidity and order flow

Research on non-displayed NASDAQ executions using LOBSTER message data. The main study measures how signed price markouts depend on execution location, trade-sign rules, and event construction.

The [paper](paper.pdf), *The Informational Bifurcation of Hidden Liquidity*, reports the measurement study. The repository also contains the original C++ burst detector and earlier strategy experiments.

## Findings

On the 2023–2024 panel, conventional same-side hidden-execution bursts measured from the burst-ending midpoint gave the following markouts:

| Subset | 3 minutes | 15 minutes | 30 minutes |
|---|---:|---:|---:|
| Aggressive, away from midpoint | +2.09 bps | +2.24 bps | +2.25 bps |
| At midpoint, tick-signed | −0.42 bps | −0.64 bps | −0.71 bps |

The magnitude depends on construction and signing. Unambiguously signed outside-quote prints without clustering give a more conservative three-minute estimate of +0.621 bps. Denying burst formation access to contemporaneous price information reduces the measured footprint further.

These are signed markouts. They do not establish causal permanent impact or an executable trading edge. Classifiers disagree when they force signs onto midpoint prints; that disagreement is part of the result.

Current definitions, sensitivity checks, and excluded results are in [VERIFIED_RESULTS.md](VERIFIED_RESULTS.md). Script and run mappings are in [RESULTS_PROVENANCE.md](RESULTS_PROVENANCE.md).

## Run a synthetic example

```sh
python -m pip install -r requirements-core.txt
python examples/synthetic_book.py
python -m unittest discover -s tests -p 'test_*.py'
make
```

The example reconstructs an invented book from submissions, executions, and cancellations. It requires no licensed data. `make` builds the original C++17 burst detector.

See [REPRODUCING.md](REPRODUCING.md) for dependencies and the distinction between code checks and historical replication.

## Research design

- Reconstruct displayed quotes from NASDAQ messages.
- Separate visible and hidden executions and state the signing convention for each analysis.
- Compare event construction and quote timing rather than treating the observed footprint as invariant.
- Aggregate within name-day and use the day as the inference unit.
- Report Newey–West inference, placebo comparisons, and construction sensitivity.
- Keep rejected strategy results separate from the surviving measurement results.

The original C++ detector uses a self-exciting decaying counter as a burst-clustering heuristic. The hidden-liquidity measurements use Python reconstruction and extraction scripts.

## Files

| Path | Contents |
|---|---|
| `paper.tex` | Manuscript source |
| `main.tex` | Longer research record |
| `VERIFIED_RESULTS.md` | Current results and exclusions |
| `RESULTS_PROVENANCE.md` | Run and table provenance |
| `src_py/` | Reconstruction, extractors, inference, diagnostics |
| `src_cpp/` | Original burst detector |
| `hoffman2/` | Cluster job scripts |
| `archive/`, `passive/` | Earlier experiments |

## Limits

LOBSTER reconstructs NASDAQ alone, so local quote movement cannot be cleanly separated from adjustment to a consolidated quote. Trade signs and burst formation can condition on price movement. The main panel's universe and archive coverage limit generalization. Trader identities and parent orders are not observed.

Earlier strategy work failed statistical or design checks documented in the verified record. Historical tables require licensed data; the synthetic tests do not validate those empirical estimates.
