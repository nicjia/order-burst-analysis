# Order-burst research on NASDAQ ITCH (LOBSTER)

Empirical market microstructure. Do clustered executions ("bursts") in the NASDAQ order book reveal who is
trading and where price goes next? Nicholas Jiang, UCLA, supervised by Prof. Mihai Cucuringu.

**Where things stand.** Bursts are real execution algorithms. Identical child sizes recur far above chance, and
the same algorithm returns on later days, on the same side. Their price impact follows the concave metaorder
shape. But every directional signal built from them is a fixed fraction of the bid–ask spread, so none survives
trading costs. The intraday forecasts that do exist come from the stock's own intraday reversal, not from burst
information. Every number is in `VERIFIED_RESULTS.md`, with its code and job.

## Start here

| file | what it is |
|---|---|
| **`VERIFIED_RESULTS.md`** | the only numbers that may be cited: one section per result, each with its job IDs; §2 lists every excluded result and why |
| **`RESULTS_PROVENANCE.md`** | result → script (sha256) → Hoffman2 job → stored panel, plus known gaps |
| `DEFINITIONS_TRIED.md` | every burst definition and configuration tried (the multiple-testing ledger) |
| `studies/` | one folder per study: its pre-registered design and its results |
| `main.tex` | the full internal record (all studies, including failed ones) |
| `paper.tex`, `Response_to_Referees.md` | the hidden-liquidity submission draft and referee response (see the note below) |
| `PROJECT_GOALS.md` | the current mandate |
| `archive/MANIFEST.md` | everything moved out of the way in the 2026-09-25 cleanup, and why |

## How to verify a result

1. Find it in `VERIFIED_RESULTS.md`. The heading names the SGE array(s) and the study.
2. Look it up in `RESULTS_PROVENANCE.md` for the script, its sha256 prefix, the job ID and the output location.
3. Check that the script has not changed: `shasum -a 256 src_py/<script>.py | cut -c1-12` must match the ledger.
   Kept code is deliberately left byte-identical, so the pinned hashes still apply.
4. Re-run it. Extractors need the LOBSTER archive, reachable only from Hoffman2 (`hoffman2/*.sh` submit them).
   Aggregators and every local analysis run from the stored panels. Unit tests: `python3 -m pytest tests/`.
5. The pre-registered design for the study is in `studies/<study>/`. Every amendment is dated there, including
   the ones made after a read.

## Run the code checks without licensed data

```sh
python -m pip install -r requirements-core.txt
make                                   # builds the original C++17 burst detector
python -m unittest discover -s tests -p 'test_*.py'
python examples/synthetic_book.py      # reconstructs an invented order book
```

These check code behavior on invented inputs; they do not reproduce the empirical findings, which need the
licensed LOBSTER archive. [REPRODUCING.md](REPRODUCING.md) has the details, and [RESULTS.md](RESULTS.md) the
definitions and code paths for the hidden-liquidity study. The same checks run on every push (`.github/workflows/`).

## Repository layout

| path | contents |
|---|---|
| `src_py/` | all analysis code behind a recorded result, plus the shared modules it imports. **`src_py/INDEX.md`** lists every script by study, with a one-line description |
| `src_cpp/`, `Makefile` | the original C++ burst detector. Legacy: it signs every hidden print as a sell (`studies/p4_revisit/PRIOR_WORK_INVENTORY.md` §3) and is kept only for provenance |
| `hoffman2/` | SGE job scripts for the kept code |
| `studies/<study>/` | design and results documents; `studies/burst_forecasting/` also holds its own code and job scripts |
| `examples/`, `requirements-core.txt`, `REPRODUCING.md`, `RESULTS.md` | the no-data code checks and the hidden-liquidity result definitions |
| `tests/` | unit tests for packets, bursts, aggregation, inference and the leakage firewalls |
| `config/` | frozen gate and model files cited in the provenance ledger |
| `measurements/` | panel universes and date lists, and `data/earnings_dates.csv` (§1.12) |
| `universes/` | legacy ticker universes (`full_500.txt` is survivorship-biased; see §2 of the ledger) |
| `figures/` | figures for `main.tex` and `paper.tex` |
| `archive/` | superseded code, job scripts, documents and the legacy pipeline (`archive/MANIFEST.md`) |
| `data/`, `results/` | **gitignored**. Licensed extracts (LOBSTER derivatives, CRSP, Compustat, TAQ, 13F) and every panel |

Top-level `*_all.csv`, `*_update_2025_2026.csv`, `Yearly/`, `yearly-clop/` and `data_processor` are also
gitignored. They are local price panels and a build product. Several kept scripts read the price panels from the
repository root, so they stay where the scripts expect them.

## Studies

| study | question | ledger |
|---|---|---|
| hidden liquidity (`paper.tex`) | is hidden liquidity informed, and is that identified? | §1.1–1.14, §1.22–1.23 |
| spread-scaling law (ledger and `main.tex`) | why no directional burst signal pays for the spread | §1.5, §1.21 |
| earlier strategy checks (ledger and `main.tex`) | multi-day reversion, trade count and volatility, point-in-time reversal | §1.15–1.17 |
| `two_avenue` | do packet fragments carry price discovery or continuation? | §1.18–1.20 |
| `burst_information` | do burst features forecast flow or returns? can bursts be reconstructed? | §1.24–1.26 |
| `fingerprint` | are bursts real single algorithms? which definition keeps them together? | §1.27 |
| `program_evidence` | what kind of flow are program bursts? | §1.28–1.29 |
| `metaorder` | child clusters, parent linkage, passive orders | §1.30 |
| `p4_revisit` | the original MATH 279 "informed bursts" proposal, run without leakage | §1.31 |
| `fingerprint_multiday` | do algorithms return on later days? impact shape; cross-name flow | §1.32, §1.35 |
| `forced_flow` | does burst impact hold less on quarter-end days? | §1.33 |
| `daily_labels` | does burst flow track daily retail or institutional imbalances? | §1.34 |
| `earnings_flow` | does pre-announcement burst flow predict announcement returns? | null (design only) |
| `burst_forecasting` | the 279 model rebuilt leak-free: permanence, forecasting, trading, oracle checks | §1.36; active |

**Note on `paper.tex`.** Its headline magnitudes (+2.09 / −0.42 bps) are message-level. On reconstructed
economic orders the same bifurcation is +0.603 / −0.056, and the sharp bounds over unsigned hidden prints contain
zero (§1.22). Cite §1.22 for magnitudes.

## Limits

LOBSTER reconstructs NASDAQ alone, so a local quote move cannot be cleanly separated from adjustment to the
consolidated quote. Trade signs and burst formation can condition on price movement, which several studies here
test for directly. Archive coverage limits generalization. Trader identities and parent orders are never observed:
every institutional statement is an association with external labels, not an identification.

## Rules for this repository

- The repository is **public**. Never commit LOBSTER derivatives, WRDS extracts or anything under `data/` or
  `results/`.
- WRDS credentials are read only by `src_py/wrds_access.py` from the gitignored `.env`. They are never printed or
  passed on a command line.
- Designs are frozen before outcomes are read. Validation and test cells are read once. Every amendment is dated
  in the study's design document.

## Known gaps in the record

- `RESULTS_PROVENANCE.md` cites `src_py/agg_sw3.py` and `config/informed_models_2023.json`. Neither is in this
  working copy or in git history. Both were already absent before the 2026-09-25 cleanup; the cleanup did not
  remove them.
- Between about 20% (2022–25) and 66% (2012–16) of requested name-days are absent from the LOBSTER archive.
  Coverage is quoted with each result.
