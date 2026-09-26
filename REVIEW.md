# Review guide

For anyone reviewing this code, by hand or with an automated reviewer. The code turns NASDAQ order-book data
(LOBSTER) into the estimates in `VERIFIED_RESULTS.md`. A bug here is anything that can change a reported estimate,
the sample behind it, or its significance. The licensed data is not in the repository. `tests/`,
`studies/burst_forecasting/tests/` and `examples/synthetic_book.py` run on invented data.

## Reviewing in parts

The code is about 43,000 lines, more than one diff review can take, so `tools/review_parts.py` splits it into six
parts. For each part, branch `review/part-N` compared with tag `review-base-N` shows that part's files as added.
**Everything in such a diff is existing code under review, not new code.** Report every problem in it, including
long-standing ones. Files outside the diff are context: read them (shared modules, callers, job scripts, the study
designs in `studies/`), but report only on the diff.

## Important: report these

1. **Look-ahead.** A feature, filter, threshold, normaliser or event definition that uses information from after
   the decision time. Forms seen in this project:
   - whole-day statistics (quantiles, scales, guards) used for intraday decisions;
   - rolling windows that include the current day or event;
   - a run's end used before it is knowable (a run with gap G is known only G seconds after its last trade);
   - `groupby(...).transform` over a whole day;
   - merges or sorts that pull later rows forward;
   - targets whose window starts before the decision.

   Decision rules: in `studies/burst_forecasting/`, act only once the burst is knowable (its `README.md`). In
   `studies/p4_revisit/`, T_dec = max(t_b + 600 s, t_e + 10 s).
2. **Train/test contamination.** Any fit, tuning, feature choice, winsorisation or scaling computed on validation
   or test cells. Splits are disjoint in names and in years. In P4, DEV is 2017–19 group 0, and VAL (2020–21) and
   TEST (2022–25) are groups 1–2, with groups keyed on PERMNO. Each study's design lists its split, and its test
   cells are read once.
3. **Alignment, sign and units.** Check the burst-side sign, mid versus trade price, and LOBSTER prices (dollars ×
   10,000). Clocks count seconds after midnight: 09:30 is 34,200 and 16:00 is 57,600. Horizons are measured from the
   decision, not from the burst start. Check bps versus fractions, and that next-day prices are on the event day's
   split basis.
4. **Inference.** Use the unit each design specifies, usually the day: equal weights within a name-day, a daily
   cross-sectional mean, then Newey–West with 10 lags. Flag:
   - clusters on the wrong key;
   - overlapping horizons treated as independent;
   - multiple testing counted against the wrong family (`DEFINITIONS_TRIED.md`, and the Holm families in the
     designs).
5. **Silent sample changes.**
   - Models that are compared on different rows (NaN handling, inner joins).
   - Days dropped without a count.
   - Ticker joins where PERMNO is needed.
   - Universes chosen with hindsight.
6. **Code that does not do what its documentation says.** Cite the code (`file:line`) and the document: the
   `VERIFIED_RESULTS.md` section or the study design.
7. **Real-data paths that crash or mis-handle cases.** Examples: empty days, halts, locked or crossed quotes,
   zero spread, missing book levels.

## Nit at most, or do not report

- Style, naming, refactors, docstrings, and speed that does not change a number. Report at most 10 nits per review.
- `archive/`, `config/*.json` (frozen model outputs), documents and figures.
- Behaviour that is documented and accepted. Report it only if the documentation is wrong:
  - the legacy C++ detector in `src_cpp/` signs every hidden print as a sell
    (`studies/p4_revisit/PRIOR_WORK_INVENTORY.md` §3);
  - `universes/full_500.txt` is survivorship-biased (`VERIFIED_RESULTS.md` §2);
  - P4's tCLOSE target overlaps the burst window and is used only in the leakage demonstration.
- Scripts listed under "Legacy pipeline" in `src_py/INDEX.md` have no current result. Label findings there
  "legacy".

## Evidence and reporting

- Every finding needs `file:line` and a concrete path to the wrong number. Say which variable is computed from which
  rows at which time, and what that does to the reported statistic.
- A synthetic input that shows the problem is the strongest evidence. A guess from a name is not a finding.
- Scripts are hash-pinned in `RESULTS_PROVENANCE.md`. Name the ledger section whose number a bug affects rather
  than proposing a silent edit.
- Open the summary with a tally: leakage and split, alignment and units, inference, other. If there are none, say
  "no protocol violations found".
