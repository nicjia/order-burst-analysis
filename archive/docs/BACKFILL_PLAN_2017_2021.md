# Backfill Plan: 2017–2021 (do NOT execute yet)

**Goal.** Extend the burst + permanence dataset backward from 2022–2026 to 2017–2021 to buy
**statistical power** for the one marginal positive result — the tick-constrained overnight
mean-reversion (current OOS Sharpe ≈ 0.81, **t = 1.40 over 3.0 years**). This is the
highest-value next step because it needs *data, not a new idea*.

**Expected payoff (if the per-year Sharpe holds).** `t = SR·√(years)`:
- 3.0 yr → t = 1.40 (now). Adding 2017–2021 → ~8 usable years → **t ≈ 0.81·√8 ≈ 2.3** → clears significance.
- Caveat: the per-year Sharpe was uneven/front-loaded; more data raises power but cannot rescue a
  non-stationary signal. This test is decisive either way (confirms real, or confirms it was noise).

---

## 1. Current data state (recon 2026-07-04, read-only)
- **lobster2** `/lobster/YEAR/YYYYMMDD/TICKER.7z` — 2017–2021 all present, ~150k–170k `.7z`/year,
  **clean `TICKER.7z` naming** (identical to 2022+ ⇒ `master_orchestrator.sh` needs no naming change).
- **STILL POPULATING** — `manifest.csv` rewritten Jul 4 01:25; **`AAPL.7z` absent for 2017–2021** right now.
  ⇒ Our universe is NOT fully staged. A coverage check must gate any fetch.
- **Prices already done** — `open_all.csv`/`close_all.csv` span **2016-01-04 → 2026** (2,633 days).
  No price/permanence-label backfill needed; permanence just needs the new burst files + existing prices.

## 2. Scope decision — backfill the RIGHT names, not all 482
The reversal (the target) lives in the **tick-constrained (low-price) subset** (~100 names). Backfilling
only those is ~5× cheaper than the full universe and sufficient to power the headline test.
- **Phase A (recommended first):** the ~100 lowest-price names (from `results/summary/` price ranking)
  + a mid/high-price control group (~40) for the cross-sectional regime comparison. ≈ 140 names.
- **Phase B (optional later):** the full 482 for a complete 2017–2026 COI panel.
- Newer tickers (RIVN, HOOD, COIN, LCID, ABNB, SNOW, …) IPO'd after 2017 and will simply have partial/no
  early coverage — expected; the pipeline already tolerates missing ticker-days.

## 3. Readiness gate (run this WHEN the user says go — before any fetch)
```
# Which of our target names actually have 2017-2021 data on lobster2, per year?
ssh nicjia@lobster2.math.ucla.edu 'for y in 2017 2018 2019 2020 2021; do
  for t in <TARGET_NAMES>; do
    n=$(find /lobster/$y -name "$t.7z" 2>/dev/null | wc -l)
    echo "$y $t $n"; done; done' > backfill_coverage.txt
```
Proceed only for (year, ticker) pairs with adequate day counts (≳ 200/yr). Skip absent ones.

## 4. Pipeline (reuse existing HPC machinery — no new code)
Identical to the 2022–2026 campaign, only the year range changes:
1. **`hoffman2/master_orchestrator.sh`** (run in `tmux` on the DTN): partitions target names into
   batches of 20, `rsync`s `/lobster/{2017..2021}/*/TICKER.7z` from lobster2 to scratch staging,
   submits SGE job arrays. Set the YEAR loop to `2017 2018 2019 2020 2021`.
2. **`hoffman2/sge_compute_worker.sh`** (compute nodes): extract `.7z` → run C++ `data_processor`
   (`-H 1.0 -k 0` …) → **delete raw immediately** (2 TB scratch quota) → emit `bursts_<T>_baseline.csv`
   for the new years. Fast (~3 s/day C++).
3. **Merge** 2017–2021 burst files with existing 2022–2026 into per-ticker files (append; dedupe by date).
4. **`compute_permanence.py`** on the extended burst files (prices already cover 2016+).
5. **Downstream:** re-run `panel_regression.py`, `online_sgd_backtest.py` (→ `debug_trades`), then
   `reversion_sweep.py` / `reversion_walkforward.py` on the full **2017–2026** sample. The walk-forward
   burn-in shifts to 2017; OOS window becomes 2018–2026 (~8 yr).

## 5. Resource estimate
| Item | Phase A (~140 names) | Phase B (full 482) |
|---|---|---|
| Raw fetched (≈40 MB/ticker-day, 5 yr ≈ 1,250 days) | ~7 TB streamed (delete-as-you-go) | ~24 TB |
| Peak scratch (bounded by delete-after-extract) | < 100 GB | < 100 GB |
| C++ compute (3 s/day, SGE 16-wide) | ~10–15 h wall | ~30–40 h wall |
| Transfer time (lobster2→scratch, main bottleneck) | ~1–2 days | ~4–7 days |

## 6. Risks / watch-items
- **Incomplete population** — 2017–2021 still loading; AAPL not yet present. Re-run the §3 gate right
  before starting; expect coverage to keep improving.
- **Transfer is the bottleneck**, not compute. Parallelize `rsync` across batches; monitor lobster2 load.
- **2 TB scratch quota** — the delete-after-extract discipline in `sge_compute_worker.sh` is essential.
- **Survivorship** — using the *current* 482-name list for 2017–2021 introduces mild survivorship bias
  (names that delisted before 2022 are excluded). Acceptable for the reversal test; note it in the paper.
- **Signal non-stationarity** — if the reversal was regime-specific to 2022–2023, more history may *lower*
  the Sharpe. That is itself an informative outcome.

## 7. Execution checklist (when ready)
1. [ ] Run §3 coverage gate → finalize the (year, ticker) list.
2. [ ] Extend YEAR range in `master_orchestrator.sh`; launch in tmux on the DTN for Phase A names.
3. [ ] Monitor SGE arrays; verify burst-file counts + non-empty per batch.
4. [ ] Merge + `compute_permanence.py` for 2017–2021.
5. [ ] Re-run reversion walk-forward on 2017–2026; report OOS Sharpe/t vs the current 1.40.
6. [ ] Update `main.tex` §\ref{sec:reversion} + `FINDINGS_LOG.md` with the extended-sample result.
