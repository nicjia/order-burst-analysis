---
name: order-burst-project
description: Full context for the LOBSTER/NASDAQ order-burst microstructure research project — Hoffman2 and lobster2 access patterns, SGE job conventions, repo layout, established findings, and the traps that have already cost re-runs. Load this at the start of any session touching /Users/nick/order-burst-analysis.
---

# Order-Burst Analysis — Project Context

Empirical market-microstructure research on NASDAQ ITCH (LOBSTER) message data.
Author: Nicholas Jiang (nicjia@g.ucla.edu), UCLA.

## 1. What this project is, and how it got here

**Original goal (still the goal):** find a tradeable signal in intraday order-submission
bursts — temporally clustered same-side executions, the visible signature of an institution
splitting a large order.

**Arc:**
1. Built a C++ burst detector + ML pipeline (Optuna, online SGD) targeting the overnight
   close-to-open move. Result: information coefficient −0.001 (t=−0.4). A clean null.
2. A conditional reversal strategy in tick-constrained names looked alive (~0.8 Sharpe) but
   failed on three counts: deflated-Sharpe, an ex-post universe, and calibration overlapping
   evaluation in calendar time.
3. Pivoted from prediction to **measurement**: do bursts leave a permanent price footprint?
   Of four reconstructions tried, only hidden-execution (LOBSTER type-5) clustering showed
   anything, so the project became a hidden-liquidity paper.
4. Found the **bifurcation**: hidden prints away from the midpoint carry a positive markout,
   prints at the midpoint carry a small negative one. Pooling them returns ~zero, which is
   why the four standard trade-sign classifiers disagree by 2 bps on identical data.
5. Six adversarial referee rounds hardened it and cut the headline from +2.09 to +0.62.

## 2. Access

### Hoffman2 (compute)
Non-interactive SSH REQUIRES overriding the host config, or it hangs:
```bash
ssh -o RemoteCommand=none -o BatchMode=yes hoff '<command>'
```
Output carries a harmless `Pseudo-terminal will not be allocated` line — pipe through
`grep -v Pseudo`.

rsync MUST override RemoteCommand too, or it fails with "Cannot execute command-line and
remote command":
```bash
rsync -a -e "ssh -o RemoteCommand=none" <src> hoff:/u/scratch/n/nicjia/order-burst-analysis/
```

Project root on cluster: `/u/scratch/n/nicjia/order-burst-analysis`

Python environment:
```bash
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
```
System `python3` has numpy/pandas and works; `.venv/bin/activate` is unreliable.

LaTeX (there is NO local pdflatex — compile on the cluster):
```bash
. /u/local/Modules/default/init/bash && module load texlive
pdflatex -interaction=nonstopmode paper.tex; bibtex paper   # x2, then pdflatex again
```

### lobster2 (raw data) — reachable ONLY from Hoffman
```bash
ssh -o RemoteCommand=none hoff 'ssh nicjia@lobster2.math.ucla.edu "<cmd>"'
```
- Raw files: `/lobster/YEAR/YYYYMMDD/TICKER.7z`, extracted with `~/bin/7z`
- Archive spans **2012–2026**, ~1,900–2,100 tickers per date
- `/lobster/manifest.csv` — 256k rows, one per downloaded chunk, format
  `R<id>_<TICKER>_<start>_<end>_0.zip,<ndates>,<date1>,<date2>,...`. Use this to derive
  per-ticker date spans rather than scanning the filesystem.
- Download logs: `~/nasdaq/`, `~/nyse/` on lobster2 (`download_success.txt`,
  `tickers.csv`). **`tickers.csv` is the REQUEST list, not what was downloaded** — do not
  infer coverage from it (this error cost a wrong conclusion about survivorship).

### GitHub
`https://github.com/nicjia/order-burst-analysis` — **PUBLIC**. Never commit licensed data
extracts. Local clone at `/Users/nick/order-burst-analysis` is authoritative for code.
The cluster also has a git repo (initialised at commit `bc73ebd`); `/u/scratch` is NOT
durable and has purged `src_py/` mid-session — the repo and the local clone are the only
safe copies.

**User rule: do NOT make git commits or push unless explicitly asked. Stage only.**

## 3. SGE job conventions

Every extraction follows the same shape — one array task per ticker, `xargs -P6` over dates:
```bash
qsub -cwd -V -N <name> -l h_data=8G,h_rt=14:00:00 -t 1-474 -tc 60 \
  -o results/<g>/log -e results/<g>/log hoffman2/<driver>.sh
```
Drivers read `results/<g>/universe.txt` and `results/<g>/dates.txt`, stream each ticker-day
from lobster2, run `python3 src_py/<extractor>.py --msg <file> --ticker <TK>`, and write
`results/<g>/out/$TK.csv`. New drivers are made by `sed`-ing an existing one.

Extractors all import `src_py/burst_alt.py`:
`reconstruct(msg_path) -> (bt, bmid, bb, ba, bbsz, basz, ofi, trades)`, plus `mid_at(bt,bm,times)`
and `bbo_at(bbo_t, arr1, arr2, q)` (generic — passing sizes returns sizes). `SCALE=10000.0`.

**PRE-FLIGHT, every time:** `test -f src_py/burst_alt.py` on the cluster before launching.
Scratch has purged it.

Timing: ~5h for 474 names x 502 dates at `-tc 50-60`. Cost is dominated by `reconstruct`
(AMD 4.3M messages = 80s/day; small names 7s/day), not by the analysis.

Do NOT stage aggregates in `/tmp` — login nodes differ and files vanish between calls. Write
`results/<g>/all.csv`.

Inference convention everywhere: the DAY is the unit. Equal-weight within name-day, average
cross-sectionally, Newey–West (10 lags) on the daily-mean series. Drop name-days with
|markout| > 1000 bps.

## 4. Repo layout

- `paper.tex` / `paper.pdf` — the ~25-page submission draft. Only defensible results.
- `main.tex` / `main.pdf` — ~51-page internal record. **User rule: keep ALL findings here;
  never spin a data-analysis error into a "learning experience"; no numbers from erroneous
  analysis in either document.**
- `RESULTS_PROVENANCE.md` — table -> script -> SGE array -> results dir -> row counts, plus an
  honest list of gaps. Update it when new results land.
- `src_py/` — extractors (`hidden_*.py`, `burst_zoo*.py`, `idea_zoo*.py`) and aggregators
  (`agg_*.py`). `src_cpp/` — original C++ burst detector.
- `hoffman2/*.sh` — SGE drivers. `measurements/data/` — bulk row dumps, gitignored (223MB).

## 5. Established findings — do not re-derive

**The spread-scaling law (closes the directional search).** 68 burst definitions tested.
Every directional signal is ~0.7x the half-spread: `mk3 = 0.033 + 0.709 * halfspread`,
correlation +0.79, **intercept zero**, 0 of 40 names with `mk3 > 2 * halfspread`. Mechanism:
Glosten–Milgrom as arithmetic — a signal that identifies informed flow IS the spread's
compensation for informed flow, so it can never pay for crossing it. Do not iterate further
on directional definitions.

**The bifurcation (the paper's contribution).** Aggressive hidden prints +2.09 bps vs
at-midpoint −0.42, but the LEVEL is convention-dependent: +2.09 conventionally measured,
+0.62 signed unambiguously with no burst formed, ~0 when formation is also denied sight of
price. Survives a matched tick-rule placebo (placebo recovers 5.4%). The convention-dependence
is the claimed contribution; the magnitude is provisional.

**Trade count forecasts volatility.** +0.10 incremental R² over an intraday HAR, 473 names,
holds temporally out-of-sample. But it is VISIBLE trade count — the hidden term has mean
t = −0.25 and 1 of 472 names significant. This is a replication of Jones–Kaul–Lipson (1994),
not a new result.

**Delisted names ARE in the archive.** SIVB, SBNY, FRC, ATVI, VMW, SGEN, SPLK, PXD, TWTR,
XLNX, ZNGA, CERN, ABMD, HZNP, CTLT — each series truncated exactly at its corporate-action
date. Survivorship is fixable.

## 6. Traps that have already cost re-runs

- **Circularity in burst formation.** Defining a burst as a run of same-side prints, where
  side = price vs the CONTEMPORANEOUS midpoint, lets a drifting mid manufacture bursts. Form
  clusters on timing/size/message-type only.
- **Circular conditioning.** Conditioning on a quote move inside the first second and then
  measuring a markout that starts at the print conditions on most of the outcome (the
  footprint is ~80% impounded within 1s). Separate conditioning and measurement windows.
- **Stale vs contemporaneous quotes.** Always use `bbo_at(..., t - 1e-3)` for anything the
  print could have moved. Using `bbo_at(..., t)` made "outside the quote" endogenous and
  invalidated a whole run (`hid_tp_v1`, array 14225689 — do not cite).
- **Native signs exist.** ITCH discloses Direction for message types 1/2/3/4. ONLY type-5
  hidden prints are unsigned. Prefer native signs and infer nothing.
- **R² is non-decreasing.** "N of N names show a positive increment" is meaningless when a
  regressor is added. Check the t-statistic.
- **Non-overlapping sorts are phase-dependent.** A 20-day non-overlapping cohort sort spanned
  −36 to +25 bps purely by starting offset. Use overlapping calendar-time portfolios.
- **`close_all.csv` has integer dates** (20160104). `pd.to_datetime` reads them as epoch
  nanoseconds and silently yields 1970. Cast to str with `format="%Y%m%d"`.
- **Single-day results are worthless.** An "iceberg" signal showed +6.9 bps on one day and
  −0.2 across 499.

## 7. Open, and blocked

- **Untested branches:** uninformed/forced flow (index rebalance, expiry — needs calendars);
  order anticipation (only crude proxies tried); passive execution (never simulated).
- **The original reversal strategy** was never killed by the spread law — it operates at daily
  frequency, paying spread once per rebalance. Its three defects: DSR failure (not fixable),
  ex-post universe (**now fixable**), calibration/evaluation overlap (**fixable — 2022 data
  exists**).
- **Permanently blocked:** SIP/NBBO signing (LOBSTER is NASDAQ-only); options data for
  trading any volatility signal.

## 8. Working style the user expects

- Do not ask permission before running tests; iterate autonomously toward a tradeable burst
  definition.
- Keep a running count of definitions tried — it is the input to the deflated-Sharpe hurdle.
- Validate on names disjoint from the exploration set, and split temporally as well as
  cross-sectionally (cross-sectional-only OOS is the exact flaw that sank the first strategy).
- Report negative results plainly; never present a favourable cut as the headline.
- Be economical with tokens.
