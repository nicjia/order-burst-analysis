# P4 revisit v1: persistent-impact bursts, done without leakage — pre-registration

Frozen 2026-09-15, before any P4 outcome was computed on the data below.

- **Background:** `studies/p4_revisit/PRIOR_WORK_INVENTORY.md` (what was tried and why it failed) and `studies/p4_revisit/P4_REVISIT_PLAN.md` (the plan this freezes).
- **Proposal:** `UCLA_279__P4_informed_bursts.pdf` (v1, 2026-01-23). Equation numbers below refer to it.

**User decisions (2026-09-14).**
- Years from 2017 onward, and earlier if the archive allows.
- Both trade bursts and submission bursts, compared to see which is more promising.
- NYSE- and NASDAQ-listed names.
- SEC 13F download approved.
- Start with everything, including the Q0 audit.
- A null Q4 is acceptable if Q2 separates institutional from other flow.

Any change after this freeze is logged under **Amendments**, with its date and whether any outcome had been seen.

---

## 1. Hypotheses

- **H1 (Phase I, persistence).** Bursts that pass the P4 filter (eq. 3.3) keep their displacement after the decision time T_dec more than:
  - comparable bursts that fail it;
  - pseudo-bursts put through the same filter at random times.
- **H2 (parents).** Persistent-impact bursts are children of parent orders:
  - more same-side identical-size linkage than other bursts;
  - more loading on institutional trading proxies (13F, mutual-fund holdings, index events);
  - no more loading on retail flow.
- **H3 (Phase II).** Post-T_dec persistence is predictable from information available by T_dec.
- **H4 (Phase III).** The daily informative flow S_i,t (eq. 4.4) predicts tCLOSE, CLOP and CLCL returns beyond:
  - unfiltered flow;
  - the stock's own return;
  - standard controls.
- **H5.** Where any effect lives: listing exchange, NASDAQ coverage share, size, tick constraint, year.
- **Family question.** Which event family, trade (T) or submission (S), gives the stronger H1, H2 and H4 evidence?

---

## 2. Data, universe, samples

**LOBSTER archive.**
- `lobster2:/lobster/YYYY/YYYYMMDD/TICKER.7z`, NASDAQ ITCH messages, 2012–2026.
- For NYSE-listed names this is NASDAQ's own book and trades.

**Prices.** CRSP `dsf_v2` (CIZ) through 2025-12-31:
- official open and close (`dlyopen`, `dlyclose`);
- `dlyret` (includes distributions and delisting);
- `dlyvol`, `dlyprcvol`, `shrout`, `dlycumfacpr`.

**Point-in-time universe for year Y.**
- CRSP common stocks: `sharetype NS`, `securitytype EQTY`, `securitysubtype COM`, `usincflg Y`, `issuertype ∈ {ACOR, CORP}`, `primaryexch ∈ {N, Q, A}`.
- Top **1,000** by average daily dollar volume over Oct–Dec of Y−1, with ≥ 40 valid days. This is the program-evidence-v1 rule with a larger N.
- Every trading day of Y is requested under the ticker CRSP assigns that day.
- If two PERMNOs share a ticker on a date, keep the one with higher Oct–Dec dollar volume. The archive has no dotted class tickers.
- Names missing from the archive on a date are recorded as `missing`, never as zero flow.
- Coverage by year, exchange and prior-year return is reported (the known coverage bias, §1.29).

**Name split.** Keyed on PERMNO, so it is stable across ticker changes and years:

    group = int(sha256("p4-revisit-v1|" + str(permno)).hexdigest()[:8], 16) % 3

**Samples.**

| cell | years | names | use |
|---|---|---|---|
| **DEV** | 2017–2019 | group 0 | implementation checks; distributions; Phase II model fitting; any descriptive look. Choices made here must be logged before VAL is read |
| **VAL** | 2020–2021 | groups 1–2 | read once: gates, Holm family selection, family comparison |
| **TEST** | 2022–2025 | groups 1–2 | read once, after VAL is written up; per-year breakdown |
| **ERA2** | 2012–2016 | all groups | second-era replication, read once after TEST |

Other cells (2017–2019 groups 1–2; 2020–2025 group 0) are not extracted in v1.

---

## 3. Events

Both families use regular hours [9:30, 16:00). No book or price information enters burst formation.

### 3.1 Trade bursts (T)

- **Packets.** Economic packets from `execution_packets.reconstruct_packets` (timestamp-collapsed executions).
  - Native visible sign.
  - Hidden rows inherit a unique same-timestamp visible sign, else the outside-prequote rule, else unsigned.
  - Type-5 Direction is never used.
- **Rule.** Validated **run60** (`program_bursts.run_bursts(day, gap=60)`):
  - maximal same-sign packet sequences;
  - no opposite-signed or unsigned packet in between;
  - consecutive gaps < 60 s;
  - ≥ 3 packets.
- Q_b = side × Σ packet shares.

### 3.2 Submission bursts (S)

The P4 §2 object: short same-side sequences of order submissions at similar price levels.

- **Qualifying add.** A type-1 message that is:
  - (i) priced at or better than the same-side best quote prevailing strictly before it (inside the spread or at the touch);
  - (ii) not the add half of an ITCH replace (same timestamp and side as a type-3 delete);
  - (iii) not the posted remainder of a marketable order (shares a timestamp with an execution);
  - (iv) not fleeting: not fully deleted within 1.0 s of submission without any execution against it.
- **Rule.** The run rule on qualifying adds, mirroring run60:
  - maximal same-side sequences;
  - ended by an opposite-side qualifying add;
  - consecutive gaps < 60 s;
  - ≥ 3 adds.
- Side +1 = bid adds (buying interest).
- Q_b = side × Σ added shares.
- Recorded, not used for formation:
  - modal add size and its share;
  - share of added volume executed by T_dec;
  - share of adds later cancelled.

### 3.3 Pseudo-bursts

For every real burst, a pseudo-burst on the same name-day:
- same side, Q_b and duration;
- start drawn uniformly from [9:30, 16:00 − duration − 600 s];
- seeded by `(ticker, date, family, burst index)`.

A pseudo-burst has no events of its own. It is measured on the real mid path exactly as a real burst would be.

---

## 4. Measurement (per burst b; side s; start t_b; end t_e)

- **Mid.** m(t) is the NASDAQ mid (bid+ask)/2 prevailing strictly before t, from book reconstruction.
- **Reference.** m_ref = m(t_b).
- **PeakImpact** (eq. 3.1, signed variant). max over τ ∈ [0, t_e − t_b + 10 s] of s·(m(t_b+τ) − m_ref), floored at one tick ($0.01).
- **D_b** (eq. 3.2, verbatim, from initiation). Mean over h ∈ {60, 180, 300, 600} s of s·(m(t_b + h) − m_ref).
- **Decision time.** T_dec = max(t_b + 600 s, t_e + 10 s). Nothing about b may be used before T_dec. Bursts with T_dec ≥ 16:00 are excluded.
- **Informative** (eq. 3.3). D_b ≥ κ·PeakImpact.
  - Primary κ = **0.5**.
  - κ ∈ {0.25, 0.75} reported as sensitivity only, never selected on.
- **Large.** |Q_b| ≥ 80th percentile of |Q_b| for the same name and family over the previous 20 available trading days (≥ 5 prior days required). This is the P4 bracket's order-splitting size threshold.
- **Post-decision displacement** (H1 outcome, bps):
  - d_close = s·(m_close − m(T_dec))/m(T_dec), where m_close is the last mid before 16:00:00;
  - d_open = s·(O_t+1 − m(T_dec))/m(T_dec);
  - d_cc = s·(C_t+1 − m(T_dec))/m(T_dec).

  Next-day CRSP prices are put on day t's split basis with `dlycumfacpr`.
- **Permanence ratio** (eq. 4.1, Phase I descriptive only). φ(b; x) = s·(x − m_ref)/PeakImpact for x ∈ {m_close, O_t+1, C_t+1}, winsorized at 1/99% per year.

---

## 5. Daily signals and targets (Phase III)

For name i, day t and family F, all flows are divided by ADV20 (CRSP mean share volume over t−20..t−1):

| signal | bursts included |
|---|---|
| S_info | informative ∧ large |
| S_large | large |
| S_all | all |
| S_pseudo | large real bursts whose pseudo-burst is informative |
| S_pred | large bursts with Phase II predicted persistence > θ. θ is the DEV quantile giving the same selection rate as the realized filter |

**Decision clocks and targets.**

| target | bursts used | return |
|---|---|---|
| **CLOP** | T_dec ≤ 15:50:00 | C_t → O_t+1 (CRSP) |
| **CLCL** | T_dec ≤ 15:50:00 | C_t → C_t+1 (`dlyret`) |
| **tCLOSE** | T_dec ≤ 15:30:00 | m(15:30) → m_close (NASDAQ mids) |

P4's tCLOSE, measured from the burst time, overlaps D_b's own window. It appears only in the leakage demonstration (§7.3).

**Controls** (all known at the decision clock):
- total signed packet flow to the clock ÷ ADV20;
- own return O_t → m(clock);
- r_t−1 and r_t−5..t−1;
- log market cap (t−1);
- log dollar ADV20;
- 20-day return volatility;
- quoted half-spread at the clock;
- 20-day turnover.

---

## 6. Tests, statistics, gates

**Inference defaults.**
- Name-day means, then daily cross-name means, with Newey–West (10) on the daily series.
- Fama–MacBeth: daily cross-sectional OLS with regressors winsorized 1/99% per day; NW(10) on the slopes.
- Contrasts of per-burst means use a name bootstrap (1,000 draws) for CIs.
- Hurdles: HLZ t > 3 in VAL; same sign with t > 2 in TEST.

### Q0. Audit of the legacy pipeline

Sample: 2023, NVDA, TSLA, JPM, MS plus 36 random group-0 names × 20 random dates.

- **Q0a sign bias.** Python replica of `src_cpp/burst.cpp`:
  - message-level;
  - Hawkes β = 1, trigger 0.3;
  - direction by count ratio ≥ 0.763 and minority/majority volume ≤ 0.28;
  - volume ≥ 0.00197 × ADV14.

  Run with the legacy sign mapping, then with type 5 excluded, then on native packets. Report:
  - sell share of directional bursts;
  - share of net-short name-days;
  - 1-minute directional hit rate;
  - same-day correlation of signed burst volume with open-to-close return.
- **Q0b beta benchmark.** Long-only close-to-open Sharpe for NVDA and TSLA 2023–24 (CRSP), against the reported 1.58 and 1.57.
- **Q0c κ circularity.** On the same sample, the 3-minute markout with the legacy D_b gate against κ = 0.

**Output.** Audit table; no gate.

### Q1 (H1), per family

- **Primary statistic.** Δ_pseudo = daily mean of name-day [d_close(info ∧ large, real) − d_close(info ∧ large, pseudo)].
- **Secondary.**
  - Δ_non = d_close(info ∧ large) − d_close(¬info ∧ large);
  - the same for d_open and d_cc;
  - levels for each class.
- **Gate G1.** Δ_pseudo > 0 with t > 3 in VAL; same sign with t > 2 in TEST.
- **Phase I description.** φ and d_close by deciles of duration, |Q_b|, PeakImpact, D_b/PeakImpact and time of day. DEV and VAL tables, no test.

### Q2 (H2), per family

- **(a) Internal linkage.**
  - For each burst, a link is another same-name, same-day burst of the same family starting within ±30 min with the same modal non-round child size:
    - T: modal untruncated packet size, ≥ 2 children;
    - S: modal add size, ≥ 2 adds.
  - Same-side link rate minus opposite-side link rate = directional linkage.
  - Contrast info ∧ large vs ¬info ∧ large in a regression with name-day fixed effects and hour × size-quintile dummies.
  - **Pass:** coefficient > 0 with name-bootstrap 95% CI excluding 0 (VAL).
- **(b) 13F.**
  - ΔIO_i,q = change in total 13F-reported shares ÷ shares outstanding, from the SEC Form 13F data sets.
  - Pooled panel with quarter fixed effects and name-clustered SE. Regressors:
    - quarter sums of S_info and S_large − S_info (each ÷ ADV);
    - prior-quarter return;
    - log size;
    - turnover.
  - **Pass:** β_info − β_(large−info) > 0 with t > 2 (VAL).
- **(c) Mutual funds.** Same regression with the CRSP MF quarterly holdings change, and separately with flow-induced trading (Lou 2012). Pass rule as (b).
- **(d) Index events.** S&P 500 and Nasdaq-100 additions and deletions:
  - E−5..E−1 z-scored imbalance, additions minus deletions;
  - informative minus other large flow.
  - VAL and TEST pooled for power, reported by cell.
  - **Pass:** difference > 0 with t > 2.
- **(e) Retail.** Daily cross-sectional correlation of S_info with BJZZ retail imbalance, minus that of S_large − S_info.
  - **Fail** if > 0 with t > 2 (informative flow more retail-like).
- **Verdict.** "Persistent-impact bursts load on institutional parent flow" requires, in VAL, replicated in TEST:
  - (a) passes;
  - at least two of (b)–(d) pass;
  - (e) does not fail.

  Otherwise it is reported as not shown.

### Q3 (H3), per family

- **Target.** d_close (primary); d_open and d_cc secondary. Winsorized 1/99% per year.
- **Features at T_dec.**
  - log(|Q_b|/ADV20), event count, duration;
  - PeakImpact (bps), D_b (bps), D_b/PeakImpact, D profile slope (D600 − D60);
  - spread at t_b and T_dec;
  - time of day; own return O_t → m(T_dec); 30-min pre-burst return;
  - modal-size share, backward same-side link count (previous 30 min);
  - T only: untruncated share, program score (`program_model_run60.json`);
  - S only: executed share and cancel share by T_dec.
- **Models (fixed hyperparameters).**
  - Ridge (α = 1, standardized, winsorized);
  - `HistGradientBoostingRegressor` (max_depth 3, 300 iterations, learning rate 0.05, min_samples_leaf 200).
  - Fit on DEV (up to 3M bursts sampled uniformly by name-day).
- **Statistic.** Daily Spearman rank IC between prediction and target, NW(10).
- **Gate G3.** Mean IC > 0 with t > 3 in VAL (best of two models, Bonferroni ×2); same model t > 2 in TEST.

### Q4 (H4)

- **Primary cells:** 2 families × 3 targets = **6**. In each, the FM coefficient on S_info with S_large, S_all and all §5 controls in the regression.
- **Holm** at 5% across the 6 cells in VAL. A cell must also have t > 3.
- **TEST:** only VAL-passing cells, same sign with t > 2.
- **Secondary, reported without selection:**
  - S_pred in place of S_info;
  - S_pseudo in place of S_info;
  - the κ sensitivity.
- **Portfolios.**
  - Daily decile sort on S_info among names with nonzero S_info; long top decile, short bottom, equal-weighted, dollar-neutral.
  - Costs: 2 bps per side for auction trades (CLOP, CLCL); 1 bp sensitivity. tCLOSE entry pays the 15:30 half-spread, exit 2 bps.
  - Report Sharpe, NW t, and the deflated Sharpe probability given the ledger count (`DEFINITIONS_TRIED.md` plus this design's cells).

### Q5 (H5)

On TEST (and VAL), for Q1 Δ_pseudo and the Q4 primary coefficient, split by:
- listing exchange (N vs Q);
- NASDAQ share of consolidated volume (packet shares ÷ CRSP volume, terciles);
- size terciles;
- tick constraint (quoted spread ≤ 1.5 ticks at 15:50);
- year.

Descriptive; heterogeneity tests are Wald tests of equal coefficients.

### Family comparison

- **VAL scoreboard.**
  - Q1 Δ_pseudo t;
  - Q2(a) linkage coefficient t;
  - best Q4 primary t across the 3 targets.
- The family higher on at least 2 of 3 is "more promising".
- Both families go to TEST regardless.

---

## 7. Firewalls and placebos

1. **Time ordering.** Signals use only bursts with T_dec before the clock. Targets start at or after the clock.
2. **No selection on VAL or TEST.**
   - κ, the size threshold, the burst rules, the model hyperparameters and θ are fixed above or set on DEV.
   - The κ grid is reported, never chosen.
3. **Leakage demonstration.** It must show spurious significance, or the pipeline is not sensitive enough. Two versions:
   - (i) P4's original tCLOSE from P_tb regressed on D_b-filtered flow;
   - (ii) informative defined with D measured to the close, used for the tCLOSE target.
4. **Pseudo-bursts.** The momentum control for H1 and H4 (§3.3).
5. **Sample access.**
   - VAL outputs are produced by one aggregator run, logged with a checksum, before any TEST aggregation.
   - TEST likewise before ERA2.

---

## 8. Compute plan

- **Extractor.** `src_py/p4_extract.py`, run on a Hoffman2 compute node per name-day.
  - Inputs: the raw message file, previous state none.
  - One book reconstruction pass.
  - Output npz on the cluster (licensed derivative, never committed):
    - per-burst arrays for T and S (float32; §4 quantities, pseudo-burst quantities, features);
    - a 1-minute mid grid;
    - daily packet flow totals at 15:30 and 15:50;
    - a status JSON.
- **Execution.**
  - Bundled long shards on `bertozzi_pod.q` with `-l highp`.
  - One lobster2 ControlMaster per shard.
  - ≤ 72 concurrent archive streams in total.
  - Resumable by status file.
- **Speed.** If the pilot shows Python book reconstruction dominates, a C++ BBO helper replaces it only after exact equality with `burst_alt.reconstruct` on the pilot days.
- **Aggregation.** Trailing size thresholds, ADV scaling, signals, joins to CRSP and external data. Runs per cell, on the cluster or locally from derived non-price aggregates. Price-derived per-burst data stays on the cluster.
- **External data.** Local, gitignored:
  - WRDS: CRSP `dsf_v2` universe and returns; CRSP MF; TAQ BJZZ; index constituents.
  - SEC Form 13F data sets, 2016Q4–2025Q4. Public; the request User-Agent carries a project name only.

---

## 9. Multiple-testing accounting

This design adds:
- Q1: 2 primary statistics;
- Q2: 2 × 5;
- Q3: 2 × 2 models;
- Q4: 6 primary cells plus 3 secondary signal variants × 6;
- the κ grid, reported, not selected.

Every number is added to `DEFINITIONS_TRIED.md` when run. Deflated-Sharpe trial counts include the legacy ledger.

---

## Amendments

The freeze hash of this file before amendments is `3a33beb216b3` (sha256 prefix), 2026-09-15.

### A1 — 2026-09-15, before any P4 outcome was computed or viewed

Only counts, timings and file sizes had been looked at.

1. **13F source.**
   - Direct SEC download is refused without a contact email in the User-Agent (HTTP 403), and the user's email is not to be sent.
   - Q2(b) instead uses WRDS SEC Analytics `wrdssec.wrds_13f_holdings`: the same EDGAR 13F-HR filings, parsed by WRDS, through 2025Q3 (`src_py/p4_external.py f13`).
   - Per manager and quarter:
     - the latest original filing;
     - replaced by the latest restatement amendment;
     - plus "new holdings" amendments;
     - share positions only, no put/call rows.
   - TEST quarters for (b) are 2022Q1–2025Q3.
2. **Implementation paths, no definition change.**
   - Two speedups, each accepted only after equality with the canonical code:
     - `src_cpp/p4_bbo.cpp` replaces the Python book loop of `burst_alt.reconstruct`;
     - `src_py/p4_packets.py` replaces `execution_packets.reconstruct_packets`.
   - Equality was checked on synthetic tapes (`tests/test_p4_extract.py`) and on five real name-days: HOG 2018-06-01, KO 2017-06-01, MS 2023-06-01, JPM 2023-06-01, AAPL 2019-06-03. All 49 saved arrays were identical (`hoffman2/p4_verify.sh`, job 14752753).
   - Busy name-days drop from 70–160 s to about 3 s.
3. **Storage.**
   - Per-burst times and prices are saved as float32 (about 4 ms and 1e-7 relative).
   - t_dec and the pseudo t_dec are not saved; they are recomputed from t_b and t_e.
   - d180 and d300 are saved only through dmean, the P4 average of the four horizons.
4. **Universe mechanics.**
   - S&P 500 deletion events exclude CRSP's data-end closing of current memberships (2025-12-31).
   - The universe pull (`src_py/p4_universe.py`) requested 248,459 DEV, 323,918 VAL, 644,138 TEST and 1,219,067 ERA2 name-days before archive coverage.
5. **Q0 sample.**
   - 2023: NVDA, TSLA, JPM, MS plus 36 random group-0 names × 20 random dates (800 name-days), seed 20230601.
   - Q0 computes legacy-detector statistics only, no P4 outcome, so drawing four flagship names that may belong to TEST names does not read TEST outcomes.
6. **Pilot outputs.** Extractor pilot name-days from TEST years (`out/PILOT`) are used for timing only. Their outcome columns are not to be examined.

### A2 — 2026-09-15, before any P4 outcome was computed or viewed

1. **Renamed tickers.**
   - lobster2 files some historical name-days under a company's *later* ticker: `META.7z` on 2022-01-03, when CRSP's ticker was FB; `ELV.7z` in early 2023.
   - Among the first 1,133 missing DEV/VAL name-days, 753 had such a candidate.
   - After each cell's main pass, missing name-days are retried under the PERMNO's later CRSP tickers, most recent change first (`src_py/p4_alt_tickers.py`, `RETRY=1` in `hoffman2/p4_shard.sh`).
   - A ticker is excluded if CRSP assigns it to another security on that date. Coverage is reported before and after the retry.
2. **Q2(a) implementation.**
   - Links are computed only among bursts with a non-round modal child size (≥ 2 children of that size). A burst without one cannot link, so it is excluded, not counted as zero.
   - The hour × size-quintile control is implemented as fully interacted name-day × hour × size-quintile cells, with within-cell informative-minus-other differences weighted n_i·n_n/(n_i + n_n).
   - Size quintiles rank |Q_b| among the name-day's large bursts. This is stricter than additive dummies.
   - The pass rule applies to the directional contrast (same-side minus opposite-side).
3. **Q4 portfolio costs.**
   - For CLOP and CLCL, both legs pay entry and exit at the auctions: 4 × the per-side cost per day on a $1-long/$1-short book.
   - For tCLOSE, each leg pays its 15:30 quoted half-spread plus 2 bps.
   - Deflated Sharpe uses 311 trials (legacy subtotals 110 + 75 + 66, plus 60 for this design) and a per-period Sharpe variance of 1/T.
4. **Q2(b)–(c) mechanics.**
   - 13F ownership uses the 8-character CUSIP valid at each quarter end.
   - Mutual-fund holdings changes use portfolios reporting at both quarter ends.
   - FIT follows the program-evidence-v1 construction. Share counts go on a common basis with the CRSP price factor (`dlycumfacpr`; the share factor was not pulled).
   - All regressions winsorize 1/99% and include quarter fixed effects with PERMNO-clustered errors.
5. **Execution.**
   - The name-day threshold uses the exact 80th percentile of the pooled previous 20 available name-days.
   - Per-burst samples keep at most 10 bursts per name-day and family, chosen deterministically at random. This is the design's "sampled uniformly by name-day"; the DEV fit uses 1.30M trade and 1.38M submission bursts.
   - A pseudo-burst enters S_pseudo or the Q1 pseudo class only if its own decision time is before the clock.
   - Extraction shards were moved partly to the general queue for capacity. No definition changed.

### A3 — 2026-09-15, after the first DEV descriptive run and before any VAL, TEST or ERA2 output was computed

The first DEV run (`analysis/DEV_primary.json`, input `nameday_DEV` sha256 `49150bfcbebf`) showed impossible levels:
- mean post-decision displacement of −18 to −22 bps in every burst class;
- a +60 bps gross tCLOSE portfolio.

The cause is data, not economics.

1. **Early-close sessions.** On the eight DEV early closes (13:00) the NASDAQ book after the close holds stub quotes:
   - median 15:50 spread 300–900 bps;
   - close mid often half the CRSP close;
   - 33–50% of names more than 2% off.

   These dates are now excluded entirely: they are not signal days and not counted as available days for the size threshold. The list is the exchange early-close calendar 2012–2025 (`p4_aggregate.EARLY_CLOSE`). A data check confirms every date in DEV and VAL with a cross-name median 15:50 spread above 200 bps is on the list (DEV: seven of eight; the eighth, 2017-11-24, is listed but not flagged).
2. **Stub quotes on normal days.** A mid is used only when the quoted spread at that time is at most 500 bps; 0.04% of non-early-close name-days exceed it at 15:50.
   - A burst's decision mid must pass, or the burst is dropped from signals, Q1 classes, Q2(a) and the sample.
   - The close mid must pass, or every d_close on that name-day is missing. d_open and d_cc use CRSP prices and are kept.
   - Clock mids at 15:30 and 15:50 must pass, or they are missing for that name-day's controls and tCLOSE target.
3. **No other change.** Definitions, gates and statistics are unchanged. The earlier DEV output is kept as `agg/DEV_preA3` on the cluster.
4. **Renamed-ticker retry (A2.1) outcome.** It recovered no name-days in DEV (10,220 candidates) or VAL (14,106). Coverage is therefore unchanged by the retry:
   - DEV 55.8% of requested name-days (208 of 419 names);
   - VAL 59.7% (417 of 779).

### A4 — 2026-09-15, after the corrected DEV descriptive run and before any VAL, TEST or ERA2 output was computed

The A3-corrected DEV run (input `nameday_DEV` sha256 `80d871e5b693`; kept on the cluster as `agg/DEV_preA4`) showed the pseudo-burst placebo leaking into its own outcomes:
- the pseudo class had +2.50 bps post-decision displacement (trade bursts), against −1.44 for real informative bursts;
- the pseudo tCLOSE signal had t = 27.7.

Two mechanical links caused this.
1. **Real bursts decided after the clock.** S_pseudo summed Q_b over large real bursts whose pseudo-burst passed, without requiring the real burst itself to be known by the clock. A buy burst at 15:45 then "predicts" the 15:30 → close return through its own impact.
2. **Pseudo windows before their real burst.** A pseudo window that closes before its real burst has a displacement to the close that contains the real burst's impact, which is in the real burst's side direction by construction.

**Rule from this amendment.** A pseudo-burst enters S_pseudo, the Q1 pseudo class and the sample's pseudo flag only if:
- (i) its real burst is used at that clock (decided by the clock, valid decision mid);
- (ii) the pseudo window starts after the real burst ends;
- (iii) the pseudo decides by the clock.

Conditions (ii) and (iii) together imply the time part of (i).

No other statistic, gate or definition changes. The pre-A4 DEV numbers are kept for the record in `analysis/DEV_primary_preA4.json`.

### A5 — 2026-09-15, a coding error found *after* the first TEST read; TEST is re-read once

**What was wrong.** `crsp_frame` put next-day CRSP prices on day t's split basis with the cumulative price factor
inverted: `price_{t+1} × cfacpr_{t+1} / cfacpr_t` instead of `price_{t+1} × cfacpr_t / cfacpr_{t+1}`. CRSP's
`dlycumfacpr` is 10 before a 10:1 split and 1 after, so every split-day gap became roughly −99%. NVDA on
2024-06-07 read −0.9990 instead of −0.0043.

**What it touched.** Only quantities built from next-day prices:
- the CLOP target and its portfolios;
- the d_open and d_cc secondary contrasts in Q1 and Q3.

Unaffected: d_close (the Q1 primary and the Phase II target), CLCL (CRSP returns), tCLOSE (NASDAQ mids), every
Q2 test, Q2(a) linkage, and all signals.

**Scale.** 53 of 511,168 TEST name-days had |CLOP| > 0.5, enough to drive a decile portfolio mean of −334 bps a day.

**Disclosure.** The error was found while checking that impossible portfolio number, after TEST had been read
once. It is a data-processing bug, not a result-dependent choice. Pre-fix outputs are kept:
`analysis/TEST_primary_preA5.json`, `TEST_q3_preA5.json`, `TEST_q5_preA5.json`, `TEST_phase1_preA5.json`, and the
VAL equivalents. All cells are re-aggregated with the corrected line and the analyses re-run in the protocol
order VAL, then TEST, then ERA2. The Phase II models are not refit: they use only features and d_close, which the
bug never touched, so they remain the DEV-frozen models of `freeze_before_VAL.json`.

**Statistics that change** between the pre-A5 and corrected runs are listed in `studies/p4_revisit/P4_REVISIT_RESULTS.md`.

### A6 — 2026-09-15, 13F quarter completeness (filer counts only, no outcome inspected)

A quarter enters Q2(b) only when its cross-sectional median filer count is at least 100. Median filers per stock:
- 2012Q4: 8, 2013Q1: 17, 2013Q2 onward: about 200–310 (the XML-era structured filings begin 2013Q2);
- 2025Q3: 63, because WRDS has not yet ingested the full quarter.

So Q2(b) covers 2013Q2–2025Q2. Dropped quarters are listed in each output. The first TEST run, before this rule,
included the incomplete 2025Q3.
