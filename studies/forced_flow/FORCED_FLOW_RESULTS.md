# forced-flow-v1 — results

Design: `studies/forced_flow/FORCED_FLOW_DESIGN.md` (frozen 2026-09-23 after the DEV run, before TEST). Code: `src_py/forced_flow.py`.
Name fixed effects, standard errors clustered by date, trade bursts, P4 `d_close` (post-decision displacement
to the close, bps).

| coefficient | DEV 2017–19 (exploration) | TEST 2022–25 (confirmation) |
|---|---|---|
| **quarter-end × informative class** | **−1.77 (t −2.74)** | **−1.37 (t −2.09)** |
| quarter-end × all large bursts | −0.96 (t −1.94) | −0.51 (t −1.02) |
| quarter-end × non-informative class | +0.28 (t 0.52) | +1.01 (t 1.25) |
| quarterly expiry × informative class | +1.38 (t 1.14) | −1.02 (t −1.39) |
| quarter-end × |flow| / adv20 | −0.001 (t −2.08) | −0.001 (t −3.90) |

Normal-day means for reference: informative −1.46 (DEV), −1.09 (TEST).

**Gate (quarter-end, informative class, negative with t < −2 in TEST): passed.**

On quarter-end days — index rebalancing, benchmark tracking, window dressing — bursts that pass the P4
informativeness filter give back about 1.4 bps *more* of their impact than on ordinary days, roughly doubling
the normal give-back, while bursts that fail the filter are unaffected in both cells. The timing is fixed by
the calendar and is exogenous to any one stock's information.

**Reading.** The filter keeps bursts whose impact held for ten minutes. On days when a larger share of trading
is mechanical, what it keeps is more likely to be price pressure, and reverts more. This is direct evidence
that "informative" in the P4 sense is partly a label for predictable non-information flow, and it is consistent
with p4-revisit-v1's central result that the filter's bursts give impact back rather than hold it.

Quarterly expiry does not replicate (opposite signs across cells) and is not claimed.
