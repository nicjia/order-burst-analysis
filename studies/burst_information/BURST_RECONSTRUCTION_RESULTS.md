# Why a burst detector can miss an institutional execution program

Completed controlled diagnostic, 2026-09-13. These are synthetic results, not estimates of NASDAQ parent-reconstruction accuracy.

The fixed same-side detector suffers from both false merging and fragmentation. A clean-looking detected burst can be a small fragment of a much larger parent. Adding stricter filters can improve precision while reducing recovery.

| Controlled condition | Correct links / inferred links | Recovered links / true observed links | Parent packets captured / original parent packets |
|---|---:|---:|---:|
| Isolated parents | 100.0% | 60.4% | 99.1% |
| Background flow | 93.1% | 47.2% | 98.3% |
| Dense background | 15.6% | 13.9% | 89.5% |
| Overlapping parents | 41.5% | 33.4% | 73.9% |
| Random pauses | 100.0% | 25.8% | 93.5% |
| 35% venue observation | 100.0% | 24.5% | 25.0% |
| 20% ambiguous signs | 100.0% | 11.8% | 67.9% |
| Combined disturbances | 70.5% | 6.2% | 12.4% |

A link means a pair of executions assigned to the same detected burst. A correct link joins executions from the same simulated parent. Recovery counts pairs within the observed tape; the last column also penalizes unobserved executions. Means are across 30 independent simulated days with paired mechanisms. The detector uses same-side packets, gaps below one second, and at least three packets.

Even in the isolated-parent condition, the one-second boundary splits occasional long gaps and recovers only about 60% of same-parent pairs. This is detector fragmentation in a world where every execution really does come from a parent order. Random pauses reduce that to about 26%, while precision remains 100%. Dense unrelated flow instead makes most inferred links false.

The size-consistency sensitivity improves pair precision in dense background from 15.6% to 27.3%, but reduces pair recovery from 13.9% to 7.7%. Timing-only clustering in that condition recovers 88.0% of true observed pairs, but only 9.0% of its inferred links are correct. No detector is selected as a winner from these outcomes.

## What is and is not identifiable

A separate constructed counterexample assigns the exact same observed times, sizes, and signs either to split parent orders or to independent traders. Every observable detector output stays identical while parent labels change. This establishes that those observables alone do not uniquely determine parent identity without restrictions on the generating process. It does not establish that realistic splitting and realistic herding are always statistically indistinguishable, or that herding dominates real markets.

Theoretical reconstruction is possible as probabilistic inference under assumptions: estimate the chance of an active parent and remaining flow from a specified execution model. Exact parent labels need stronger information or validation. Unknown parent size, urgency, passive participation, overlapping programs, and omitted venues all broaden uncertainty.

The older project simulator labels random long gaps as liquidity-sensitive pauses, but its pause process is independent of its simulated order book. It therefore cannot test whether a book-adaptive institutional policy can be recovered. This controlled experiment likewise tests random pauses, not adaptive execution.

Evidence with participant identifiers supports a substantial role for splitting in short-horizon flow persistence: [Toth et al.](https://arxiv.org/abs/1108.1632). That does not imply that anonymous short bursts reveal individual parents. Nor are large price changes exclusively a signature of large institutional orders: liquidity gaps can amplify comparatively small orders, as studied by [Farmer et al.](https://arxiv.org/abs/cond-mat/0312703).

## Verification and scope

The independent pair-enumeration audit passed all 810 cells; maximum discrepancy was 1.1e-16. It shares tape generation and detector definitions but computes inferred and true pair sets separately. The summary aggregates were checked against every saved daily metric. Bootstrap intervals describe Monte Carlo uncertainty only.

Reproduction: `src_py/burst_recovery_diagnostic.py --days 30 --out results/burst_information_v1/recovery.json`, then `src_py/audit_burst_recovery.py --root results/burst_information_v1`. Use the project Python environment. Outputs: `recovery.csv`, `recovery.json`, `recovery_audit.json`, and `recovery_diagnostic.png` under that group.

The separate real-data information experiment is complete: 75 fits and 120 independently audited comparisons on 36 names x 40 dates, using panel 14732764 and repairs 14732766/14732767. Its primary flow contrast was negative; see `studies/burst_information/BURST_INFORMATION_RESULTS.md` for the full exploratory 2023/2024 matrix. No claim of profitability, institutional parent recovery, or publication novelty follows from this synthetic diagnostic.

## Additional completed diagnostic: join-training session bug

The legacy join model sorts fragments by intraday time across pooled simulated days. Parent
IDs restart each day. In a paired test with 30 new training days, 3,411 of 5,066 candidate
pairs crossed days (67.3%); 37 cross-day pairs even received positive same-parent labels.
The legacy implementation/config is preserved; `metaorder_join_v2.py` scopes candidates and
labels to simulation day, date, and instrument where those fields exist.

On 18 new simulated test days, using the same 2,880 valid within-day candidate pairs and
fixed probability threshold 0.5:

| Fit or rule | AUC | Join precision | Join recall |
|---|---:|---:|---:|
| Legacy pooled-day candidate construction | 0.904 | 97.1% | 31.0% |
| Session-scoped candidate construction | 0.966 | 92.6% | 91.6% |
| **Time gap alone** ("gap < 30s" for precision/recall) | **0.955** | **92.7%** | **88.4%** |
| Spread / depth / imbalance change alone | 0.498 / 0.499 / 0.490 | — | — |

**Correction, 2026-09-13: the corrected model is not evidence that reconstruction works.** It
is a time-proximity rule. The legacy simulator places ~45 parents in a 6.5-hour day (~190
fragments), so consecutive same-side fragments a few seconds apart are almost always one
parent: the median gap is 2.9 s for same-parent pairs and 317 s otherwise. Its book-state
features are drawn independently of parents, so they carry no information (AUC ~0.50). A
real name such as AMD produces thousands of qualifying same-side bursts per day, where a gap
rule has no such guarantee. The earlier statement that correcting the bug "raised recall from
31% to 92%" is arithmetically right and scientifically empty; it is listed as excluded evidence
in `VERIFIED_RESULTS.md` §2.

Chaining the corrected joins gives 80.3% fragment-pair precision and 82.6% recall on 2,973
candidate-eligible fragment nodes; the legacy fit gives 97.3% / 14.6%. Same caveat.

The CSV-based independent audit verifies day/parent labels, Brier errors, and confusion
counts. Files are `join_session_diagnostic.json`, its test/training CSVs, the paired model
parameters, and `join_session_audit.json` in the experiment results folder. Reproduce with
`diagnose_join_sessions.py` and `audit_join_sessions.py`.

What the bug fix does establish: the old joining failure cannot be used to argue that anonymous
reconstruction is exhausted. It does not invalidate the separately fitted fragment classifier,
the strict-flow continuation result, or the Hurst counterexample.

## Any-program participation label (added 2026-09-13)

`metaorder_participation.py` labels a fragment by the share of its volume from any simulated
parent, so mixtures of several parents count as program activity. On 30 legacy simulated days it
turns 447 of 4,927 fragments (9%) from negative to positive relative to the old single-parent
purity label. The change is small because simulated parents rarely overlap; the label is correct
plumbing, not evidence about real markets.

## What this means for real-data validation

Every accuracy number in this file comes from a simulator whose parents are sparse and whose
book ignores them. None of it transfers to NASDAQ. LOBSTER itself provides no aggressor identity:
the seventh message column is NASDAQ MPID attribution, but on the probed 2019–2025 stock-days it
appears almost only on non-marketable market-maker quotes (at most a few hundred executed
orders, essentially all UBSS); hidden executions carry no order id; and same-timestamp links
from a trader's resting order to an aggressive execution occur at 0.2–0.3% of execution timestamps.
Real-data validation of burst definitions therefore needs a signature of common origin whose
chance rate can be measured. `studies/fingerprint/BURST_FINGERPRINT_DESIGN.md` uses repeated identical child sizes
against a cross-day null.
