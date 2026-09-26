# Burst research — the full backlog (v2, 2026-09-25)

Status: ✅ done · 🔄 running · ⭐ top candidate · ⬜ not started · ✖ tried and dead. Every definition must
state a **decision time** — the first moment the event is known to exist *and* to have ended (or the moment of
decision if we act during it) — and nothing measured after it may be a feature. Every trading idea is first
judged as a **forecast at the mid** (supervisor's framing: forecasting first, spread crossing second).

Contents: 0 literature · 1 burst definitions (≈120) · 2 decision timing (≈20) · 3 features (≈90) · 4 targets
(≈30) · 5 trading ideas (≈110) · 6 models (≈25) · 7 validation and robustness (≈30) · 8 data sources and
ground truth (≈20) · 9 natural experiments (≈20) · 10 far-fetched (≈30) · 11 execution order.

---

## 0. Literature to build on (checked 2026-09-25)

| paper | what it does | use for us |
|---|---|---|
| ClusterLOB — Zhang, Cucuringu, Shestopaloff, Zohren, *Quantitative Finance* 2026 (arXiv 2504.20349) | 6 time/book features per MBO event (add, cancel, trade) → K-means++ → directional / opportunistic / market-making; OFI per cluster in 30-min buckets forecasts same bucket, next bucket and to the close (LOBSTER, 15 NASDAQ stocks, 2021). Opportunistic OFI → next bucket Sharpe 1.34 vs 0.60 plain OFI (small tick). | Definition B8.1, features F11, trading T3.6 |
| Order Flow Decomposition — Sitaru, Calinescu, Cucuringu, ICAIF 2023 | "Decomposed OFI": order flow split into components; contemporaneous and forward-looking impact | F10, T3.7 |
| Neutrinos of the order book; Price impact of nothing; An open book (Level-4 Hyperliquid) — Albers, Cucuringu, Howison, Shestopaloff, 2026 | public trader addresses; rejected post-only orders (67% of messages) predict returns | ground truth G1; B7.8 |
| Intraday lead-lag in idiosyncratic returns — Cartea, Cucuringu, Shi, 2026 | intraday lead-lag after factor removal | T1.9–T1.12 |
| Cross-impact of OFI — Cont, Cucuringu, Zhang, QF 2023 | multi-level OFI; cross-asset OFI forecasts | F4, T1.9 |
| Generating realistic metaorders from public data — Maitrier et al. 2025 | synthetic metaorders | G4 |
| Public trader identity (arXiv 2608.04373); Trading in the sunshine or the shade (arXiv 2606.15715) | Hyperliquid identity-based predictability; hidden vs visible TWAPs | G1, T3.11 |
| When large trades are not news (arXiv 2607.01198) | informativeness depends on the tail of liquidity demand | F9.6 |
| Gould & Bonart 2016, queue imbalance one-tick-ahead | queue imbalance predicts the next mid move | baseline for every short-horizon claim |
| Kolm, Turiel & Westray 2023, deep OFI | multi-level OFI + deep nets at seconds horizons | F4, M-section |

---

## 1. Burst definitions (≈120)

Common notation: packets = economic trade packets (one marketable order's fills at one timestamp), native
signs; G = gap threshold; m = minimum members (default 3). "Decision" = leak-free decision time.

### B1. Same-side runs (broken by any opposite or unsigned packet)
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B1.1 | G ∈ {0.1, 0.25, 0.5, 1, 2, 5, 60} s, m = 3 | t_e + G | child clusters | ✅ short G wins |
| B1.2 | G ∈ {10, 25, 50} ms | t_e + G | co-located algo cadence | ✅ 10–50 ms match or beat 0.1 s (IC 0.22 at 10 s) |
| B1.3 | G ∈ {10, 30, 120, 300} s | t_e + G | slower slicers | partly (60/300 s ✖ short horizons) |
| B1.4 | m ∈ {2, 5, 10, 20} at G = 0.1 / 0.5 s | t_e + G | size of the child cluster | ⬜ |
| B1.5 | volume floor: burst volume ≥ {0.01, 0.05, 0.1}% of ADV | t_e + G | economically large clusters only | ⬜ |
| B1.6 | tolerant runs: allow ≤ 1 / 2 opposite packets, or opposite volume ≤ 20% of run volume | t_e + G | robust to interleaved noise | ✅ tolerant runs ≈ runs (IC 0.21) |
| B1.7 | unsigned packets skipped instead of breaking the run | t_e + G | hidden prints inside a run | ⬜ |
| B1.8 | per-stock adaptive gap: G = c × the stock's median inter-packet time over the prior 20 days, c ∈ {1, 2, 5, 10} | t_e + G | comparable across liquidity | ✅ adaptive gap IC 0.217, gain t 20.9 |
| B1.9 | time-of-day adaptive gap: G = c × median IAT in that 30-min slot over the prior 20 days | t_e + G | intraday seasonality | ⬜ |
| B1.10 | volume clock: gap measured as shares traded by others between members | first other-trade crossing the gap | busy vs quiet periods | ⬜ |
| B1.11 | message clock: gap = number of intervening book messages | t_e + gap in messages | book-activity scaling | ⬜ |
| B1.12 | duration cap: split runs longer than {5, 30, 120} s | at split | long slicing split into children | ⬜ |
| B1.13 | first-k-children event: act after the k-th member, k ∈ {3, 5, 10} | time of the k-th member | real-time, before the end | ✅ **best rule**: act at 3rd / 5th child (IC 0.22–0.23) |

### B2. Side-only streams (each side's own sequence; the other side is ignored)
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B2.1 | G ∈ {0.5, 1, 2} s | t_e + G | algorithms slicing through two-sided noise | ✅ ≈ runs |
| B2.2 | G ∈ {10, 50, 100, 250} ms | t_e + G | fast streams | ⬜ |
| B2.3 | G ∈ {5, 30, 60} s | t_e + G | slow streams | ✅ v1 (5 s) |
| B2.4 | stream with dominance: keep only if same-side volume ≥ {60, 75, 90}% of all volume in the window | t_e + G | directional pressure | ✅ dominance streams ≈ runs |
| B2.5 | stream with duration cap | at split | long streams chunked | ⬜ |

### B3. Intensity and point-process bursts
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B3.1 | original Hawkes (β 1, trigger 0.3, direction 0.763 / 0.28, volume floor) | **t_e + ln(λ_end/0.3)/β** (fix) | the 279 detector | ✅ fixed; weaker than short runs (IC 0.167 at 10 s) |
| B3.2 | Hawkes grid β ∈ {0.5, 1, 2, 5}, trigger ∈ {0.1, 0.3, 0.5} | confirmation time | detector sensitivity | ⬜ |
| B3.3 | side-specific Hawkes (one intensity per side) | confirmation | directional clusters directly | ✅ side-specific Hawkes IC 0.169 |
| B3.4 | size-marked Hawkes (jump ∝ volume) | confirmation | large-child clusters | ⬜ |
| B3.5 | rate bursts: trades in the last 1 s / 5 s > k × the stock's rate at that time of day (k 3, 5, 10) | when the rate falls back below k | activity spikes | ⬜ |
| B3.6 | signed-volume bursts: net signed volume over 5 s / 30 s > k × typical | when it falls back | imbalance episodes | ⬜ ⭐ |
| B3.7 | Poisson-surprise: count in a window above the 99th percentile given the baseline | window end | statistically unusual clustering | ⬜ |
| B3.8 | Bayesian online change-point on arrival rate / signed flow | change-point confirmation | regime starts | ⬜ |
| B3.9 | CUSUM on signed flow | alarm time | sequential detection | ⬜ |

### B4. Size and fingerprint based
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B4.1 | clip chains: same side + same non-round untruncated size within G ∈ {5, 30} s | t_e + G | one parent's identical clips | ✅ weaker than short runs |
| B4.2 | clip chains G ∈ {1 s, 5 min, 30 min} | t_e + G | fast / slow clipping | ⬜ |
| B4.3 | near-identical clips (sizes within ±2% / ±5%) | t_e + G | randomised clip sizes | ⬜ |
| B4.4 | dollar-clip chains (notional within ±1%) | t_e + G | dollar-sized parents | ⬜ |
| B4.5 | round-lot chains (100 / 200 / 500) — as a convention control | t_e + G | convention, not parents | ⬜ |
| B4.6 | odd-lot runs (all members < 100 shares) | t_e + G | small-clip algos / retail | ⬜ |
| B4.7 | block runs (members ≥ 1,000 shares or ≥ $100k) | t_e + G | institutional-size children | ⬜ |
| B4.8 | fingerprint-filtered runs: short-gap runs whose modal-clip share ≥ 0.5 | t_e + G | runs that look like one parent | ⬜ ⭐ |
| B4.9 | decaying / geometric sizes (each child a fixed fraction of the last) | t_e + G | size-scheduling algos | ⬜ |
| B4.10 | multi-day clip episodes (same side + clip on consecutive days) | next open | multi-day programs | ✅ (fingerprint-multiday) |

### B5. Timing-regularity based
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B5.1 | timer chains: same side, ≥ 4 members, inter-arrival CV < {0.1, 0.2, 0.3} | t_e + 1.5 × median IAT | clock-driven schedulers | ✅ ✖ timer chains forecast worst (gain ≈ 0) |
| B5.2 | phase-locked: sub-second phase concentration R > 0.8 over ≥ 4 members spanning > 3 s | t_e + 1 s | whole-second timers | ✅ ✖ phase-locked gain ≈ 0 |
| B5.3 | periodogram segments: dominant period at 1 / 5 / 10 / 30 / 60 s in same-side arrivals | segment end | TWAP / timer frequencies | ⬜ |
| B5.4 | TWAP-like: equal clips at equal intervals (size CV < 0.1 and IAT CV < 0.2) | t_e + interval | TWAP children | ✅ ✖ TWAP-like gain ≈ 0 |
| B5.5 | VWAP-like: child size proportional to recent market volume | t_e + G | VWAP children | ⬜ |
| B5.6 | POV-like: same-side volume a stable fraction of market volume over 5-min windows | window end | participation algos | ⬜ |

### B6. Price-path based
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B6.1 | sweeps: packets executing at ≥ 2 price levels | t_e | urgency | ✅ sweeps IC 0.18, gain t 14.6 |
| B6.2 | level-clearing runs: the run exhausts the touch queue at least once | t_e + G | depth-consuming pressure | ✅ level-clearing IC 0.218, gain t 16.9 |
| B6.3 | price-walking runs: each member at a price ≥ the previous one's (buys) | t_e + G | momentum ignition | ⬜ |
| B6.4 | absorbed runs: same-side run with no mid change (liquidity absorbed it) | t_e + G | hidden/replenished liquidity | ✅ absorbed runs IC 0.199 |
| B6.5 | impact runs: mid moved ≥ 1 tick during the run | t_e + G | impact-making clusters | ⬜ |
| B6.6 | inside-spread prints clusters (midpoint and hidden executions) | t_e + G | dark-ish liquidity use | ⬜ |

### B7. Passive, submission and cancellation bursts (message-level)
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B7.1 | submission runs: adds at / inside the touch, same side, G ∈ {0.1, 0.5, 1} s | t_e + G | passive parents | P4 ✅ (10-min decision) · real-time ⬜ ⭐ |
| B7.2 | cancellation bursts: cancels on one side, G ∈ {0.1, 0.5} s, ≥ 5 cancels | t_e + G | liquidity withdrawal (pulled asks = bullish) | ✅ cancellation bursts IC 0.192, gain t 14.2; directional continuation AUC 0.68 |
| B7.3 | replace chains (cancel + add at the same timestamp on one side) | t_e + G | quote-chasing algos | ⬜ |
| B7.4 | quote-stuffing: messages per 100 ms > k × normal, no trades | end of spike | HFT activity spikes | ⬜ |
| B7.5 | layering: adds at ≥ 3 levels one side, then cancels within 2 s | cancel time | spoofing-like patterns | ⬜ |
| B7.6 | queue-joining: many small adds at the same price level within 1 s | t_e + G | herding at a level | ⬜ |
| B7.7 | fleeting-order bursts: adds cancelled within 1 s, clustered | t_e + G | NASDAQ analogue of rejected orders | ⬜ ⭐ |
| B7.8 | touch-improvement runs: successive adds that improve the best price | t_e + G | aggressive passive pressure | ⬜ |
| B7.9 | depth-building: net adds (adds − cancels) at the touch > k × typical | window end | liquidity supply | ⬜ |

### B8. Mixed-event and participant-type bursts
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B8.1 | ClusterLOB: per-event 6 features → K-means++ (fit on training months) → bursts of "directional"-cluster events | t_e + G | directional participants | ✅ clusters distinct; no forecasting gain over plain OFI |
| B8.2 | take-and-make: same side aggressive trades and passive adds within G | t_e + G | parents using both order types | ⬜ |
| B8.3 | two-sided (market-making) bursts: alternating sides, |net| < 20% | t_e + G | neutral control group | ⬜ |
| B8.4 | imbalance windows: |net signed volume| / total > 0.8 over ≥ 10 packets | window end | one-sided episodes | ⬜ |
| B8.5 | trade + opposite-cancel bursts: buys accompanied by ask cancels | t_e + G | informed pressure (makers pulling) | ⬜ ⭐ |

### B9. Hidden-liquidity based
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B9.1 | hidden-execution runs (type 5 only, signed where defensible) | t_e + G | dark liquidity use | ✅ hidden-heavy runs: best 30-min gain (+0.011, t 2.5) |
| B9.2 | hidden-heavy runs (hidden share > 50%) | t_e + G | mixed | ⬜ |
| B9.3 | midpoint-hidden clusters | t_e + G | midpoint pegs | ⬜ |
| B9.4 | iceberg refills: same size re-displayed at the same price after execution | refill time | icebergs | ⬜ ⭐ |
| B9.5 | reserve depletion: repeated executions at one price without the displayed size changing | t_e + G | hidden reserve | ⬜ |

### B10. Multi-scale and hierarchical
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B10.1 | two-level: 100-ms runs as children, chained into parents with 30 s / 5 min same-side gaps | at each child / parent end | child vs parent | ⬜ ⭐ |
| B10.2 | merged runs (5 / 30 min) | t_e + 60 s | parents | ✅ v1 (10-min decision) · real-time ⬜ |
| B10.3 | wavelet multiscale bursts on the signed-flow series | scale-dependent | multiscale structure | ⬜ |
| B10.4 | day-level programs (repeated clip, same side, many hours) | end of day | daily parents | ✅ |

### B11. Cross-sectional events
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B11.1 | basket bursts: same-side bursts in ≥ k ∈ {5, 10, 20} stocks within 100 ms | the k-th burst | program / basket trading | ⬜ ⭐ |
| B11.2 | sector bursts: ≥ 3 same-sector names within 1 s | 3rd burst | sector rotation | ⬜ |
| B11.3 | ETF-triggered: constituent bursts within 1 s of a SPY / QQQ / sector-ETF burst | constituent burst end | arbitrage flow | ✅ see T1.10 |
| B11.4 | index-weight-proportional bursts across constituents | basket end | index rebalancing | ⬜ |

### B12. Session-specific
| id | rule / grid | decision | captures | status |
|---|---|---|---|---|
| B12.1 | opening bursts (first 5 min) | t_e + G | open-auction spillover | ⬜ |
| B12.2 | closing bursts (after 15:50, MOC imbalance period) | t_e + G | close positioning | ⬜ |
| B12.3 | post-halt / LULD resumption bursts | t_e + G | price discovery restarts | ⬜ |
| B12.4 | news-minute bursts (first 60 s after an 8-K / earnings timestamp) | t_e + G | information arrival | ⬜ |

---

## 2. Decision timing (≈20)
| id | idea | status |
|---|---|---|
| D1 | 10 min after the start (P4) | ✅ discards short-horizon information |
| D2 | when the burst is confirmed over (t_e + G) | ✅ strong short-horizon forecasts |
| D3 | after the k-th child (k 3, 5, 10) — act *during* the burst | ✅ best short-horizon rule |
| D4 | at the first child (pure real-time; nothing but the first packet and context) | ⬜ |
| D5 | fixed latency grid after confirmation: +0, +100 ms, +1 s, +10 s (how fast the signal decays) | ✅ IC 0.21 → 0.15 (1 s late) → 0.06 (10 s late) |
| D6 | event-time decisions: after the next N trades | ⬜ |
| D7 | volume-time decisions: after X% of ADV more trades | ⬜ |
| D8 | decision at the next 1 / 5 / 30-min bucket boundary (aggregated signals) | ✅ 30-min · ✅ 5-min panel (v4): bursts add ≤ 0.002 IC |
| D9 | end-of-day decision (15:50) for overnight holds | ✅ null |
| D10 | Hawkes confirmation time (fix of the v2 leak) | ✅ confirmed-decision Hawkes |
| D11 | exchange-timestamp vs receive-timestamp latency sensitivity (add 1–5 ms) | ⬜ |
| D12 | only decide when the spread is at its minimum tick (execution-quality gate) | ⬜ |

## 3. Features (≈90)

**F1 controls (non-burst)**: move since the open ✅, 30-min pre-move ✅, 60-s pre-move ✅, time of day ✅, spread ✅,
daily volatility, realized volatility last 5 / 30 min, volume so far vs normal, market move so far, day-of-week,
days to earnings, overnight gap.
**F2 burst price path**: move during the burst ✅, peak impact ✅, move per share, impact per child, time to peak,
fraction of the move in the first child, retracement before decision, max adverse excursion.
**F3 burst structure**: children ✅, duration ✅, volume / ADV ✅, volume / visible depth, average child size, child
size trend (growing / shrinking), inter-arrival trend (accelerating / decelerating), share of children that
walked the book, number of price levels touched.
**F4 order book**: queue imbalance at start / end ✅, quote OFI before / during ✅, trade-flow imbalance ✅, multi-level
depth imbalance (levels 1–5) ⭐, depth-weighted mid vs mid (microprice gap) ⭐, opposite-touch depletion fraction ⭐,
time to refill the touch after the burst, spread change during the burst, number of levels cleared, book slope,
cancellation rate on each side during the burst ⭐, add rate same side during the burst, queue position proxy.
**F5 size regularity**: modal-clip share ✅, non-round clip ✅, size CV ✅, size / touch depth ✅, fraction of truncated
children, dollar-round clips, clip size relative to the stock's typical clip. **Forecasting ablation (2026-09-25): the
F5 + F6 group adds 0.000 IC at every horizon, 10 s to the close, 2024 and 2025.**
**F6 timing regularity**: inter-arrival CV ✅, median IAT ✅, sub-second phase concentration ✅, dominant periodogram
frequency, IAT autocorrelation, fraction of children at whole-second boundaries. (Adds nothing to forecasting — see F5.)
**F7 history**: bursts on the same / opposite side in the last 1 / 5 / 30 min, same clip seen earlier today,
yesterday's programs ✅ (daily), how many bursts today relative to normal, last burst's outcome (did it revert?).
**F8 hidden liquidity**: hidden share of the burst's volume, midpoint executions, iceberg refills at the touch,
reserve-depletion events.
**F9 stock characteristics**: tick-constrained flag, price level, market cap, volatility regime, ADV, NASDAQ market
share, short interest, options activity, index membership, flow tail-heaviness (F9.6).
**F10 decomposed OFI** (Sitaru et al.): OFI by event type (visible trades, hidden trades, adds, cancels, replaces).
**F11 ClusterLOB features**: the six per-event features and cluster shares of the burst's children.
**F12 cross-sectional**: peer burst flow ✅ (1 min), sector burst flow, ETF burst flow, basket-burst indicator,
correlation-cluster flow, lead-lag network centrality.
**F13 market state**: SPY / QQQ bursts in the last 1 / 10 s, VIX level, market-wide burst intensity, macro
release minute.

## 4. Targets (≈30)
Signed mid move to +10 s ✅, +60 s ✅, +5 min ✅, +30 min ✅, close ✅; +1 / 2 / 5 / 30 s, +2 / 10 / 60 min, next open,
next close ✅ v4 (IC 0.18 at +1 s → 0.20 peak at +5–10 s → 0.04 close → 0.03 next open → 0.01 next close); +100 ms; market-excess ✅, sector-excess, beta-adjusted; one-tick-ahead direction (up / down / none) ⭐;
first-passage: time until the mid moves one tick and which way ⭐; max favourable / adverse excursion within 5 min;
realized volatility next 1 / 5 / 30 min ✅ v4 (pre-event RV alone IC 0.65–0.88; bursts add ≤ 0.008) — |move| ✅ (IC 0.30–0.43, bursts add +0.01–0.02); spread and depth next minute (liquidity targets); does another same-side
burst follow within 60 s ⭐ (continuation of the parent); remaining duration / volume of the parent ⭐; time to
the next burst in the stock (hazard); permanence ratio from start ✅ (overlap trap); P&L of a passive quote posted
at decision (fill-aware target).

## 5. Trading ideas (≈110)

Each is judged first as a forecast at the mid (IC, decile spread, hit rate), then with costs. Format: signal →
entry → exit; horizon; success criterion. Universe: the test stocks unless stated.

### T1. Seconds to one minute (microstructure)
| id | idea | status |
|---|---|---|
| T1.1 | Fade the burst: model predicts the burst's impact reverts in 10–60 s → trade against the burst at the mid, exit at +10 / +60 s | ✅ mid P&L positive (run 0.1 s +2.3 bps at 10 s, t 26); loses after the spread |
| T1.2 | Ride the burst: predicted continuation (parent still active) → trade with it, exit at +60 s | ✅ mostly 'ride' trades at 10–60 s: +3.7 bps at 60 s, hit 60% |
| T1.3 | Decide after the 3rd child: forecast whether the burst continues, trade with it for the rest | ✅ act at 5th child, 60 s: +4.95 bps at the mid, hit 62%, t 27 |
| T1.4 | One-tick-ahead classifier combining queue imbalance with burst state; trade only when P(next move) > 0.6 | ⬜ ⭐ |
| T1.5 | Queue-depletion trade: burst consumed > 80% of the opposite touch → continuation for the next tick | ⬜ |
| T1.6 | Opposite-cancellation signal: buys + ask cancels → go long for 30 s | ⬜ |
| T1.7 | Absorbed burst: run with no mid change → the absorber is informed; trade against the burst | ⬜ ⭐ |
| T1.8 | Sweep follow: multi-level sweep → continuation 10 s | ⬜ |
| T1.9 | Peer flow → next minute (cross-impact, t 16 already) as a trade | ⬜ ⭐ |
| T1.10 | ETF → constituent: SPY / QQQ burst → trade lagging high-beta constituents for 1–10 s | ✅ ETF-specific flow **reverses** in constituents next minute (SPY t −8.6 / −2.8; QQQ t −10.1 / −3.3) |
| T1.11 | Constituent basket → ETF: basket burst in heavy weights → trade the ETF | ⬜ |
| T1.12 | Sector lead-lag: burst in the sector leader → trade laggards | ⬜ |
| T1.13 | Fleeting-order burst → short-horizon direction (analogue of rejected-orders paper) | ⬜ |
| T1.14 | Iceberg detected on the ask → short-horizon ceiling: fade approaches to that price | ⬜ |
| T1.15 | Timer-algo detected → predict the next child time and side; trade just before it | ⬜ far-fetched |
| T1.16 | Microprice gap after the burst → mean reversion to microprice | ⬜ |
| T1.17 | Burst in a tick-constrained stock → queue-position trade (join the side about to be hit) | ⬜ |
| T1.18 | Spread-widening after a burst → provide liquidity when the spread is wide (expected revert) | ⬜ |
| T1.19 | Latency test: how much of T1.1 survives 1 ms / 10 ms / 100 ms of delay | ✅ 60 s trade: +4.2 → +3.4 (1 s late) → +1.8 bps (10 s late) |
| T1.20 | Pairs: burst in one name of a correlated pair → trade the spread for 1 min | ⬜ |

### T2. Five to thirty minutes
| id | idea | status |
|---|---|---|
| T2.1 | Fade bursts predicted to revert over 5 min (IC 0.08 at 5 min) | ✅ +3.8 bps at 5 min (t 16) at the mid |
| T2.2 | Follow early-stage parents (two-level definition B10.1): child cluster likely followed by more same-side clusters | ⬜ ⭐ |
| T2.3 | Burst-conditioned intraday reversal: the since-open reversal is ~2× stronger at burst times → trade reversal only at bursts | ✅ **+6.8 bps (t 4.5) at bursts vs +2.9 (t 1.4) at random times** |
| T2.4 | Clip-chain continuation: identical clips keep coming → follow for 30 min | ⬜ |
| T2.5 | TWAP detection: detected TWAP → predict its remaining schedule → front-run lightly / provide liquidity | ⬜ |
| T2.6 | Basket-program detection: the remaining names of a basket will be hit → trade them | ⬜ ⭐ |
| T2.7 | Hidden vs visible: visible aggressive bursts vs hidden-heavy bursts — which one's impact persists 30 min? | ⬜ ⭐ |
| T2.8 | Two-sided (market-making) bursts as a volatility signal: trade straddle-like mean reversion | ⬜ |
| T2.9 | Imbalance windows (B8.4) → 5–30 min continuation | ⬜ |
| T2.10 | 1 / 5-minute bucket portfolios of model forecasts, cross-sectional long–short | ⬜ ⭐ |
| T2.11 | ClusterLOB replication: directional / opportunistic cluster OFI, 30-min buckets, next bucket | ✅ ✖ not replicated here: cluster OFI ≤ plain OFI (plain next-bucket Sharpe 2.59; opportunistic −0.08 t) |
| T2.12 | Burst-intensity regime: only trade T2.1 when the stock's burst intensity is in its top tercile | ⬜ |
| T2.13 | Cancellation-burst reversal: after mass ask cancels, fade any upward move over 5 min | ⬜ |
| T2.14 | Momentum ignition fade: price-walking runs followed by no follow-through → fade | ⬜ |
| T2.15 | Round-number interaction: bursts that push through a round price → continuation or rejection | ⬜ |

### T3. To the close (the supervisor's "burst time to close")
| id | idea | status |
|---|---|---|
| T3.1 | Per burst, decision → close at the mid, market-excess | ✅ ≤ 5 bps spread, to close, net +3 to +7 bps (level-clearing t 3.1) — **2025 confirmation running** |
| T3.2 | Reversal-to-close only at burst times (T2.3 at the close horizon) | ✅ same as T2.3 at the close |
| T3.3 | 15:30 → close cross-sectional portfolio from the day's burst forecasts | ✅ null (daily) · ⬜ with real-time model |
| T3.4 | Close-auction: bursts after 15:50 predict the closing price vs the last mid | ⬜ ⭐ |
| T3.5 | Open → close: first-hour bursts predict the rest of the day | ⬜ |
| T3.6 | ClusterLOB FREB (return to end of day) from cluster OFI | ⬜ |
| T3.7 | Decomposed OFI (Sitaru) to the close | ⬜ |
| T3.8 | Multi-day program day 2: enter at the open of day 2 in the program's direction, exit at close | ⬜ |
| T3.9 | Earnings-day bursts → close | ⬜ |
| T3.10 | Index-rebalance-day bursts → close auction imbalance | ⬜ |
| T3.11 | Hidden-metaorder cost: stocks with heavy hidden bursts vs visible → close return | ⬜ |
| T3.12 | Quarter-end give-back: fade informative-looking bursts on quarter-end days | ✅ give-back confirmed · ⬜ as a trade |

### T4. Overnight and multi-day
| id | idea | status |
|---|---|---|
| T4.1 | Close → next open from the day's bursts | ✅ ✖ (v4 re-test with real-time bursts: gain −0.010 / −0.002) |
| T4.2 | Close → next close | ✅ ✖ (v4 re-test: every burst IC within ±0.02) |
| T4.3 | Multi-day programs: after day 1 of a same-side clip program, position for day 2 | ⬜ |
| T4.4 | Weekly: stocks with persistent one-sided programs across the week | ⬜ |
| T4.5 | Pre-earnings burst flow (5 days) → announcement return | ✅ ✖ |
| T4.6 | Post-earnings drift conditioned on burst flow in the first hour | ⬜ |
| T4.7 | Fingerprint persistence as a liquidity-provision opportunity next day (predictable flow) | ⬜ |

### T5. Execution (buy-side use of the forecasts)
| id | idea | status |
|---|---|---|
| T5.1 | Delay our own buy children when a same-side burst is detected (avoid trading behind it) | ⬜ ⭐ |
| T5.2 | Accelerate our buys into opposite-side bursts (trade against their liquidity demand) | ⬜ ⭐ |
| T5.3 | Passive vs aggressive switch: post passively when the model predicts the mid will come to us | ⬜ |
| T5.4 | Schedule optimisation: TWAP vs burst-aware schedule, simulated implementation shortfall | ⬜ |
| T5.5 | Avoid detectable fingerprints: randomise our own clip sizes / timing (defensive) | ⬜ |

### T6. Market making
| id | idea | status |
|---|---|---|
| T6.1 | Skew quotes away from the side a predicted-to-continue burst is hitting | ⬜ ⭐ |
| T6.2 | Widen after sweeps, tighten after absorbed bursts | ⬜ |
| T6.3 | Queue-aware posting only when the model predicts reversion (fills that are not adversely selected) | ⬜ ⭐ |
| T6.4 | Inventory control using burst continuation forecasts | ⬜ |

### T7. Portfolio and meta
| id | idea | status |
|---|---|---|
| T7.1 | Combine horizons: 10 s, 5 min and close forecasts in one allocation | ⬜ |
| T7.2 | Regime filter: trade only when market-wide burst intensity is high / low | ⬜ |
| T7.3 | Capacity: signal strength vs position size (impact of our own trading) | ⬜ |
| T7.4 | Decay curve of every signal with latency (how fast it must be acted on) | ⬜ ⭐ |
| T7.5 | Correlation of burst strategies with standard factors (short-term reversal, momentum) | ⬜ ⭐ |

## 6. Models (≈25)
Gradient boosting (fixed) ✅; ridge / logistic ✅; per-stock vs pooled; stock embeddings; monotone-constrained
boosting; quantile regression; sequence models on the child sequence (1-D CNN, GRU, transformer); DeepLOB-style
CNN on book snapshots around the burst; marked point-process (Hawkes) continuation models; survival models for
burst end; online / daily-refit models; conformal intervals; causal forests (which bursts differ); mixture of
experts by liquidity bucket; ClusterLOB K-means; spectral clustering of bursts (Cucuringu-style); graph neural
nets over stocks; contrastive embeddings of bursts (similar algorithms close together); Bayesian hierarchical
models across stocks; symbolic regression for interpretable rules; distillation of the boosted model into a
small decision tree for presentation.

## 7. Validation and robustness (≈30)
Disjoint names and years ✅; market-excess ✅; placebo events at random times ✅🔄; placebo labels shuffled ✅;
multiple-testing count vs chance ✅; third untouched period (2012–16 or 2025); per-year stability; per-stock
distribution of IC; large vs small stocks; tick-constrained vs not; NYSE- vs NASDAQ-listed; morning vs afternoon;
high vs low volatility days; decision-latency sensitivity; winsorization sensitivity; alternative mid (microprice);
trade-price vs mid outcomes; excluding the first and last 30 minutes; excluding earnings days; stale-quote
filters; bootstrap by stock; permutation of burst sides; factor-neutral returns; correlation with known
short-term reversal; out-of-time rolling windows; data-snooping-robust test (Hansen SPA / White reality check)
across all definitions; deflated Sharpe for any trading claim; pre-registration of each confirmation.

## 8. Data sources and ground truth (≈20)
LOBSTER message + orderbook files (levels 1–10 not yet used ⭐); TAQ (consolidated trades, institutional-size and
retail flags ✅ daily); Hyperliquid Level-4 (trader addresses — true parents) ⭐; synthetic parents injected into
real tapes ✅; 13F / mutual funds ✅; FINRA ATS / OTC weekly volumes; SEC Rule 605 execution-quality reports;
CRSP ✅; Compustat ✅; options (OptionMetrics) for informed-flow cross-checks; news timestamps (RavenPack-like);
index-event calendars ✅; ETF creation / redemption data; short-sale volume (FINRA daily).

## 9. Natural experiments (≈20)
Tick Size Pilot 2016–18 ✅ exploratory; 2023 odd-lot / round-lot redefinition; NASDAQ fee changes; LULD pauses
and halts; index rebalances ✅; quarter-end ✅; option expiries ✅; FOMC / CPI minutes; earnings minutes; Russell
reconstitution day; S&P additions; stock splits (tick size in bps changes overnight); COVID March 2020; meme-stock
episodes (Jan 2021); circuit-breaker days; exchange outages; ETF launches in a sector.

## 10. Far-fetched (≈30)
Language model over the order-flow "sentence" of a stock-day; algorithm "species" tracked for years by
fingerprint; the same timing signature detected in crypto and equities; spectral (audio-style) analysis of the
trade stream for timer frequencies; reinforcement-learning market maker quoting around detected bursts;
diffusion model of the book conditioned on a burst ("painting the market") to simulate continuations; graph of
algorithms that trade against each other; detecting the *same* parent across venues via TAQ clip matching;
predicting which broker algo (TWAP / VWAP / POV / IS) produced a burst from its shape; learning a detector on
Hyperliquid (true parents) and transferring it to NASDAQ; adversarial detection (can an algorithm hide from our
detector?); burst "weather maps" of the market by sector and minute; entropy of the order flow as a state
variable; topological data analysis of child-size / timing point clouds; optimal-transport distance between a
stock's burst distribution today and its normal day; transfer entropy between stocks' burst flows; queueing-
theory model of the touch with burst arrivals; information-theoretic bound on how much any tape-only detector can
know; human-in-the-loop labelling of 500 bursts by eye to calibrate definitions.

## 11. Execution order (what runs next) — forecasting only (2026-09-25)
Fill-based trading items in §5 are kept for the record but are no longer run: the criterion is forecasting power
at the mid (IC, R², hit rate on non-zero moves, decile spreads), per the 2026-09-25 instruction.
1. ✅ Placebo; ✅ v3 definitions / act during the burst / latency; ✅ 2025 out-of-time check; ✅ feature-group
   ablation and stability (`code/forecast_power.py`).
2. ✅ v4 (`code/burst_defs_raw4.py`): term structure +1 s … next close for 7 definitions and matched random-time
   events (burst-specific through the close, zero overnight); realized volatility controlled for pre-event RV (bursts
   add ≤ 0.008); 5-minute panel (bursts add ≤ 0.002); daily (nothing); market timing (nothing).
3. Targets not yet forecast: first-passage (which way the mid moves one tick, and when); one-tick-ahead direction;
   spread and depth next minute; remaining parent volume (continuation size).
4. Features not yet tried as forecasters: multi-level depth imbalance (needs the book rebuilt from messages, as in
   the ClusterLOB pass), decomposed OFI by event type (F10), burst history in the last 1 / 5 / 30 min (F7).
5. G1 Hyperliquid ground truth (awaiting the user's go-ahead to download).
