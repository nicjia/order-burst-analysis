from pathlib import Path
import json,csv,gzip,collections,hashlib
p=Path('results/burst_information_v1');s=json.loads((p/'summary.json').read_text())
# Descriptive model-level losses, independently grouped from saved predictions.
levels=[];name_rows=[];leave_rows=[]
models=['ridge_state','gbt_state','gbt_state_regime','gbt_state_regime_burst','gbt_state_regime_burst_score']
for file in sorted((p/'predictions').glob('*.csv.gz')):
 stage,target=file.name.removesuffix('.csv.gz').split('_',1)
 with gzip.open(file,'rt') as f:rows=list(csv.DictReader(f))
 for cohort in ['seen','heldout']:
  group=[r for r in rows if r['cohort']==cohort]
  nd=collections.defaultdict(list)
  for r in group:
   for m in models:nd[r['date'],r['ticker'],m].append((float(r['target'])-float(r[m]))**2)
  nd={k:sum(v)/len(v) for k,v in nd.items()}
  daily=collections.defaultdict(list)
  for (d,n,m),v in nd.items():daily[d,m].append(v)
  for m in models:
   vals=[sum(v)/len(v) for (d,mm),v in daily.items() if mm==m]
   levels.append(dict(stage=stage,target=target,cohort=cohort,model=m,mse=sum(vals)/len(vals)))
  if (stage,target,cohort)==('third','return_60s','heldout'):
   names=sorted({r['ticker'] for r in group})
   for name in names:
    base=[v for (d,n,m),v in nd.items() if n==name and m==models[2]]
    aug=[v for (d,n,m),v in nd.items() if n==name and m==models[3]]
    name_rows.append(dict(ticker=name,base_mse=sum(base)/len(base),burst_mse=sum(aug)/len(aug),improvement_pct=100*(1-sum(aug)/sum(base))))
    dd=collections.defaultdict(list)
    for (d,n,m),v in nd.items():
     if n!=name:dd[d,m].append(v)
    means={m:sum(sum(v)/len(v) for (d,mm),v in dd.items() if mm==m)/len({d for d,mm in dd}) for m in models}
    leave_rows.append(dict(omitted=name,improvement_pct=100*(1-means[models[3]]/means[models[2]])))
for name,rows in [('model_loss_levels.csv',levels),('posthoc_return_name_breakdown.csv',name_rows),('posthoc_return_leave_one_name.csv',leave_rows)]:
 with (p/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
text='''# Burst information: completed exploratory tests

Completed 2026-09-13. **The primary burst-information test was negative, and this study does not establish a tradable signal. Reconstruction remains open because the legacy joining model had a session-mixing bug.**

## Main results

The study fitted all 75 fixed specifications and reported all 120 paired comparisons. It covers 36 stocks and 40 sampled dates: training on 2023 for 24 stocks, evaluation on 2024 for those stocks plus 12 held-out stocks. Of 1,440 requested stock-days, 1,410 archives were available and 30 confirmed missing. There are 64,618 valid sampled landmarks across 1,382 stock-days with usable landmarks; 28 available stock-days have none. No rows were removed by the extreme-price rule.

1. **Primary test failed.** At the third packet, adding burst features to the state/regime tree model increased five-minute flow MSE by **0.155%** on held-out names (improvement t = **−4.30**). The primary hypothesis is not supported in this specification/sample.
2. **A secondary price improvement does not beat the simpler baseline.** At the third packet, burst features reduce one-minute return MSE by **4.009%** versus the state/regime tree (t = **9.25**). However, the resulting MSE is **87.093 bps²**, compared with **80.997 bps²** for the linear state model: **7.526% worse**. Thus this is an improvement within one model family, not demonstrated superiority to the strongest baseline. Adding the simulation score worsens the burst model's one-minute return MSE by **1.323%**.
3. **Execution evidence is inconclusive.** For held-out third-packet opportunities, the burst timing policy saves **0.185 bps** versus always executing immediately (t = **3.39**), but only **0.096 bps** versus the state/regime policy (t = **0.82**). The corresponding seen-name incremental saving is **−0.083 bps** (t = **−0.56**). This is a required one-share order executed now or 60 seconds later, with one-second latency and common non-overlapping opportunities. It is not a directional trading profit or a finite-size execution backtest.
4. **Reconstruction failure was partly an implementation problem.** Correcting cross-day join training raises simulated candidate-pair recall from **31.0% to 91.6%** with **92.6% precision**. Chaining those joins has **80.3% fragment-pair precision / 82.6% recall**, illustrating accumulated merge errors. These are synthetic dominant-parent labels, not verified market parent identities. See [the reconstruction report](BURST_RECONSTRUCTION_RESULTS.md).

A post-result descriptive check of the secondary one-minute return improvement finds positive improvements in 9 of 12 held-out names. Omitting each name in turn leaves an improvement of 3.062% to 4.584% versus the tree baseline. This rules out a single-name explanation within this panel; it does not repair the weaker performance versus the linear model. These checks added no fits or inferential tests and were performed after inspecting the result.

## Interpretation and publication direction

The useful question is now **which observable execution structure supports reliable probabilistic reconstruction under overlap, pauses, and incomplete observation**, and whether it adds information beyond strong flow/book baselines. The short same-side burst is often a fragment; a clean fragment is not necessarily a complete parent. Exact parent labels are not identified from anonymous times, sizes, and signs without additional assumptions.

The corrected join implementation is an engineering correction, not a publication novelty claim. The controlled recovery failures and the previously established Hurst/recovery ranking counterexample support a potential methodology paper: compare validation criteria against known parents rather than treating persistence or attractive-looking clusters as proof of reconstruction. A realistic next benchmark would introduce explicitly book-adaptive execution, concurrent parents and observed venue thinning, then evaluate on withheld execution policies. Our current random-pause simulator does not implement book adaptation. Actual participant/parent labels would be the strongest external validation.

For price prediction, the next justified test is a burst augmentation of the stronger linear/shrinkage baseline with adequate training data, frozen before new evaluation. That experiment was not in this completed matrix and has not been run. Do not choose it merely to rescue a positive result. The simple two-state regime benchmark here is not a replication of published score-driven online changepoint models or ClusterLOB.

None of these findings establishes first-in-literature novelty. Order splitting and persistent flow are established; synthetic calibration alone does not validate inferred parents. Relevant primary sources: [Toth et al.](https://arxiv.org/abs/1108.1632), [Tsaknaki et al.](https://arxiv.org/abs/2307.02375), [Maitrier et al.](https://arxiv.org/abs/2503.18199), [Goliath and Gebbie](https://arxiv.org/abs/2602.19590), and [ClusterLOB](https://arxiv.org/abs/2504.20349).

## Inference and audit limits

Both calendar years had already been explored elsewhere in this project. These are exploratory results, even though names were withheld from these fits. Inference uses equal stock-day means, then equal stocks per date, and Newey–West with 10 lags across only 20 sampled dates. Reported t statistics are descriptive and are not multiplicity-adjusted confirmation gates. The third/sixth/completion training samples contain 5,445 / 952 / 5,438 rows respectively; the sixth-packet fits are particularly small. No evaluation-driven tuning or refitting was performed.

All landmarks use information available at recognition; outcomes begin one second later. Completion is recognized by a breaking packet or one-second silence. Type-5 Direction is ignored. Features include ordinary flow/book state, a simple completed-second regime filter, burst characteristics and a frozen simulation-trained fragment score. The burst feature block also includes starting book conditions and hidden participation; the contrast does not isolate timing alone.

The independent CSV audit reproduces all **120 forecast comparisons and 90 execution statistics**, maximum absolute difference **2.84e−14**. It verifies saved predictions and cost calculations, not independent raw-data reconstruction or causal identification. Separate audits verify all 810 controlled recovery cells and the paired join diagnostic. All 24 unit tests pass. Frozen source hashes matched before fitting. Full provenance and job accounting are in `RESULTS_PROVENANCE.md` and the result group.

## Full fixed comparison matrix

Positive percentages mean lower MSE for the added block; parentheses give the paired daily t statistic. Nonlinearity compares tree state to linear state; regime adds the online filter; burst adds burst features; score adds the frozen simulation score. Reporting all cells avoids selecting only favorable outcomes.

| Landmark | Target | Cohort | Nonlinearity % (t) | Regime % (t) | Burst % (t) | Score % (t) |
|---|---|---|---:|---:|---:|---:|
'''
lookup={(r['stage'],r['target'],r['cohort'],r['contrast']):r for r in s['comparisons']}
for stage in ['third','sixth','completion']:
 for target in ['flow_60s','flow_300s','return_60s','return_300s','wait_cost_60s']:
  for cohort in ['seen','heldout']:
   cells=[]
   for c in ['nonlinearity','regime','burst','simulation_score']:
    r=lookup[stage,target,cohort,c];cells.append(f"{100*r['improvement_fraction']:+.3f} ({r['t']:+.2f})")
   text+='| '+' | '.join([stage,target,cohort]+cells)+' |\n'
text+='''
## Full execution diagnostic

Savings in bps per required one-share order. All five policies use the same score-independent non-overlapping schedule within each landmark/cohort. The parenthesized t statistics have the same exploratory limits as above.

| Landmark | Cohort | Model | Orders | Saving vs immediate (t) | Saving vs state/regime (t) |
|---|---|---|---:|---:|---:|
'''
for r in s['execution_diagnostics']:
 a=r['savings_vs_now'];b=r['savings_vs_state_regime'];fmt=lambda x:f"{x['mean']:+.4f} ({x['t']:+.2f})" if x['t'] is not None else f"{x['mean']:+.4f} (undefined)"
 text+='| '+' | '.join([r['stage'],r['cohort'],r['model'],str(r['n_orders']),fmt(a),fmt(b)])+' |\n'
text+='''
## Reproduction and outputs

Use the project Python environment, one numerical-library thread, and the frozen design. Run `src_py/evaluate_burst_information.py --root results/burst_information_v1`, then `src_py/audit_burst_information.py --root results/burst_information_v1`.

The result group contains `summary.json`, `independent_audit.json`, `daily_comparisons.csv`, 75 saved models, 15 prediction panels, `model_loss_levels.csv`, and the explicitly post-result name diagnostics. `manifest.json` records extraction sources; `evaluation_manifest.json` records evaluation sources before fitting. `tail_merge_audit.json` accounts for all planned extractions. No confirmation sample was consumed in this study.
'''
Path('BURST_INFORMATION_RESULTS.md').write_text(text)
print('wrote report and descriptive loss/robustness tables')
