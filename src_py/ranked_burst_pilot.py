#!/usr/bin/env python3
"""Exploratory quote-snapshot screen; NOT a validated execution backtest.

Frozen realtime run60 score, first-three-packet decisions, next cached prepacket
quote after one second, next quote after 60/300 seconds holding. No model fitting.
Quotes are event-sampled NASDAQ quotes, not NBBO or guaranteed available fills.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import metaorder_features as MF


def pnl(mid0, spread0, mid1, spread1, side):
    """One-share touch P&L / entry midpoint, in bps; includes both half spreads."""
    entry = mid0 * (1 + side * spread0 / 20000)
    exit_price = mid1 * (1 - side * spread1 / 20000)
    return side * (exit_price - entry) / mid0 * 10000


def nw_t(x, lag=10):
    x = np.asarray(x, float)
    if len(x) < 3:
        return None
    z = x - x.mean()
    v = np.dot(z, z) / len(z)
    for k in range(1, min(lag, len(z) - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * np.dot(z[k:], z[:-k]) / len(z)
    se = np.sqrt(max(v, 0) / len(z))
    return float(x.mean() / se) if se > 0 else None


def run(root, model_path, out):
    model = json.loads(model_path.read_text())
    trades, coverage, audits, manifest = [], [], [], []
    files = sorted(root.glob('*/*.npz'))
    assert files, 'No input caches'
    for path in files:
        manifest.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        with np.load(path) as archive:
            day = {k: archive[k] for k in archive.files}
        t = day['time']
        assert np.all(np.diff(t) >= 0)
        _, tab = MF.burst_table(day, 'run', 60)
        if not tab:
            coverage.append(dict(ticker=path.parent.name, date=path.stem, bursts=0))
            continue
        scores = MF.score(model, tab)
        # For run (NOT side streams), previous bursts have already ended at t3.
        # Independently recompute a sample using only the tape known at t3.
        for k in np.unique(np.linspace(0, len(scores)-1, min(5, len(scores)), dtype=int)):
            cutoff = np.searchsorted(t, tab['t3'][k], side='right')
            _, pre = MF.burst_table({key: value[:cutoff] for key, value in day.items()}, 'run', 60)
            match = np.flatnonzero((pre['start'] == tab['start'][k]) & (pre['side'] == tab['side'][k]))
            assert len(match) == 1
            np.testing.assert_allclose(MF.score(model, pre)[match[0]], scores[k], rtol=1e-10, atol=1e-10)
            audits.append(dict(ticker=path.parent.name, date=path.stem, burst=int(k)))
        valid_quote = np.isfinite(day['mid']) & (day['mid'] > 0) & np.isfinite(day['spread_bps']) & (day['spread_bps'] > 0) & (day['spread_bps'] < 20000)
        qi = np.flatnonzero(valid_quote)
        qt = t[qi]
        candidates = np.flatnonzero(np.isfinite(scores) & (tab['t3'] >= 34260) & (tab['t3'] <= 57000))
        coverage.append(dict(ticker=path.parent.name, date=path.stem, bursts=len(scores), candidates=len(candidates)))
        for group in ('all', 'high', 'low'):
            selected = candidates
            if group == 'high':
                selected = candidates[scores[candidates] >= model['threshold_q80']]
            elif group == 'low':
                selected = candidates[scores[candidates] <= model['threshold_q20']]
            for horizon in (60, 300):
                free = -np.inf
                for k in selected:
                    decision = tab['t3'][k]
                    if decision < free:
                        continue
                    i = np.searchsorted(qt, decision + 1, side='left')
                    if i == len(qt) or qt[i] > 57600:
                        continue
                    j = np.searchsorted(qt, qt[i] + horizon, side='left')
                    if j == len(qt) or qt[j] > 57600:
                        raise RuntimeError('Missing session exit: refuse silently censored P&L')
                    a, b = qi[i], qi[j]
                    free = qt[j]
                    m0, m1 = day['mid'][a], day['mid'][b]
                    s0, s1 = day['spread_bps'][a], day['spread_bps'][b]
                    for direction in ('follow', 'fade'):
                        side = int(tab['side'][k]) * (1 if direction == 'follow' else -1)
                        trades.append(dict(ticker=path.parent.name, date=path.stem, group=group,
                            horizon=horizon, direction=direction, side=side, score=float(scores[k]),
                            decision=decision, entry=qt[i], exit=qt[j], entry_delay=qt[i]-decision,
                            exit_delay=qt[j]-qt[i]-horizon, gross_mid_bps=side*(m1/m0-1)*10000,
                            touch_bps=pnl(m0,s0,m1,s1,side), entry_spread=s0, exit_spread=s1))
    df = pd.DataFrame(trades)
    assert len(df)
    assert (df.entry >= df.decision + 1).all()
    assert (df.exit >= df.entry + df.horizon).all()
    for _, g in df.groupby(['ticker','date','group','horizon','direction']):
        assert (g.entry.to_numpy()[1:] >= g.exit.to_numpy()[:-1]).all()
    cells = ['group','horizon','direction']
    # Equal-name within date, equal date overall; never treat bursts as independent.
    nd = df.groupby(cells+['date','ticker'])[['gross_mid_bps','touch_bps']].mean().reset_index()
    daily = nd.groupby(cells+['date'])[['gross_mid_bps','touch_bps']].mean().reset_index()
    summaries = []
    for keys, g in daily.groupby(cells):
        cell = df[(df.group==keys[0]) & (df.horizon==keys[1]) & (df.direction==keys[2])]
        summaries.append(dict(group=keys[0], horizon=int(keys[1]), direction=keys[2], days=len(g), trades=len(cell),
            gross_mid_bps=float(g.gross_mid_bps.mean()), touch_bps=float(g.touch_bps.mean()),
            net_05bps=float(g.touch_bps.mean()-.5), net_1bps=float(g.touch_bps.mean()-1),
            descriptive_nw10_t=nw_t(g.touch_bps), entry_delay_p95=float(cell.entry_delay.quantile(.95)),
            exit_delay_p95=float(cell.exit_delay.quantile(.95))))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/'trades.csv.gz',index=False)
    daily.to_csv(out/'daily.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    summary = dict(status='exploratory_snapshot_screen_not_execution_backtest',
        stocks=sorted(df.ticker.unique()), dates=sorted(df.date.unique()),
        input_files=len(files), prefix_audits=len(audits), latency_seconds=1,
        score_model=str(model_path), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cells=summaries, inputs=manifest)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False))
    print('Files:',len(files),'prefix audits:',len(audits))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--packets',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    run(a.packets,a.model,a.out)
