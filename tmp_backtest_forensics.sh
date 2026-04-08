#!/bin/bash
set -euo pipefail

DIR=${1:-$(ls -dt results/grid_backtest_* 2>/dev/null | head -n 1)}
if [[ -z "${DIR}" || ! -d "${DIR}" ]]; then
  echo "ERROR: grid directory not found. Pass it explicitly: ./tmp_backtest_forensics.sh results/grid_backtest_YYYYMMDD_HHMMSS"
  exit 1
fi

echo "Analyzing: $DIR"

summary="$DIR/summary.csv"
if [[ ! -f "$summary" ]]; then
  echo "ERROR: missing $summary"
  exit 1
fi

python3 - <<'PY' "$DIR"
import csv,glob,os,re,sys

dirp=sys.argv[1]
rows=[]
for p in sorted(glob.glob(os.path.join(dirp,'run_*.log'))):
    txt=open(p).read()
    m_run=re.search(r'run_(\d+)_',os.path.basename(p))
    run_id=int(m_run.group(1)) if m_run else -1
    m_tgt=re.search(r'--target\s+(\S+)',txt)
    target=m_tgt.group(1) if m_tgt else ''
    m_sm=re.search(r'Signal mode:\s*(\S+)',txt)
    sm=m_sm.group(1) if m_sm else ''
    m_cb=re.search(r'CostAware buffer=([0-9.]+)',txt)
    cb=m_cb.group(1) if m_cb else 'NA'
    m_tr=re.search(r'Total Trades Fired:\s*([0-9,]+)\s*\(([0-9,]+) Long / ([0-9,]+) Short\)',txt)
    trades=int(m_tr.group(1).replace(',','')) if m_tr else 0
    longs=int(m_tr.group(2).replace(',','')) if m_tr else 0
    shorts=int(m_tr.group(3).replace(',','')) if m_tr else 0
    m_pnl=re.search(r'Cumulative Simulated PnL \(raw\):\s*([-0-9.]+)',txt)
    pnl=float(m_pnl.group(1)) if m_pnl else float('nan')
    m_sh=re.search(r'Annualized Sharpe Ratio:\s*([-0-9.]+)',txt)
    sharpe=float(m_sh.group(1)) if m_sh else float('nan')
    m_sp=re.search(r'Total Spread Cost \(raw\):\s*([-0-9.]+)',txt)
    spread=float(m_sp.group(1)) if m_sp else 0.0
    ppt=(pnl/trades) if trades else 0.0
    long_pct=(100.0*longs/trades) if trades else 0.0
    rows.append((run_id,target,sm,cb,trades,longs,shorts,long_pct,pnl,ppt,sharpe,spread,os.path.basename(p)))

rows.sort(key=lambda r:r[10],reverse=True)
print('run target signal cb trades longs shorts long% pnl pnl/trade sharpe spread log')
for r in rows:
    print(f"{r[0]:>3} {r[1]:>7} {r[2]:>10} {r[3]:>4} {r[4]:>6} {r[5]:>6} {r[6]:>6} {r[7]:>6.1f} {r[8]:>10.2f} {r[9]:>10.4f} {r[10]:>7.2f} {r[11]:>8.2f} {r[12]}")

print('\nTop by Sharpe:')
for r in rows[:3]:
    print(f"run {r[0]} {r[1]} {r[2]} cb={r[3]} trades={r[4]} sharpe={r[10]:.2f} pnl={r[8]:.2f} long%={r[7]:.1f}")

print('\nWorst by Sharpe:')
for r in rows[-3:]:
    print(f"run {r[0]} {r[1]} {r[2]} cb={r[3]} trades={r[4]} sharpe={r[10]:.2f} pnl={r[8]:.2f} long%={r[7]:.1f}")
PY
