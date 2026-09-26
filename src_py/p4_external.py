#!/usr/bin/env python3
"""P4 revisit v1: external institutional and retail proxies for Q2 (P4_REVISIT_DESIGN.md section 6).

Subcommands (outputs under data/p4/, licensed-data derivatives, gitignored):
  cusip  CRSP v2 historical 8-character CUSIPs of every universe PERMNO (stksecurityinfohist).
  f13    13F institutional holdings from WRDS SEC Analytics (wrdssec, parsed from EDGAR 13F-HR filings),
         summed per CUSIP-8 and report quarter over universe CUSIPs, 2012Q4-2025Q3. Per manager (CIK) and
         quarter: the latest original 13F-HR, replaced by the latest 13F-HR/A restatement if any, plus
         13F-HR/A "new holdings" amendments. Share positions only (SH), no put/call rows.
  mf     CRSP mutual-fund holdings at quarter ends, portfolio map and monthly TNA/returns (for holdings
         changes and flow-induced trading), per year for that year's universe PERMNOs.
  iid    TAQ WRDS intraday indicators (BJZZ retail and >= $50k institutional volume) per stock-day.
  index  S&P 500 (CRSP) and Nasdaq-100 (Compustat idxcst_his, linked to PERMNO) additions and deletions.
"""
import argparse
from pathlib import Path

import pandas as pd

import wrds_access as W

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "p4"
YEARS = range(2012, 2026)


def universe_permnos(years=YEARS):
    out = {}
    for y in years:
        p = DATA / ("universe_%d.csv" % y)
        if p.exists():
            out[y] = pd.read_csv(p).permno.astype(int).tolist()
    return out


def pull_cusip(conn):
    perm = sorted({p for v in universe_permnos().values() for p in v})
    frames = []
    for i in range(0, len(perm), 500):
        frames.append(W.query("""
            select permno, secinfostartdt, secinfoenddt, cusip, ticker, primaryexch, issuernm
            from crsp.stksecurityinfohist where permno in %s""", (tuple(perm[i:i + 500]),), conn=conn))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(DATA / "cusip_hist.csv.gz", index=False, compression="gzip")
    print("cusip rows", len(df), "permnos", df.permno.nunique())


F13_SQL = """
with s as (
    select fname, cik, fdate, form, upper(coalesce(amendmenttype, '')) as atype
    from wrdssec.wrds_13f_summary where rdate = %(rdate)s
), orig as (
    select distinct on (cik) cik, fname from s where form = '13F-HR' order by cik, fdate desc, fname desc
), rest as (
    select distinct on (cik) cik, fname from s
    where form = '13F-HR/A' and atype like 'RESTATEMENT%%' order by cik, fdate desc, fname desc
), keep as (
    select coalesce(r.fname, o.fname) as fname from orig o full outer join rest r on o.cik = r.cik
    union
    select fname from s where form = '13F-HR/A' and atype like 'NEW HOLDINGS%%'
)
select left(h.cusip, 8) as cusip8, sum(h.sshprnamt) as shares, count(distinct h.cik) as n_filers,
       sum(h.value) as value
from wrdssec.wrds_13f_holdings h join keep k on h.fname = k.fname
where h.rdate = %(rdate)s and h.sshprnamttype = 'SH' and h.putcall is null and left(h.cusip, 8) in %(cusips)s
group by 1
"""


def pull_f13(conn, quarters=None):
    cus = pd.read_csv(DATA / "cusip_hist.csv.gz", dtype={"cusip": str})
    cusips = tuple(sorted(cus.cusip.dropna().str[:8].unique()))
    qe = pd.date_range("2012-12-31", "2025-09-30", freq="Q") if quarters is None else pd.to_datetime(quarters)
    out = DATA / "f13"; out.mkdir(exist_ok=True)
    for q in qe:
        path = out / ("f13_%s.csv.gz" % q.strftime("%Y%m%d"))
        if path.exists():
            continue
        df = W.query(F13_SQL, dict(rdate=q.strftime("%Y-%m-%d"), cusips=cusips), conn=conn)
        df["rdate"] = q.strftime("%Y%m%d")
        df.to_csv(path, index=False, compression="gzip")
        print("13F", q.date(), "cusips", len(df), "shares", float(df.shares.sum()), flush=True)


def pull_mf(conn):
    for y, perm in universe_permnos(range(2016, 2026)).items():
        path = DATA / ("mf_holdings_%d.csv.gz" % y)
        if path.exists():
            continue
        qe = tuple(d.strftime("%Y-%m-%d") for d in pd.date_range("%d-12-31" % (y - 1), "%d-12-31" % y, freq="Q"))
        frames = []
        for i in range(0, len(perm), 500):
            frames.append(W.query("""
                select crsp_portno, report_dt, permno, nbr_shares
                from crsp_q_mutualfunds.holdings where permno in %s and report_dt in %s""",
                (tuple(perm[i:i + 500]), qe), conn=conn))
        pd.concat(frames).to_csv(path, index=False, compression="gzip")
        W.query("select 1", conn=conn)
        pd.DataFrame(W.query("""
            select crsp_fundno, crsp_portno, begdt, enddt from crsp_q_mutualfunds.portnomap
            where enddt >= %s and begdt <= %s""", ("%d-01-01" % (y - 1), "%d-12-31" % y), conn=conn)).to_csv(
            DATA / ("mf_portno_map_%d.csv.gz" % y), index=False, compression="gzip")
        pd.DataFrame(W.query("""
            select crsp_fundno, caldt, mtna, mret from crsp_q_mutualfunds.monthly_tna_ret_nav
            where caldt between %s and %s""", ("%d-09-01" % (y - 1), "%d-12-31" % y), conn=conn)).to_csv(
            DATA / ("mf_monthly_%d.csv.gz" % y), index=False, compression="gzip")
        print("mf", y, "done", flush=True)


def pull_iid(conn):
    for y in range(2012, 2026):
        path = DATA / ("iid_%d.csv.gz" % y)
        crsp = DATA / ("crsp_%d.csv.gz" % y)
        if path.exists() or not crsp.exists():
            continue
        names = pd.read_csv(crsp, usecols=["ticker"]).ticker.dropna().unique().tolist()
        df = W.query("""
            select date, sym_root, buyvol_retail, sellvol_retail, buynumtrades_retail, sellnumtrades_retail,
                   buyvol_inst50k, sellvol_inst50k, total_vol, total_trade
            from taqm_{y}.wrds_iid_{y} where sym_root in %s and sym_suffix is null""".format(y=y),
                     (tuple(sorted(names)),), conn=conn)
        df.to_csv(path, index=False, compression="gzip")
        print("iid", y, len(df), flush=True)


def pull_index(conn):
    sp = W.query("""select permno, mbrstartdt, mbrenddt from crsp.dsp500list_v2
                    where mbrenddt >= '2011-06-01' or mbrstartdt >= '2011-06-01'""", conn=conn)
    ndx = W.query("""select gvkey, iid, "from" as dfrom, thru from comp.idxcst_his
                     where gvkeyx = '000208' and (thru is null or thru >= '2011-06-01')""", conn=conn)
    link = W.query("""select gvkey, liid, lpermno as permno, linkdt, linkenddt from crsp.ccmxpf_lnkhist
                      where gvkey in %s and linktype in ('LU', 'LC') and linkprim in ('P', 'C')""",
                   (tuple(ndx.gvkey.unique().tolist()),), conn=conn)
    link["linkdt"] = pd.to_datetime(link.linkdt)
    link["linkenddt"] = pd.to_datetime(link.linkenddt.fillna(pd.Timestamp("2099-12-31")))
    events = []
    data_end = pd.Timestamp(sp.mbrenddt.max())      # CRSP closes current memberships at its data end
    for r in sp.itertuples():
        for kind, d in (("add", r.mbrstartdt), ("delete", r.mbrenddt)):
            if kind == "delete" and pd.notna(d) and pd.Timestamp(d) >= data_end:
                continue
            if pd.notna(d) and pd.Timestamp("2012-01-01") <= pd.Timestamp(d) <= pd.Timestamp("2025-12-31"):
                events.append(dict(index="SP500", kind=kind, permno=int(r.permno), date=pd.Timestamp(d).strftime("%Y%m%d")))
    for r in ndx.itertuples():
        for kind, d in (("add", r.dfrom), ("delete", r.thru)):
            if pd.notna(d) and pd.Timestamp("2012-01-01") <= pd.Timestamp(d) <= pd.Timestamp("2025-12-31"):
                dt = pd.Timestamp(d)
                m = link[(link.gvkey == r.gvkey) & (link.liid == r.iid) & (link.linkdt <= dt) & (link.linkenddt >= dt)]
                if len(m):
                    events.append(dict(index="NDX", kind=kind, permno=int(m.permno.iloc[0]), date=dt.strftime("%Y%m%d")))
    ev = pd.DataFrame(events).drop_duplicates().sort_values(["date", "index", "kind"])
    ev.to_csv(DATA / "index_events.csv", index=False)
    print(ev.groupby(["index", "kind", ev.date.str[:4]]).size().unstack(0).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=["cusip", "f13", "mf", "iid", "index"])
    ap.add_argument("--quarters", default=None, help="comma-separated quarter ends for f13 (test)")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    conn = W.connect()
    try:
        for w in args.what:
            if w == "f13":
                pull_f13(conn, args.quarters.split(",") if args.quarters else None)
            else:
                globals()["pull_" + w](conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
