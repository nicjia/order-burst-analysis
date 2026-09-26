#!/usr/bin/env python3
"""WRDS extracts for program-evidence-v1 modules D, E and G2 (cached under gitignored data/wrds/).

  crsp     CRSP daily file for point-in-time names (2021, 2024, plus early next-year days) and
           CRSP names (SIC, tickers) for fingerprint-v1 names
  iid      WRDS Intraday Indicators: BJZZ retail and >= $50k trade buy/sell volume per stock-day
  mf       CRSP mutual-fund holdings at calendar quarter-ends, portfolio TNA and returns (for flows)
  etf      ETF Global: daily sum over the 40 largest US equity ETFs of weight x dollar fund flow per
           constituent ticker; first-trading-day weights for pair overlap
  index    CRSP S&P 500 and Compustat Nasdaq-100 changes mapped to permnos
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import wrds_access as W

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wrds"
EVID = ROOT / "data" / "evidence"
YEARS = (2021, 2024)


def pit_permnos(year):
    return pd.read_csv(EVID / ("pit_%d.csv" % year)).permno.astype(int).tolist()


def fingerprint_tickers():
    names = set()
    for g in ("explore_2024", "confirm_2021"):
        p = ROOT / "results" / "fingerprint_v1" / "universe_474.txt"
        names |= set(p.read_text().split())
    return sorted(names)


def pull_crsp(conn):
    for y in YEARS:
        perm = tuple(pit_permnos(y))
        W.cached("crsp_dsf_pit_%d" % y, """
            select permno, date, ret, retx, prc, openprc, vol, shrout, cfacpr, cfacshr, hsiccd, bid, ask
            from crsp.dsf where permno in %s and date between %s and %s""",
                 (perm, "%d-12-01" % (y - 1), "%d-01-15" % (y + 1)), conn=conn)
        W.cached("crsp_names_pit_%d" % y, """
            select permno, namedt, nameendt, ticker, shrcd, exchcd, siccd, comnam
            from crsp.dsenames where permno in %s and nameendt >= %s and namedt <= %s""",
                 (perm, "%d-01-01" % (y - 1), "%d-12-31" % y), conn=conn)
    tick = tuple(fingerprint_tickers())
    W.cached("crsp_names_fingerprint", """
        select permno, namedt, nameendt, ticker, shrcd, exchcd, siccd, comnam
        from crsp.dsenames where ticker in %s and nameendt >= '2020-01-01'""", (tick,), conn=conn)
    W.cached("crsp_dsi", "select date, vwretd, ewretd from crsp.dsi where date between '2020-12-01' and '2025-01-15'", conn=conn)


def pull_iid(conn):
    for y in YEARS:
        names = pd.read_csv(DATA / ("crsp_names_pit_%d.csv.gz" % y)).ticker.dropna().unique().tolist()
        names = tuple(sorted(set(names) | set(fingerprint_tickers())))
        W.cached("iid_%d" % y, """
            select date, sym_root, sym_suffix, buyvol_retail, sellvol_retail, buynumtrades_retail, sellnumtrades_retail,
                   buyvol_inst50k, sellvol_inst50k, buyvol_lr, sellvol_lr, total_vol, total_trade,
                   quotedspread_percent_tw, effectivespread_percent_ave
            from taqm_{y}.wrds_iid_{y} where sym_root in %s and sym_suffix is null""".format(y=y), (names,), conn=conn)


def pull_mf(conn):
    for y in YEARS:
        perm = tuple(pit_permnos(y))
        qe = tuple(pd.Timestamp(d).date() for d in pd.date_range("%d-12-31" % (y - 1), "%d-12-31" % y, freq="QE"))
        W.cached("mf_holdings_%d" % y, """
            select crsp_portno, report_dt, permno, nbr_shares
            from crsp_q_mutualfunds.holdings where permno in %s and report_dt in %s""", (perm, qe), conn=conn)
        W.cached("mf_filers_%d" % y, """
            select report_dt, crsp_portno, count(*) as n_positions
            from crsp_q_mutualfunds.holdings where report_dt in %s group by 1, 2""", (qe,), conn=conn)
        W.cached("mf_portno_map_%d" % y, """
            select crsp_fundno, crsp_portno, begdt, enddt from crsp_q_mutualfunds.portnomap
            where enddt >= %s and begdt <= %s""", ("%d-01-01" % (y - 1), "%d-12-31" % y), conn=conn)
        W.cached("mf_monthly_%d" % y, """
            select m.crsp_fundno, m.caldt, m.mtna, m.mret
            from crsp_q_mutualfunds.monthly_tna_ret_nav m
            where m.caldt between %s and %s""", ("%d-09-01" % (y - 1), "%d-12-31" % y), conn=conn)


def etf_list(year):
    top = pd.read_csv(DATA / ("etf_top_%d.csv" % year))
    top = top[~top.composite_ticker.isin(["PFF"])]
    return top.composite_ticker.head(40).tolist()


def pull_etf(conn):
    for y in YEARS:
        etfs = tuple(etf_list(y))
        W.cached("etf_flow_demand_%d" % y, """
            select c.as_of_date, c.constituent_ticker, sum(c.weight * f.fundflow) as flow_demand_usd,
                   count(*) as n_etfs
            from (select distinct on (as_of_date, composite_ticker, constituent_ticker)
                         as_of_date, composite_ticker, constituent_ticker, weight
                  from etfg_constituents.constituents
                  where composite_ticker in %s and as_of_date between %s and %s and weight is not null
                    and constituent_ticker is not null
                  order by as_of_date, composite_ticker, constituent_ticker, weight desc) c
            join etfg_fund_flow.fund_flow f on f.composite_ticker = c.composite_ticker and f.as_of_date = c.as_of_date
            where f.fundflow is not null
            group by 1, 2""", (etfs, "%d-12-20" % (y - 1), "%d-12-31" % y), conn=conn, refresh=True)
        first = W.query("""select min(as_of_date) as d from etfg_constituents.constituents
                           where as_of_date >= %s and composite_ticker = 'SPY' and weight is not null""",
                        ("%d-01-01" % y,), conn=conn).d.iloc[0]
        W.cached("etf_weights_%d" % y, """
            select distinct on (composite_ticker, constituent_ticker) as_of_date, composite_ticker, constituent_ticker, weight
            from etfg_constituents.constituents where composite_ticker in %s and as_of_date = %s and weight is not null
              and constituent_ticker is not null
            order by composite_ticker, constituent_ticker, weight desc""", (etfs, first), conn=conn, refresh=True)


def pull_index(conn):
    sp = pd.read_csv(DATA / "crsp_sp500_membership_2016on.csv.gz")
    ndx = pd.read_csv(DATA / "idxcst_ndx_2016on.csv.gz", dtype={"gvkey": str, "iid": str})
    link = W.cached("ccm_link_ndx", """
        select gvkey, liid, lpermno as permno, linkdt, linkenddt, linktype, linkprim
        from crsp.ccmxpf_lnkhist where gvkey in %s and linktype in ('LU', 'LC')""",
                    (tuple(ndx.gvkey.unique().tolist()),), conn=conn)
    events = []
    for r in sp.itertuples():
        for kind, d in (("add", r.mbrstartdt), ("delete", r.mbrenddt)):
            if isinstance(d, str) and d[:4] in ("2021", "2024"):
                events.append(dict(index="SP500", kind=kind, permno=int(r.permno), date=d[:10]))
    link["linkdt"] = pd.to_datetime(link.linkdt); link["linkenddt"] = pd.to_datetime(link.linkenddt.fillna("2099-12-31"))
    for r in ndx.itertuples():
        for kind, d in (("add", r.dfrom), ("delete", r.thru)):
            if isinstance(d, str) and d[:4] in ("2021", "2024"):
                dt = pd.Timestamp(d)
                m = link[(link.gvkey == r.gvkey) & (link.liid == r.iid) & (link.linkdt <= dt) & (link.linkenddt >= dt)]
                if len(m):
                    events.append(dict(index="NDX", kind=kind, permno=int(m.permno.iloc[0]), date=d[:10]))
    ev = pd.DataFrame(events).drop_duplicates()
    ev.to_csv(DATA / "index_events_2021_2024.csv", index=False)
    print(ev.groupby(["index", "kind", ev.date.str[:4]]).size())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=["crsp", "iid", "mf", "etf", "index"])
    args = ap.parse_args()
    conn = W.connect()
    try:
        for w in args.what:
            globals()["pull_" + w](conn)
            print("done", w, flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
