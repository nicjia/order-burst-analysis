#!/usr/bin/env python3
"""WRDS PostgreSQL access for program-evidence-v1.

Credentials are read from the repository's gitignored ``.env`` (keys ``username`` and ``password``)
and passed straight to the driver. They are never printed, logged or written anywhere else.
Every extract is licensed data: write it under ``data/wrds/`` (gitignored) and never commit it.
"""
import os
from pathlib import Path

import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parents[1]
HOST, PORT, DBNAME = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"
CACHE = ROOT / "data" / "wrds"


def _credentials():
    values = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip().strip("'\"")
    user = values.get("username") or os.environ.get("WRDS_USERNAME")
    password = values.get("password") or os.environ.get("WRDS_PASSWORD")
    if not user or not password:
        raise RuntimeError(".env must define username and password")
    return user, password


def connect(timeout=60):
    user, password = _credentials()
    return psycopg2.connect(host=HOST, port=PORT, dbname=DBNAME, user=user, password=password,
                            sslmode="require", connect_timeout=timeout,
                            options="-c statement_timeout=0")


def query(sql, params=None, conn=None):
    own = conn is None
    conn = conn or connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c.name for c in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        if own:
            conn.close()


def cached(name, sql, params=None, conn=None, refresh=False):
    """Run a query once and keep the result as a parquet (or csv.gz) file under data/wrds/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (name + ".csv.gz")
    if path.is_file() and not refresh:
        return pd.read_csv(path, low_memory=False)
    frame = query(sql, params, conn)
    tmp = path.with_name(path.name + ".part")
    frame.to_csv(tmp, index=False, compression="gzip")
    tmp.rename(path)
    return frame


if __name__ == "__main__":
    c = connect()
    print(query("select current_date as today, count(*) as libraries from information_schema.schemata", conn=c))
    c.close()
