# Reproducing the code checks

The message reconstruction example requires Python 3.10 or newer, NumPy, and pandas. The original burst detector requires a C++17 compiler and POSIX headers.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-core.txt
make
python -m unittest discover -s tests -p 'test_*.py'
python examples/synthetic_book.py
```

The example creates invented messages in a temporary directory. Tests cover best-quote depletion, partial cancellation, hidden executions leaving the displayed book unchanged, and backward-looking quote lookup.

`requirements-core.txt` covers these checks, not every historical modeling script. Research scripts may additionally require SciPy, statsmodels, scikit-learn, Optuna, and their data sources.

## Research records

Use [VERIFIED_RESULTS.md](VERIFIED_RESULTS.md) for the current result definitions and exclusions, and [RESULTS_PROVENANCE.md](RESULTS_PROVENANCE.md) for extraction and aggregation records. The paper sources contain the manuscript tables. Older strategy and methods notes describe earlier experiments and should not override the verified record.

Historical tables require separately licensed NASDAQ message data. Unit tests establish code behavior on controlled inputs; they do not reproduce the empirical findings. Hidden-execution signs must come from the study's stated signing convention, not the generic parser's direction field.
