#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — Unified Order Burst Analysis Pipeline
# ─────────────────────────────────────────────────────────────────────────
#
# Master orchestration script with STRICT train/OOS universe separation.
#
#   TRAIN universe (~50 stocks): Used ONLY for Optuna parameter tuning.
#   OOS universe   (~450 stocks): Used for backtest, research, and panel
#                                  regressions. NEVER seen by Optuna.
#   ALL universe:  TRAIN ∪ OOS.  Used for data/perm (shared preprocessing).
#
# Phases:
#   1.  DATA:      Build burst CSVs from LOBSTER message files  (ALL tickers)
#   1b. HPC-DATA:  Same, but via the Hoffman2 SGE orchestrator   (500-ticker)
#   2.  PERM:      Compute permanence (forward returns)          (ALL tickers)
#   3.  OPTUNA:    Bayesian parameter sweep                      (TRAIN ONLY)
#   4.  BACKTEST:  SGD walk-forward backtest                     (OOS ONLY)
#   5.  RESEARCH:  Reviewer-required analysis scripts            (OOS ONLY)
#   6.  AGGREGATE: Cross-sectional regime + COI panel + FF alpha (OOS ONLY)
#
# Usage:
#   # Full local pipeline (recommended):
#   ./run_pipeline.sh --phase all
#
#   # Full HPC pipeline (data via cluster orchestrator):
#   ./run_pipeline.sh --phase hpc-all --cluster
#
#   # Individual phases:
#   ./run_pipeline.sh --phase data
#   ./run_pipeline.sh --phase hpc-data
#   ./run_pipeline.sh --phase optuna
#   ./run_pipeline.sh --phase research
#   ./run_pipeline.sh --phase aggregate
#
#   # Override universe files:
#   ./run_pipeline.sh --phase all --train-file universes/train_50.txt \
#                                 --oos-file   universes/oos_450.txt
#
#   # Cluster mode (SGE):
#   ./run_pipeline.sh --phase data --cluster
#
# ─────────────────────────────────────────────────────────────────────────
set -Eeo pipefail
trap 'echo "ERROR: line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# ─────────────────────────────────────────────────────────────────────────
# UNIVERSE DEFINITIONS — Strict Train / OOS Separation
# ─────────────────────────────────────────────────────────────────────────
# These can be overridden by --train-file / --oos-file pointing to text
# files with one ticker per line. By default, we use hardcoded lists.
#
# RULE: Optuna (Phase 3) ONLY sees TRAIN_TICKERS.
#       Backtest + Research (Phases 4–5) ONLY see OOS_TICKERS.
#       Data + Perm (Phases 1–2) process ALL tickers.
# ─────────────────────────────────────────────────────────────────────────

# Default TRAIN universe: ~50 representative stocks used for parameter tuning.
# Intentionally diverse: mega-cap tech, financials, healthcare, energy, consumer.
DEFAULT_TRAIN_TICKERS="NVDA TSLA JPM MS AMZN GOOG META NFLX AMD INTC \
QCOM AVGO TXN MU AMAT LRCX KLAC MRVL SNPS CDNS \
GS BAC C WFC USB PNC SCHW AXP COF BK \
UNH JNJ PFE ABBV LLY MRK TMO ABT DHR BMY \
XOM CVX COP SLB EOG MPC VLO PSX HAL OXY"

# Default OOS universe: stocks NEVER seen during Optuna tuning.
# These are the tickers used for all publishable results.
DEFAULT_OOS_TICKERS="AAPL MSFT ADBE CRM ORCL NOW PANW CRWD FTNT ZS \
V MA PYPL SQ BLK TROW IVZ BEN FNF ALL \
PG KO PEP COST WMT TGT HD LOW MCD SBUX \
DIS CMCSA CHTR NTES BIDU T VZ TMUS DISH LUMN \
BA RTX LMT GD NOC GE HON CAT DE MMM \
SPY QQQ IWM DIA XLF XLK XLE XLV XLI XLB \
AMT PLD CCI EQIX DLR O SPG VICI WELL ARE \
COIN MARA RIOT SOS HUT BTBT MSTR HOOD SOFI LC \
CVS CI HCA UHS THC ELV CNC MOH WCG DVA \
NEE DUK SO D AEP EXC SRE XEL WEC ES"

# ── Configuration defaults ────────────────────────────────────────────────
PHASE="all"
CLUSTER_MODE=0
TRAIN_FILE=""
OOS_FILE=""

# Signal-flipping: tickers whose COI should be inverted (mean-reverting)
# Reviewer R3 demanded this. Financials structurally mean-revert due to
# ETF arb (XLF) and pairs trading. Treating their reversal as a "failure"
# is the mistake the referee caught.
MEAN_REVERT_TICKERS=${MEAN_REVERT_TICKERS:-"JPM MS GS BAC C WFC USB PNC SCHW AXP COF BK BLK TROW IVZ BEN FNF ALL XLF"}

# Hawkes parameters
HAWKES_BETA=${HAWKES_BETA:-1.0}
TRIGGER_INTENSITY=${TRIGGER_INTENSITY:-0.5}
CANCEL_WINDOW=${CANCEL_WINDOW:-0.050}

# Physical filter defaults
VOL_FRAC=${VOL_FRAC:-0.0001}
DIR_THRESH=${DIR_THRESH:-0.8}
VOL_RATIO=${VOL_RATIO:-0.3}
KAPPA_LONG=${KAPPA_LONG:-0.5}
TAU_MAX=${TAU_MAX:-10.0}

# Date ranges
TRAIN_START=${TRAIN_START:-2023-01-01}
TRAIN_END=${TRAIN_END:-2024-12-31}
OOS_START=${OOS_START:-2019-01-01}
OOS_END=${OOS_END:-2024-12-31}

# Optuna
OPTUNA_TRIALS=${OPTUNA_TRIALS:-100}
OPTUNA_TARGET=${OPTUNA_TARGET:-reg_clop}
HAWKES_TAG=${HAWKES_TAG:-b1p0_i0p3}

# Execution
WORKERS=${WORKERS:-${NSLOTS:-4}}
ROOT=${ROOT:-$(pwd)}

# Aggregation / risk-adjustment inputs (Reviewer R3/R6/B6)
FACTOR_CSV=${FACTOR_CSV:-data/ff5_mom_daily.csv}          # FF5+MOM daily factors
REGIME_CSV=${REGIME_CSV:-results/regime/regime_classifications.csv}
UNIVERSE_FILE=${UNIVERSE_FILE:-universes/full_500.txt}    # 500-ticker (ALL) universe

# ── Parse arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)        PHASE="$2";            shift 2 ;;
        --train-file)   TRAIN_FILE="$2";       shift 2 ;;
        --oos-file)     OOS_FILE="$2";         shift 2 ;;
        --cluster)      CLUSTER_MODE=1;        shift   ;;
        --workers)      WORKERS="$2";          shift 2 ;;
        --root)         ROOT="$2";             shift 2 ;;
        --mean-revert)  MEAN_REVERT_TICKERS="$2"; shift 2 ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Resolve ticker universes ─────────────────────────────────────────────
resolve_tickers() {
    # Universe files are one-ticker-per-line; '#' comments and blank lines are
    # stripped, and only the first whitespace-delimited token per line is kept.
    if [ -n "${TRAIN_FILE}" ] && [ -f "${TRAIN_FILE}" ]; then
        TRAIN_TICKERS=$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "${TRAIN_FILE}" | awk '{print $1}' | tr '\n' ' ' | xargs)
        echo "INFO: Loaded TRAIN universe from ${TRAIN_FILE}"
    else
        TRAIN_TICKERS="${DEFAULT_TRAIN_TICKERS}"
    fi

    if [ -n "${OOS_FILE}" ] && [ -f "${OOS_FILE}" ]; then
        OOS_TICKERS=$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "${OOS_FILE}" | awk '{print $1}' | tr '\n' ' ' | xargs)
        echo "INFO: Loaded OOS universe from ${OOS_FILE}"
    else
        OOS_TICKERS="${DEFAULT_OOS_TICKERS}"
    fi

    # ALL = TRAIN ∪ OOS (for data/perm phases)
    ALL_TICKERS="${TRAIN_TICKERS} ${OOS_TICKERS}"

    # ── FIREWALL CHECK: ensure no overlap ──
    local overlap=""
    for t in ${TRAIN_TICKERS}; do
        for o in ${OOS_TICKERS}; do
            if [ "${t}" = "${o}" ]; then
                overlap="${overlap} ${t}"
            fi
        done
    done
    if [ -n "${overlap}" ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║  FATAL: Train/OOS OVERLAP DETECTED — DATA LEAKAGE!     ║"
        echo "║  Overlapping tickers:${overlap}"
        echo "║  Fix your --train-file or --oos-file to remove overlap. ║"
        echo "╚══════════════════════════════════════════════════════════╝"
        exit 1
    fi

    # Count
    local n_train n_oos
    n_train=$(echo ${TRAIN_TICKERS} | wc -w | xargs)
    n_oos=$(echo ${OOS_TICKERS} | wc -w | xargs)

    echo ""
    echo "  Universe Summary:"
    echo "    TRAIN tickers (Optuna tuning only):  ${n_train}"
    echo "    OOS tickers   (backtest + research):  ${n_oos}"
    echo "    TOTAL tickers (data + perm):           $((n_train + n_oos))"
    echo "    Mean-revert signal flip:               $(echo ${MEAN_REVERT_TICKERS} | wc -w | xargs) tickers"
    echo ""
}

# ── Cluster environment setup ─────────────────────────────────────────────
setup_cluster_env() {
    if [ -f /etc/profile ]; then
        . /etc/profile
    fi
    if [ -f /u/local/Modules/default/init/bash ]; then
        . /u/local/Modules/default/init/bash
        module load gcc/11.3.0 python/3.9.6
    fi
    if [ -d "${ROOT}/.venv" ]; then
        source "${ROOT}/.venv/bin/activate"
    fi
    export PYTHONNOUSERSITE=1
}

# ── Phase 1: DATA — Build burst CSVs (ALL tickers) ───────────────────────
phase_data() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Phase 1: DATA — Building burst CSVs with C++ engine"
    echo "  Scope: ALL tickers (TRAIN ∪ OOS)"
    echo "════════════════════════════════════════════════════════"

    # Ensure data_processor binary exists
    if [ ! -x "${ROOT}/data_processor" ]; then
        echo "INFO: Building data_processor..."
        make -C "${ROOT}" clean && make -C "${ROOT}"
    fi

    mkdir -p "${ROOT}/results" "${ROOT}/logs"

    for ticker in ${ALL_TICKERS}; do
        local stock_dir="${ROOT}/data/${ticker}"
        local raw_csv="${ROOT}/results/bursts_${ticker}_baseline.csv"

        if [ ! -d "${stock_dir}" ]; then
            echo "SKIP: Missing ${stock_dir}"
            continue
        fi

        # Skip if already computed and non-empty
        if [ -s "${raw_csv}" ]; then
            echo "[${ticker}] SKIP: ${raw_csv} already exists"
            continue
        fi

        echo "[${ticker}] Running data_processor → ${raw_csv}"
        "${ROOT}/data_processor" "${stock_dir}" "${raw_csv}" \
            -H "${HAWKES_BETA}" \
            -I "${TRIGGER_INTENSITY}" \
            -w "${CANCEL_WINDOW}" \
            -v "${VOL_FRAC}" \
            -d "${DIR_THRESH}" \
            -r "${VOL_RATIO}" \
            -k 0 \
            -t "${TAU_MAX}" \
            -j "${WORKERS}" \
            -b 34200 -e 57600

        echo "[${ticker}] Done."
    done
}

# ── Phase 2: PERM — Compute permanence (ALL tickers) ─────────────────────
phase_perm() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Phase 2: PERM — Computing permanence & D_b filter"
    echo "  Scope: ALL tickers (TRAIN ∪ OOS)"
    echo "════════════════════════════════════════════════════════"

    for ticker in ${ALL_TICKERS}; do
        local raw_csv="${ROOT}/results/bursts_${ticker}_baseline.csv"
        local unfiltered="${ROOT}/results/bursts_${ticker}_baseline_unfiltered.csv"
        local filtered="${ROOT}/results/bursts_${ticker}_baseline_filtered.csv"

        if [ ! -f "${raw_csv}" ]; then
            echo "SKIP: Missing ${raw_csv}"
            continue
        fi

        # Skip empty baselines (no-data sentinels / 0-burst tickers) — otherwise
        # compute_permanence errors on an empty frame and aborts the whole phase.
        if [ "$(wc -l < "${raw_csv}")" -le 1 ]; then
            echo "[${ticker}] SKIP: empty baseline (no bursts)"
            continue
        fi

        # Skip if already computed
        if [ -s "${unfiltered}" ] && [ -s "${filtered}" ]; then
            echo "[${ticker}] SKIP: permanence files already exist"
            continue
        fi

        # Unfiltered (kappa=0)
        echo "[${ticker}] Permanence (kappa=0) → unfiltered"
        python3 src_py/compute_permanence.py "${raw_csv}" \
            "${ROOT}/open_all.csv" "${ROOT}/close_all.csv" \
            --kappa 0 --ticker "${ticker}"

        # Rename output
        local perm_out="${raw_csv%.csv}_filtered.csv"
        if [ -f "${perm_out}" ]; then
            mv -f "${perm_out}" "${unfiltered}"
        fi

        # Filtered (kappa=KAPPA_LONG)
        echo "[${ticker}] Permanence (kappa=${KAPPA_LONG}) → filtered"
        local seed="${ROOT}/results/bursts_${ticker}_baseline_longseed.csv"
        cp -f "${raw_csv}" "${seed}"
        python3 src_py/compute_permanence.py "${seed}" \
            "${ROOT}/open_all.csv" "${ROOT}/close_all.csv" \
            --kappa "${KAPPA_LONG}" --ticker "${ticker}"

        local seed_out="${seed%.csv}_filtered.csv"
        if [ -f "${seed_out}" ]; then
            mv -f "${seed_out}" "${filtered}"
        fi
        rm -f "${seed}"
    done
}

# ── Phase 3: OPTUNA — Bayesian parameter sweep (TRAIN ONLY) ──────────────
phase_optuna() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  Phase 3: OPTUNA — Bayesian regression parameter sweep"
    echo "  *** TRAIN UNIVERSE ONLY — OOS tickers are FIREWALLED ***"
    echo "  Target: ${OPTUNA_TARGET}  |  Trials: ${OPTUNA_TRIALS}"
    echo "  Train window: ${TRAIN_START} → ${TRAIN_END}"
    echo "═══════════════════════════════════════════════════════════════════"

    for ticker in ${TRAIN_TICKERS}; do
        local data_path="${ROOT}/results/bursts_${ticker}_baseline_unfiltered.csv"
        if [ ! -f "${data_path}" ]; then
            echo "WARN: Missing ${data_path}; skipping Optuna for ${ticker}"
            continue
        fi

        echo "[${ticker}] Running Optuna sweep (TRAIN)..."
        python3 src_py/optuna_regression_sweep.py \
            --ticker "${ticker}" \
            --target "${OPTUNA_TARGET}" \
            --hawkes-tag "${HAWKES_TAG}" \
            --trials "${OPTUNA_TRIALS}" \
            --start-date "${TRAIN_START}" \
            --end-date "${TRAIN_END}"
    done

    echo ""
    echo "  Optuna complete. Parameters derived from TRAIN universe only."
    echo "  These params will be applied UNCHANGED to the OOS universe."
}

# ── Phase 4: BACKTEST — SGD walk-forward simulation (OOS ONLY) ────────────
phase_backtest() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  Phase 4: BACKTEST — SGD walk-forward simulation"
    echo "  *** OOS UNIVERSE ONLY — using TRAIN-derived params ***"
    echo "  OOS date range: ${OOS_START} → ${OOS_END}"
    echo "═══════════════════════════════════════════════════════════════════"

    mkdir -p "${ROOT}/results/sgd_backtests_oos"

    # Resolve the universal params from TRAIN universe.
    # Use the median/consensus params across TRAIN tickers.
    local first_train_ticker
    read -r first_train_ticker <<< "${TRAIN_TICKERS}"
    local proxy_json="${ROOT}/results/optuna_regression/${first_train_ticker}/best_regression_params_${OPTUNA_TARGET}_${HAWKES_TAG}.json"

    if [ ! -f "${proxy_json}" ]; then
        echo "ERROR: No Optuna params found. Run --phase optuna first."
        echo "  Expected: ${proxy_json}"
        return 1
    fi

    # Parse universal params
    local params
    params=$(python3 -c "
import json
obj = json.load(open('${proxy_json}'))
print(obj.get('hawkes_tag', '${HAWKES_TAG}'))
print(obj['vol_frac'])
print(obj['dir_thresh'])
print(obj['vol_ratio'])
print(obj.get('kappa', 0.0))
")
    local use_tag vol_frac dir_thresh vol_ratio kappa
    read -r use_tag vol_frac dir_thresh vol_ratio kappa <<< "${params}"

    echo "  Universal params (from TRAIN): tag=${use_tag} vf=${vol_frac} dt=${dir_thresh} vr=${vol_ratio} k=${kappa}"
    echo ""

    for ticker in ${OOS_TICKERS}; do
        local data_path="${ROOT}/results/bursts_${ticker}_baseline_unfiltered.csv"
        local out_prefix="${ROOT}/results/sgd_backtests_oos/${ticker}_${OPTUNA_TARGET}_${use_tag}"

        if [ ! -f "${data_path}" ]; then
            echo "SKIP: Missing OOS data for ${ticker}"
            continue
        fi

        echo "[${ticker}] Running SGD backtest (OOS)..."
        python3 src_py/online_sgd_backtest.py \
            --data "${data_path}" \
            --target "${OPTUNA_TARGET}" \
            --hawkes-tag "${use_tag}" \
            --vol-frac "${vol_frac}" \
            --dir-thresh "${dir_thresh}" \
            --vol-ratio "${vol_ratio}" \
            --kappa "${kappa}" \
            --start-date "${OOS_START}" \
            --end-date "${OOS_END}" \
            --ticker "${ticker}" \
            --execution-mode phase3_flow \
            --signal-mode direction \
            --position-mode fixed_aum \
            --fixed-aum 10000000 \
            --round-trip-bps-cost 1.0 \
            --daily-open-csv "${ROOT}/open_all.csv" \
            --daily-close-csv "${ROOT}/close_all.csv" \
            --debug-trades-out "${out_prefix}_debug_trades.csv" \
            --debug-signals-out "${out_prefix}_debug_signals.csv" \
            | tee "${out_prefix}.log"
    done
}

# ── Phase 5: RESEARCH — Reviewer analysis (OOS ONLY) ─────────────────────
phase_research() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  Phase 5: RESEARCH — Reviewer-Required Analysis Suite"
    echo "  *** OOS UNIVERSE ONLY — no data leakage ***"
    echo "═══════════════════════════════════════════════════════════════════"

    mkdir -p "${ROOT}/results/research"

    # ── Per-ticker analysis on OOS universe ──
    for ticker in ${OOS_TICKERS}; do
        local unfiltered="${ROOT}/results/bursts_${ticker}_baseline_unfiltered.csv"
        local filtered="${ROOT}/results/bursts_${ticker}_baseline_filtered.csv"

        if [ ! -f "${unfiltered}" ]; then
            echo "SKIP: Missing ${unfiltered}"
            continue
        fi

        # Only pass --filtered if the kappa-filtered file actually exists;
        # otherwise the research scripts try to read a missing path and crash.
        [ -f "${filtered}" ] || filtered=""

        echo ""
        echo "──── Research Analysis: ${ticker} (OOS) ────"

        # 1. Poisson baseline test (Reviewer B2)
        echo "[${ticker}] Poisson null-model test..."
        python3 src_py/poisson_baseline_test.py "${unfiltered}" \
            ${filtered:+--filtered "${filtered}"} \
            --ticker "${ticker}" \
            --start-date "${OOS_START}" --end-date "${OOS_END}" \
            2>&1 | tee "${ROOT}/results/research/${ticker}_poisson_test.log"

        # 2. Naive baseline markout (Reviewer R1)
        echo "[${ticker}] Naive baseline markout..."
        python3 src_py/naive_baseline_markout.py "${unfiltered}" \
            ${filtered:+--filtered "${filtered}"} \
            --ticker "${ticker}" \
            --start-date "${OOS_START}" --end-date "${OOS_END}" \
            2>&1 | tee "${ROOT}/results/research/${ticker}_naive_baseline.log"

        # 3. Direction ablation (Reviewer R2)
        echo "[${ticker}] Direction feature ablation..."
        python3 src_py/ablation_study.py "${unfiltered}" \
            --ticker "${ticker}" \
            --target "${OPTUNA_TARGET}" \
            --start-date "${OOS_START}" --end-date "${OOS_END}" \
            2>&1 | tee "${ROOT}/results/research/${ticker}_ablation.log"

        # 4. Time-of-day stratification (Reviewer B9)
        echo "[${ticker}] Time-of-day analysis..."
        python3 src_py/time_of_day_analysis.py "${unfiltered}" \
            --ticker "${ticker}" \
            --start-date "${OOS_START}" --end-date "${OOS_END}" \
            2>&1 | tee "${ROOT}/results/research/${ticker}_time_of_day.log"

        # 5. Transaction cost grid (Reviewer R5/B8)
        echo "[${ticker}] Transaction cost sensitivity..."
        python3 src_py/transaction_cost_grid.py "${unfiltered}" \
            --ticker "${ticker}" \
            --start-date "${OOS_START}" --end-date "${OOS_END}" \
            --output-csv "${ROOT}/results/research/${ticker}_tc_grid.csv" \
            2>&1 | tee "${ROOT}/results/research/${ticker}_tc_grid.log"
    done

    # ── Cross-ticker analysis (OOS universe only) ──

    # 6. Multiple testing correction (TRAIN tickers — corrects Optuna results)
    echo ""
    echo "[ALL] Multiple testing correction (TRAIN Optuna results)..."
    python3 src_py/multiple_testing_correction.py \
        "${ROOT}/results/optuna_regression/" \
        --all-tickers \
        --n-trials "${OPTUNA_TRIALS}" \
        2>&1 | tee "${ROOT}/results/research/multiple_testing_correction.log"

    # 7. Panel regression (OOS tickers only — with signal flipping)
    local oos_csv
    oos_csv=$(echo "${OOS_TICKERS}" | tr ' ' ',')
    local mr_csv
    mr_csv=$(echo "${MEAN_REVERT_TICKERS}" | tr ' ' ',')

    # Prefer the data-driven regime CSV (Reviewer R3) when present; the
    # hardcoded --mean-revert-tickers list remains as a fallback. Add the
    # FF5+MOM factor file for risk-adjusted long-short alpha when present.
    local pr_regime_arg="" pr_factor_arg=""
    if [ -f "${REGIME_CSV}" ]; then
        pr_regime_arg="--regime-csv ${REGIME_CSV}"
    fi
    if [ -f "${FACTOR_CSV}" ]; then
        pr_factor_arg="--factor-csv ${FACTOR_CSV}"
    fi

    echo ""
    echo "[OOS] Panel regression (Fama-MacBeth, with signal flipping)..."
    python3 src_py/panel_regression.py \
        --burst-dir "${ROOT}/results/" \
        --tickers "${oos_csv}" \
        --open-csv "${ROOT}/open_all.csv" \
        --close-csv "${ROOT}/close_all.csv" \
        --mean-revert-tickers "${mr_csv}" \
        ${pr_regime_arg} ${pr_factor_arg} \
        --start-date "${OOS_START}" \
        --end-date "${OOS_END}" \
        --output-csv "${ROOT}/results/research/coi_panel_oos.csv" \
        2>&1 | tee "${ROOT}/results/research/panel_regression_oos.log"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Research analysis complete. Results in: results/research/"
    echo "  ALL results are from the OOS universe (no data leakage)."
    echo "════════════════════════════════════════════════════════"
}

# ── Phase 1-HPC: HPC-DATA — delegate burst building to the cluster ───────
# Replaces the local phase_data loop with the Hoffman2 batch orchestrator,
# which rsyncs .7z files from lobster2, submits the SGE job array, verifies
# per-ticker outputs, and cleans up staged data (plan Component 1).
phase_hpc_data() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Phase 1-HPC: HPC-DATA — Hoffman2 batch orchestration"
    echo "  Universe: ${UNIVERSE_FILE} (500-ticker scale)"
    echo "════════════════════════════════════════════════════════"

    local orch="${ROOT}/hoffman2/master_orchestrator.sh"
    if [ ! -x "${orch}" ]; then
        echo "ERROR: missing or non-executable ${orch}"
        echo "       (run hoffman2/setup_hoffman2.sh first, and launch this"
        echo "        from a tmux session on the Hoffman2 DTN node)."
        return 1
    fi

    echo "INFO: Delegating to ${orch}"
    echo "      Run this inside tmux on the DTN; it blocks on qsub -sync y."
    "${orch}"
}

# ── Phase 6: AGGREGATE — cross-sectional post-processing (OOS) ────────────
# Runs after all per-ticker burst/permanence CSVs exist. Produces the
# data-driven regime classification (Reviewer R3), the cross-sectional IC
# distribution + COI panel (R6/B3/B4), and FF5+MOM risk-adjusted long-short
# alpha (B6/M5).
phase_aggregate() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  Phase 6: AGGREGATE — cross-sectional results (OOS universe)"
    echo "═══════════════════════════════════════════════════════════════════"

    mkdir -p "${ROOT}/results/regime" "${ROOT}/results/aggregate"

    local oos_csv
    oos_csv=$(echo "${OOS_TICKERS}" | tr ' ' ',')

    # 1. Data-driven regime classification (supersedes hardcoded list, R3)
    echo "[AGG] Regime classification (rolling beta + burst-return corr + K-means)..."
    python3 src_py/regime_classifier.py \
        --burst-dir "${ROOT}/results/" \
        --close-csv "${ROOT}/close_all.csv" \
        --tickers "${oos_csv}" \
        --output-dir "${ROOT}/results/regime" \
        2>&1 | tee "${ROOT}/results/research/regime_classifier.log" || true

    # 2. Cross-sectional aggregation + COI panel + FF-adjusted alpha
    local factor_arg=""
    if [ -f "${FACTOR_CSV}" ]; then
        factor_arg="--factor-csv ${FACTOR_CSV}"
    else
        echo "[AGG] NOTE: ${FACTOR_CSV} not found — FF5+MOM adjustment will be skipped."
    fi
    local regime_arg=""
    if [ -f "${REGIME_CSV}" ]; then
        regime_arg="--regime-csv ${REGIME_CSV}"
    fi

    echo "[AGG] Aggregating cross-sectional results..."
    python3 src_py/aggregate_results.py \
        --results-dir "${ROOT}/results/" \
        --tickers "${oos_csv}" \
        --open-csv "${ROOT}/open_all.csv" \
        --close-csv "${ROOT}/close_all.csv" \
        ${regime_arg} ${factor_arg} \
        --start-date "${OOS_START}" --end-date "${OOS_END}" \
        --run-panel-regression \
        2>&1 | tee "${ROOT}/results/research/aggregate_results.log"

    echo ""
    echo "  Aggregation complete. See results/aggregate/SUMMARY.md"
}

# ── Main dispatch ─────────────────────────────────────────────────────────
main() {
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  Order Burst Analysis Pipeline — Strict Train/OOS Separation   ║"
    echo "║  Phase: ${PHASE}                                                "
    echo "╚══════════════════════════════════════════════════════════════════╝"

    if [ "${CLUSTER_MODE}" = "1" ]; then
        setup_cluster_env
    fi

    cd "${ROOT}"

    # Resolve universes and check for overlap
    resolve_tickers

    case "${PHASE}" in
        data)      phase_data      ;;
        hpc-data)  phase_hpc_data  ;;
        perm)      phase_perm      ;;
        optuna)    phase_optuna    ;;
        backtest)  phase_backtest  ;;
        research)  phase_research  ;;
        aggregate) phase_aggregate ;;
        all)
            phase_data
            phase_perm
            phase_optuna
            phase_backtest
            phase_research
            phase_aggregate
            ;;
        hpc-all)
            phase_hpc_data
            phase_perm
            phase_optuna
            phase_backtest
            phase_research
            phase_aggregate
            ;;
        *)
            echo "ERROR: Unknown phase '${PHASE}'"
            echo "Valid phases: data, hpc-data, perm, optuna, backtest, research, aggregate, all, hpc-all"
            exit 1
            ;;
    esac

    echo ""
    echo "Pipeline phase '${PHASE}' completed at $(date)"
}

main "$@"
