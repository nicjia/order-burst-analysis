#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# setup_hoffman2.sh — One-time environment setup for Hoffman2 HPC cluster
# ─────────────────────────────────────────────────────────────────────────
#
# Run this ONCE before launching the pipeline. It:
#   1. Loads required modules (GCC 11, Python 3.9)
#   2. Creates a Python virtual environment with all dependencies
#   3. Compiles the C++ data_processor binary
#   4. Verifies that 7z is available (installs p7zip if needed)
#   5. Creates necessary directory structure
#
# Usage:
#   bash hoffman2/setup_hoffman2.sh
#
# ─────────────────────────────────────────────────────────────────────────
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRATCH="/u/scratch/n/nicjia"
PROJECT_DIR="${SCRATCH}/order-burst-analysis"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Hoffman2 Environment Setup — Order Burst Analysis Pipeline    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Project root:  ${ROOT}"
echo "  Scratch:       ${SCRATCH}"
echo ""

# ── 1. Load modules ──────────────────────────────────────────────────────
echo "── Step 1: Loading modules ──"
if [ -f /u/local/Modules/default/init/bash ]; then
    . /u/local/Modules/default/init/bash
else
    echo "WARNING: Module system not found. Attempting to proceed..."
fi

module load gcc/11.3.0 2>/dev/null || echo "WARNING: gcc/11.3.0 not available, using system gcc"
module load python/3.9.6 2>/dev/null || module load python/3.9 2>/dev/null || echo "WARNING: python/3.9.6 not available"

echo "  GCC:    $(gcc --version | head -1)"
echo "  Python: $(python3 --version)"
echo ""

# ── 2. Create Python virtual environment ─────────────────────────────────
echo "── Step 2: Creating Python virtual environment ──"
VENV_DIR="${ROOT}/.venv"
# A venv records an absolute interpreter path; if it was built against a
# python module that is no longer loaded, its interpreter dangles and every
# pip/python call fails with "bad interpreter". Detect that and rebuild.
if [ -d "${VENV_DIR}" ] && "${VENV_DIR}/bin/python3" -c "import sys" 2>/dev/null; then
    echo "  Venv already exists and is functional at ${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
else
    if [ -d "${VENV_DIR}" ]; then
        echo "  Existing venv is broken (dangling interpreter) — recreating..."
        rm -rf "${VENV_DIR}"
    fi
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    echo "  Created venv at ${VENV_DIR}"
fi

echo "  Installing Python dependencies..."
pip install --upgrade pip setuptools wheel 2>&1 | tail -1
# py7zr is a pure-Python .7z extractor used as a fallback by the SGE worker
# when no native 7z/7za binary is available on the compute nodes.
pip install \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    statsmodels \
    linearmodels \
    py7zr \
    2>&1 | tail -5

# Verify critical imports
python3 -c "
import numpy, pandas, scipy, sklearn, statsmodels, linearmodels
print('  ✓ All Python dependencies verified')
print(f'    numpy={numpy.__version__}, pandas={pandas.__version__}')
print(f'    scipy={scipy.__version__}, sklearn={sklearn.__version__}')
print(f'    statsmodels={statsmodels.__version__}, linearmodels={linearmodels.__version__}')
" || {
    echo "ERROR: Python dependency verification failed!"
    exit 1
}
echo ""

# ── 3. Compile C++ data_processor ────────────────────────────────────────
echo "── Step 3: Compiling C++ data_processor ──"
cd "${ROOT}"
make clean 2>/dev/null || true
make
if [ -x "${ROOT}/data_processor" ]; then
    echo "  ✓ data_processor compiled successfully"
else
    echo "ERROR: data_processor compilation failed!"
    exit 1
fi
echo ""

# ── 4. Verify 7z availability ────────────────────────────────────────────
echo "── Step 4: Verifying 7z (p7zip) availability ──"
if command -v 7z &>/dev/null; then
    echo "  ✓ 7z found: $(which 7z)"
elif command -v 7za &>/dev/null; then
    echo "  ✓ 7za found: $(which 7za)"
    # Create an alias script
    mkdir -p "${HOME}/bin"
    cat > "${HOME}/bin/7z" << 'SHIM'
#!/bin/bash
exec 7za "$@"
SHIM
    chmod +x "${HOME}/bin/7z"
    export PATH="${HOME}/bin:${PATH}"
    echo "  Created 7z shim at ${HOME}/bin/7z"
else
    echo "  7z not found. Attempting to build p7zip from source..."
    P7ZIP_DIR="${HOME}/local/p7zip"
    if [ -x "${P7ZIP_DIR}/bin/7z" ] || [ -x "${P7ZIP_DIR}/bin/7za" ]; then
        echo "  ✓ p7zip already built at ${P7ZIP_DIR}"
    else
        TMPDIR_BUILD=$(mktemp -d)
        cd "${TMPDIR_BUILD}"
        # Try to download p7zip source
        if command -v wget &>/dev/null; then
            wget -q "https://github.com/p7zip-project/p7zip/archive/refs/tags/v17.05.tar.gz" -O p7zip.tar.gz 2>/dev/null || true
        elif command -v curl &>/dev/null; then
            curl -sL "https://github.com/p7zip-project/p7zip/archive/refs/tags/v17.05.tar.gz" -o p7zip.tar.gz 2>/dev/null || true
        fi

        if [ -f p7zip.tar.gz ]; then
            tar xzf p7zip.tar.gz
            cd p7zip-*/
            make 7za DEST_DIR="${P7ZIP_DIR}" 2>&1 | tail -3
            mkdir -p "${P7ZIP_DIR}/bin"
            cp bin/7za "${P7ZIP_DIR}/bin/" 2>/dev/null || true
            echo "  ✓ Built p7zip at ${P7ZIP_DIR}"
        else
            echo "  WARNING: Could not download p7zip. Please install manually."
            echo "  The pipeline requires 7z to extract .7z archives."
        fi
        cd "${ROOT}"
        rm -rf "${TMPDIR_BUILD}"
    fi

    # Add to PATH
    mkdir -p "${HOME}/bin"
    if [ -x "${P7ZIP_DIR}/bin/7za" ]; then
        ln -sf "${P7ZIP_DIR}/bin/7za" "${HOME}/bin/7z"
        export PATH="${HOME}/bin:${PATH}"
    fi
fi
echo ""

# ── 4b. Confirm an extractor is usable (native OR py7zr) ─────────────────
echo "── Step 4b: Confirming a .7z extractor is available ──"
if command -v 7z &>/dev/null; then
    echo "  ✓ native 7z available ($(command -v 7z))"
elif command -v 7za &>/dev/null; then
    echo "  ✓ native 7za available ($(command -v 7za))"
elif python3 -c "import py7zr" 2>/dev/null; then
    echo "  ✓ py7zr (pure-Python) available — the SGE worker will use it"
else
    echo "  ERROR: no native 7z/7za and py7zr import failed."
    echo "         Extraction will fail. Fix before launching the pipeline."
    exit 1
fi
echo ""

# ── 5. Create directory structure ────────────────────────────────────────
echo "── Step 5: Creating directory structure ──"
mkdir -p "${ROOT}/results"
mkdir -p "${ROOT}/results/sgd_backtests_oos"
mkdir -p "${ROOT}/results/research"
mkdir -p "${ROOT}/results/regime"
mkdir -p "${ROOT}/logs"
mkdir -p "${ROOT}/hoffman2"
mkdir -p "${SCRATCH}/lobster_staging"
echo "  ✓ Directory structure created"
echo ""

# ── 6. Verify SSH access to lobster2 ────────────────────────────────────
echo "── Step 6: Verifying SSH access to lobster2 ──"
if ssh -o ConnectTimeout=5 -o BatchMode=yes nicjia@lobster2.math.ucla.edu "echo ok" 2>/dev/null | grep -q "ok"; then
    echo "  ✓ SSH to lobster2.math.ucla.edu successful"
    # Check data availability
    SAMPLE_COUNT=$(ssh nicjia@lobster2.math.ucla.edu "ls /lobster/2023/ 2>/dev/null | head -5 | wc -l" 2>/dev/null || echo "0")
    echo "  Sample date folders in /lobster/2023/: ${SAMPLE_COUNT}"
else
    echo "  WARNING: SSH to lobster2 failed. Ensure SSH keys are configured."
    echo "  The pipeline requires passwordless SSH access to lobster2.math.ucla.edu"
fi
echo ""

# ── 7. Summary ───────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete                                                ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  To run the pipeline:                                         ║"
echo "║    1. Start a tmux session on the DTN node                    ║"
echo "║    2. source ${VENV_DIR}/bin/activate                         ║"
echo "║    3. bash hoffman2/master_orchestrator.sh                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
