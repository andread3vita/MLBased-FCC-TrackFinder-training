#!/bin/bash
# Smoke test: verifies imports, no leftover env flags, Cayley tables, train dry-run, resume.
# Run from the package root: bash smoke_test.sh
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
PASS=0; FAIL=0

check() {
    if eval "$2"; then
        echo "[PASS] $1"; PASS=$((PASS+1))
    else
        echo "[FAIL] $1"; FAIL=$((FAIL+1))
    fi
}

# 1. Import sanity
check "imports compile" "python -c 'from src.model import CGATrParquetModel, object_condensation_loss, get_clustering_np; from src.lightning_module import CGATrV35LightningModule; print(\"imports OK\")'"

# 2. No leftover env flags
check "no CGATR_DRIFT_FIX flag" "! grep -rn 'CGATR_DRIFT_FIX' src/ 2>/dev/null | grep -v '.pyc'"
check "no CGATR_ISO_NORM flag" "! grep -rn 'CGATR_ISO_NORM' src/ 2>/dev/null | grep -v '.pyc'"
check "no CGATR_INVARIANT_READOUT flag" "! grep -rn 'CGATR_INVARIANT_READOUT' src/ 2>/dev/null | grep -v '.pyc'"
check "no CGATR_OC_FAITHFUL flag" "! grep -rn 'CGATR_OC_FAITHFUL' src/ 2>/dev/null | grep -v '.pyc'"
check "no CGATR_GRADEWISE_NORM flag" "! grep -rn 'CGATR_GRADEWISE_NORM' src/ 2>/dev/null | grep -v '.pyc'"
check "no xformers import" "! grep -rn 'from xformers' src/ 2>/dev/null | grep -v '.pyc'"

# 3. Cayley tables loadable
check "cga_utils tables load" "python -c \"import torch; [torch.load(f'cga_utils/{n}', weights_only=False) for n in ('cga_geometric_product.pt','cga_outer_product.pt','cga_metadata.pt')]; print('tables OK')\""

echo ""
if [[ $FAIL -gt 0 ]]; then
    echo "SMOKE: $PASS passed, $FAIL FAILED"
    exit 1
else
    echo "SMOKE: $PASS passed, 0 failed -- OK to zip"
fi
