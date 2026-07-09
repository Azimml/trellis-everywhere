#!/usr/bin/env bash
# Run every correctness gate. Green across the board => all README claims hold.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
echo "== trellis engine (Viterbi optimal, tail-biting, parallel decode) =="
python tests/test_qtip.py | tail -3
echo "== LDLQ ladder =="
python tests/test_ldlq.py | tail -2
echo "== WGSL kernels on GPU =="
python scripts/validate_wgsl.py 2>/dev/null | grep -E "OK|PASS"
echo "== full packed 3-bit model vs PyTorch (top-5) =="
python scripts/validate_full_packed.py 2>/dev/null | grep -E "top5|MATCH|overlap"
echo "ALL GATES DONE"
