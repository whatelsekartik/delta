#!/usr/bin/env bash
set -euo pipefail

echo "=================================================================="
echo " PHASE 1 & 3: Numerical Equivalence & Mathematical Formula Tests"
echo "=================================================================="
python3 -m tests.test_numerical

echo "=================================================================="
echo " PHASE 2 & 4: Split-Before-Tokenizer Corpus, Training & INT8 Export"
echo "=================================================================="
python3 scripts/run_full_audit.py

echo "=================================================================="
echo " PHASE 5A: Compiling Native C++17 AVX2 Reference Runtime"
echo "=================================================================="
g++ -O3 -mavx2 -mfma -std=c++17 -Wall -Wextra runtime-native/deltacore_cli.cpp -o runtime-native/deltacore_cli

echo "=================================================================="
echo " PHASE 5B: Compiling WebAssembly Runtime (deltacore_runtime.wasm)"
echo "=================================================================="
node scripts/compile_wasm.mjs

echo "=================================================================="
echo " PHASE 4 & 5C: 4-Way Quantization Parity & WASM vs Native Benchmark"
echo "=================================================================="
node scripts/verify_wasm_and_parity.mjs

echo "=================================================================="
echo " ALL OFFLINE & WASM AUDIT STAGES PASSED"
echo "=================================================================="
