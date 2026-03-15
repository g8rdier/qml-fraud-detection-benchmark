#!/usr/bin/env bash
# run_noise_parallel.sh
# Runs all 6 noise levels in parallel, one process per level,
# then merges results and generates the final plot.
#
# Usage:
#   bash run_noise_parallel.sh
#
# Prerequisites:
#   - pixi environment installed (pixi install)
#   - data/raw/creditcard.csv present
#   - results/models/random_forest.joblib and xgboost.joblib present

set -euo pipefail

NOISE_LEVELS=(0.0 0.001 0.005 0.01 0.02 0.05)
BASE_DIR="results/noise"
ENV="OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10"

echo "============================================================"
echo "  QML Noise Sweep — Parallel Mode (${#NOISE_LEVELS[@]} levels)"
echo "  Started: $(date)"
echo "============================================================"

mkdir -p "$BASE_DIR"

# Launch one process per noise level
PIDS=()
for P in "${NOISE_LEVELS[@]}"; do
    DIR="$BASE_DIR/p${P}"
    mkdir -p "$DIR"
    echo "  Launching noise_level=$P → $DIR"
    env $ENV pixi run python run_noise.py sweep \
        --data-path data/raw/creditcard.csv \
        --n-qubits 8 \
        --noise-levels "$P" \
        --noise-dir "$DIR" \
        --no-plots \
        > "$DIR/run.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} processes launched. Waiting for completion..."
echo "Monitor with: tail -f $BASE_DIR/p*/run.log"
echo ""

# Wait for all to finish
FAILED=0
for i in "${!PIDS[@]}"; do
    P="${NOISE_LEVELS[$i]}"
    PID="${PIDS[$i]}"
    if wait "$PID"; then
        echo "  [$(date +%H:%M:%S)] noise_level=$P done ✓"
    else
        echo "  [$(date +%H:%M:%S)] noise_level=$P FAILED ✗ (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED level(s) failed. Check logs in $BASE_DIR/p*/run.log"
fi

# Merge all results into one JSON and generate plot
echo "Merging results..."
DIRS=()
for P in "${NOISE_LEVELS[@]}"; do
    DIRS+=("$BASE_DIR/p${P}")
done

pixi run python run_noise.py merge \
    --noise-dirs "${DIRS[@]}" \
    --out-dir "$BASE_DIR"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Results: $BASE_DIR/noise_results.json"
echo "  Plot:    $BASE_DIR/noise_vs_metric.png"
echo "============================================================"
