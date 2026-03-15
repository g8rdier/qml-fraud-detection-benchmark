#!/usr/bin/env bash
# run_noise_queue.sh
# Runs noise levels in batches of 2 to control memory usage,
# then merges all results and generates the final plot.
#
# Usage:
#   bash run_noise_queue.sh

set -euo pipefail

NOISE_LEVELS=(0.0 0.001 0.005 0.01 0.02 0.05)
BASE_DIR="results/noise"
BATCH_SIZE=2
ENV_VARS="OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10"

echo "============================================================"
echo "  QML Noise Sweep — Queued Mode (batches of $BATCH_SIZE)"
echo "  Started: $(date)"
echo "============================================================"
echo ""

mkdir -p "$BASE_DIR"

# Run in batches of BATCH_SIZE
i=0
while [ $i -lt ${#NOISE_LEVELS[@]} ]; do
    BATCH=("${NOISE_LEVELS[@]:$i:$BATCH_SIZE}")
    echo "[$(date +%H:%M:%S)] Batch: noise_levels = ${BATCH[*]}"

    PIDS=()
    for P in "${BATCH[@]}"; do
        DIR="$BASE_DIR/p${P}"
        mkdir -p "$DIR"
        echo "  Launching p=$P → $DIR"
        env $ENV_VARS pixi run python run_noise.py sweep \
            --data-path data/raw/creditcard.csv \
            --n-qubits 8 \
            --noise-levels "$P" \
            --noise-dir "$DIR" \
            --no-plots \
            > "$DIR/run.log" 2>&1 &
        PIDS+=($!)
    done

    for PID in "${PIDS[@]}"; do
        wait "$PID" && echo "  [$(date +%H:%M:%S)] PID $PID done ✓" \
                    || echo "  [$(date +%H:%M:%S)] PID $PID FAILED ✗"
    done

    echo ""
    i=$((i + BATCH_SIZE))
done

# Merge all results
echo "[$(date +%H:%M:%S)] All levels done. Merging..."
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
