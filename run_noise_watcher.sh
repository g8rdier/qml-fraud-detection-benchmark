#!/usr/bin/env bash
# run_noise_watcher.sh
# Monitors running noise sweep processes. As soon as any slot frees up,
# launches the next queued job (VQC-only first, then QSVM-only per level).
# Auto-merges and plots when everything is done.
#
# Queue order for remaining levels:
#   p=0.02 VQC → p=0.05 VQC → p=0.02 QSVM → p=0.05 QSVM
# VQC jobs go first so QSVM can start as early as possible.

set -euo pipefail

BASE_DIR="results/noise"
ENV_VARS="OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10"
ALL_LEVELS=(0.0 0.001 0.005 0.01 0.02 0.05)

# Each entry: "noise_level:mode"  mode = vqc | qsvm
QUEUE=(
    "0.02:vqc"
    "0.05:vqc"
    "0.02:qsvm"
    "0.05:qsvm"
)
QUEUE_IDX=0

log() { echo "[$(date +%H:%M:%S)] $*"; }

launch_next() {
    if [ $QUEUE_IDX -ge ${#QUEUE[@]} ]; then
        return
    fi

    local ENTRY="${QUEUE[$QUEUE_IDX]}"
    local P="${ENTRY%%:*}"
    local MODE="${ENTRY##*:}"
    local DIR="$BASE_DIR/p${P}"
    mkdir -p "$DIR"

    local FLAG=""
    [ "$MODE" = "vqc" ]  && FLAG="--vqc-only"
    [ "$MODE" = "qsvm" ] && FLAG="--qsvm-only"

    log "Launching p=$P ($MODE) → $DIR"
    env $ENV_VARS pixi run python run_noise.py sweep \
        --data-path data/raw/creditcard.csv \
        --n-qubits 8 \
        --noise-levels "$P" \
        --noise-dir "$DIR" \
        --no-plots \
        $FLAG \
        >> "$DIR/run.log" 2>&1 &
    log "  PID $!"
    QUEUE_IDX=$((QUEUE_IDX + 1))
}

log "Watcher started (granular VQC/QSVM mode)."
log "Queue: ${QUEUE[*]}"
log "Monitoring current processes..."

PREV_COUNT=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')

while true; do
    sleep 120

    CURR_COUNT=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')

    if [ "$CURR_COUNT" -lt "$PREV_COUNT" ]; then
        FINISHED=$((PREV_COUNT - CURR_COUNT))
        log "$FINISHED slot(s) freed. Running: $CURR_COUNT. Launching next..."
        for ((i=0; i<FINISHED; i++)); do
            launch_next
        done
        PREV_COUNT=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')
    else
        PREV_COUNT=$CURR_COUNT
    fi

    # Done when no processes running and queue exhausted
    CURR_COUNT=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')
    if [ "$CURR_COUNT" -eq 0 ] && [ "$QUEUE_IDX" -ge "${#QUEUE[@]}" ]; then
        log "All jobs done. Merging..."
        break
    fi
done

# Merge + plot
DIRS=()
for P in "${ALL_LEVELS[@]}"; do
    DIRS+=("$BASE_DIR/p${P}")
done

pixi run python run_noise.py merge \
    --noise-dirs "${DIRS[@]}" \
    --out-dir "$BASE_DIR"

log "============================================================"
log "SWEEP COMPLETE"
log "Results: $BASE_DIR/noise_results.json"
log "Plot:    $BASE_DIR/noise_vs_metric.png"
log "============================================================"
