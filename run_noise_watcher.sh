#!/usr/bin/env bash
# run_noise_watcher.sh
# Monitors running noise sweep processes. As soon as any one finishes,
# launches the next queued level. Then merges all results when everything is done.

set -euo pipefail

BASE_DIR="results/noise"
ENV_VARS="OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10"
QUEUE=(0.02 0.05)
ALL_LEVELS=(0.0 0.001 0.005 0.01 0.02 0.05)
QUEUE_IDX=0

log() { echo "[$(date +%H:%M:%S)] $*"; }

launch_next() {
    if [ $QUEUE_IDX -lt ${#QUEUE[@]} ]; then
        local P="${QUEUE[$QUEUE_IDX]}"
        local DIR="$BASE_DIR/p${P}"
        mkdir -p "$DIR"
        log "Launching p=$P → $DIR"
        env $ENV_VARS pixi run python run_noise.py sweep \
            --data-path data/raw/creditcard.csv \
            --n-qubits 8 \
            --noise-levels "$P" \
            --noise-dir "$DIR" \
            --no-plots \
            > "$DIR/run.log" 2>&1 &
        log "p=$P launched (PID $!)"
        QUEUE_IDX=$((QUEUE_IDX + 1))
    fi
}

log "Watcher started. Monitoring p=0.001, p=0.005, p=0.01..."
log "Queue: ${QUEUE[*]}"

PREV_COUNT=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')

while true; do
    sleep 120  # check every 2 minutes

    CURR_COUNT=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')

    if [ "$CURR_COUNT" -lt "$PREV_COUNT" ]; then
        FINISHED=$((PREV_COUNT - CURR_COUNT))
        log "$FINISHED process(es) finished. Running: $CURR_COUNT"
        # Launch one queued level per finished process
        for ((i=0; i<FINISHED; i++)); do
            launch_next
        done
        PREV_COUNT=$CURR_COUNT
    fi

    # Check if everything is done
    CURR_COUNT=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')
    if [ "$CURR_COUNT" -eq 0 ] && [ "$QUEUE_IDX" -ge "${#QUEUE[@]}" ]; then
        log "All levels done. Merging..."
        break
    fi
done

# Merge all results + generate plot
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
