#!/usr/bin/env bash
# run_noise_watcher.sh
# Universal noise sweep watcher.
#
# Can be run from scratch or mid-sweep. Runs noise levels in a granular
# VQC-first then QSVM queue, with at most MAX_PARALLEL jobs at a time.
# VQC and QSVM for the same level write to separate subdirs to avoid
# overwriting each other. Auto-merges and plots when everything is done.
#
# Usage:
#   bash run_noise_watcher.sh                        # full sweep, 2 parallel
#   bash run_noise_watcher.sh --parallel 3           # 3 parallel jobs
#   bash run_noise_watcher.sh --levels "0.0 0.001"   # custom levels

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_DIR="results/noise"
DATA_PATH="data/raw/creditcard.csv"
N_QUBITS=8
MAX_PARALLEL=2
NOISE_LEVELS=(0.0 0.001 0.005 0.01 0.02 0.05)
ENV_VARS="OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10"

# ── CLI ───────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --levels)   read -ra NOISE_LEVELS <<< "$2"; shift 2 ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── Build queue: VQC jobs first, then QSVM ───────────────────────────────────
# Each entry: "noise_level:phase:outdir"
QUEUE=()
for P in "${NOISE_LEVELS[@]}"; do
    QUEUE+=("${P}:vqc:${BASE_DIR}/p${P}/vqc")
done
for P in "${NOISE_LEVELS[@]}"; do
    QUEUE+=("${P}:qsvm:${BASE_DIR}/p${P}/qsvm")
done
QUEUE_IDX=0

launch_next() {
    if [ $QUEUE_IDX -ge ${#QUEUE[@]} ]; then return; fi

    local ENTRY="${QUEUE[$QUEUE_IDX]}"
    local P="${ENTRY%%:*}"; local REST="${ENTRY#*:}"
    local PHASE="${REST%%:*}"; local DIR="${REST#*:}"
    mkdir -p "$DIR"

    local FLAG=""
    [ "$PHASE" = "vqc" ]  && FLAG="--vqc-only"
    [ "$PHASE" = "qsvm" ] && FLAG="--qsvm-only"

    log "Launching p=$P ($PHASE) → $DIR"
    env $ENV_VARS pixi run python run_noise.py sweep \
        --data-path "$DATA_PATH" \
        --n-qubits "$N_QUBITS" \
        --noise-levels "$P" \
        --noise-dir "$DIR" \
        --no-plots \
        $FLAG \
        >> "$DIR/run.log" 2>&1 &
    log "  PID $!"
    QUEUE_IDX=$((QUEUE_IDX + 1))
}

log "Watcher started — universal mode."
log "Levels: ${NOISE_LEVELS[*]} | Max parallel: $MAX_PARALLEL"
log "Queue: ${#QUEUE[@]} jobs (VQC first, then QSVM)"

# ── Seed initial jobs ─────────────────────────────────────────────────────────
RUNNING=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')
while [ "$RUNNING" -lt "$MAX_PARALLEL" ] && [ "$QUEUE_IDX" -lt "${#QUEUE[@]}" ]; do
    launch_next
    RUNNING=$((RUNNING + 1))
done

PREV_COUNT=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')

# ── Main loop ─────────────────────────────────────────────────────────────────
while true; do
    sleep 120

    CURR_COUNT=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')

    if [ "$CURR_COUNT" -lt "$PREV_COUNT" ]; then
        FREED=$((PREV_COUNT - CURR_COUNT))
        log "$FREED slot(s) freed. Running: $CURR_COUNT"
        for ((i=0; i<FREED; i++)); do launch_next; done
        PREV_COUNT=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')
    else
        PREV_COUNT=$CURR_COUNT
    fi

    CURR_COUNT=$(ps aux | grep "run_noise.py sweep" | grep -v grep | wc -l | tr -d ' ')
    if [ "$CURR_COUNT" -eq 0 ] && [ "$QUEUE_IDX" -ge "${#QUEUE[@]}" ]; then
        log "All jobs done. Merging..."
        break
    fi
done

# ── Merge all results + plot ──────────────────────────────────────────────────
DIRS=()
for P in "${NOISE_LEVELS[@]}"; do
    [ -d "$BASE_DIR/p${P}/vqc"  ] && DIRS+=("$BASE_DIR/p${P}/vqc")
    [ -d "$BASE_DIR/p${P}/qsvm" ] && DIRS+=("$BASE_DIR/p${P}/qsvm")
done

pixi run python run_noise.py merge \
    --noise-dirs "${DIRS[@]}" \
    --out-dir "$BASE_DIR"

log "============================================================"
log "SWEEP COMPLETE"
log "Results: $BASE_DIR/noise_results.json"
log "Plot:    $BASE_DIR/noise_vs_metric.png"
log "============================================================"
