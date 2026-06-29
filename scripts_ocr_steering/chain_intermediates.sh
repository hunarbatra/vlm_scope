#!/bin/bash
# Sequential chain: 25pct -> 50pct -> 75pct
# Each: wait for Steps 1-8 (running), then launch per-relation ablation, then move on
set -u
cd /data1/vlm_scope_sae_mix448_textonly

CKPT_DIR_BASE=/data1/vlm_scope_sae_mix448_textonly
LOG_DIR=$CKPT_DIR_BASE/logs

run_one() {
    local PCT=$1
    local CKPT=$CKPT_DIR_BASE/checkpoint_${PCT}pct
    local ANA=$CKPT_DIR_BASE/analysis_${PCT}pct
    local FINAL_CSV=$ANA/final_features/final_spatial_visual_features.csv
    local ABL_CSV=$ANA/ablation_per_relation_full/ablation_summary.csv
    local STEPS_LOG=$LOG_DIR/run_${PCT}pct_full.log
    local ABL_LOG=$LOG_DIR/run_${PCT}pct_ablation.log

    echo "[chain] === Stage: ${PCT}pct ==="

    # If Steps 1-8 not yet running and final CSV missing, launch
    if [ ! -f "$FINAL_CSV" ] && ! pgrep -f "local_analysis_textonly.*${PCT}pct" > /dev/null; then
        echo "[chain] Launching Steps 1-8 for ${PCT}pct"
        mkdir -p "$ANA"
        nohup python3 -u scripts/local_analysis_textonly.py \
            --checkpoint-dir "$CKPT" --analysis-dir "$ANA" \
            > "$STEPS_LOG" 2>&1 &
    fi

    # Wait for final_features CSV (Step 8 output)
    echo "[chain] Waiting for $FINAL_CSV"
    while [ ! -f "$FINAL_CSV" ]; do
        sleep 120
        if ! pgrep -f "local_analysis_textonly.*${PCT}pct" > /dev/null && [ ! -f "$FINAL_CSV" ]; then
            echo "[chain] ERROR: Steps 1-8 process for ${PCT}pct died and no final CSV. Aborting."
            tail -40 "$STEPS_LOG"
            return 1
        fi
    done
    echo "[chain] Steps 1-8 done for ${PCT}pct."

    # Launch per-relation ablation
    if [ ! -f "$ABL_CSV" ]; then
        echo "[chain] Launching per-relation ablation for ${PCT}pct"
        VLMSCOPE_CKPT_DIR="$CKPT" VLMSCOPE_ANALYSIS_DIR="$ANA" \
        nohup python3 -u scripts/ablation_per_relation_textonly_local.py \
            --checkpoint-dir "$CKPT" --analysis-dir "$ANA" \
            > "$ABL_LOG" 2>&1 &
    fi
    while [ ! -f "$ABL_CSV" ]; do
        sleep 300
        if ! pgrep -f "ablation_per_relation_textonly_local.*${PCT}pct" > /dev/null \
             && ! pgrep -f "VLMSCOPE_CKPT_DIR=.*${PCT}pct" > /dev/null \
             && [ ! -f "$ABL_CSV" ]; then
            echo "[chain] ERROR: Ablation died for ${PCT}pct. Aborting."
            tail -40 "$ABL_LOG"
            return 1
        fi
    done
    echo "[chain] Ablation done for ${PCT}pct: $ABL_CSV"
}

run_one 25 && run_one 50 && run_one 75 && echo "[chain] ALL DONE"
