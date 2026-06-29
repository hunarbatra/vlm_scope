#!/bin/bash
# Watches for 25pct ablation_summary.csv, then:
# 1. Downloads pt-448 if needed
# 2. Launches pt448_vsr_ablation.py on 8 GPUs

SUMMARY_CSV="/data1/vlm_scope_sae_mix448_textonly/analysis_25pct/ablation_per_relation_full/ablation_summary.csv"
MIX_CSV="/data1/vlm_scope_sae_mix448_textonly/analysis/ablation_per_relation_full/ablation_summary.csv"
OUT_DIR="/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_vsr_ablation"
GEMMA_ABL_DIR="/data1/vlm_scope_sae_mix448_textonly/analysis/gemma_base_vsr_ablation"
SCRIPT_DIR="/data1/vlm_scope_sae_mix448_textonly/scripts"
LOG="/data1/vlm_scope_sae_mix448_textonly/logs/pt448_vsr_ablation.log"
PYTHONPATH_EXTRA="/data1/hbatra/site-packages"

echo "[watcher] Waiting for 25pct ablation to complete..."
while [ ! -f "$SUMMARY_CSV" ]; do
    sleep 120
done
echo "[watcher] 25pct ablation_summary.csv found. Starting pt-448 ablation pipeline."

mkdir -p "$OUT_DIR" "$(dirname $LOG)"

# Step 1: download pt-448 if not cached
if [ ! -d "/data1/hf_cache/hub/models--google--paligemma2-3b-pt-448" ]; then
    echo "[watcher] Downloading paligemma2-3b-pt-448..."
    PYTHONPATH="$PYTHONPATH_EXTRA" python3 "$SCRIPT_DIR/pt448_vsr_ablation.py" --download-only \
        >> "$LOG" 2>&1
    echo "[watcher] Download done."
else
    echo "[watcher] pt-448 already cached, skipping download."
fi

# Step 2: run ablation
echo "[watcher] Launching pt-448 ablation (8 GPUs)..."
PYTHONPATH="$PYTHONPATH_EXTRA" python3 -u "$SCRIPT_DIR/pt448_vsr_ablation.py" \
    --ablation-csv "$MIX_CSV" \
    --out-dir "$OUT_DIR" \
    --gemma-abl-dir "$GEMMA_ABL_DIR" \
    --n-gpus 8 \
    --top-n 100 \
    >> "$LOG" 2>&1

echo "[watcher] pt-448 ablation complete. Log: $LOG"
