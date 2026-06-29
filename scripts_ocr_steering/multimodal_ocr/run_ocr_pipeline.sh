#!/usr/bin/env bash
# OCR-Bench pipeline orchestrator (local machine, post-reboot).
# GPU 1 is in use by another user (syang217/Qwen3-VL). We use GPUs 0, 2-7 → 7 GPUs.
# Runs: firing → fisher → lexical → intersect → ablation → baseline CAA → MMDiff-CAA boost.
set -u

ROOT=/data1/vlm_scope_sae_mix448_textonly
SCRIPTS=$ROOT/scripts/multimodal_ocr
ANA=$ROOT/analysis_ocr
LOG=/tmp/ocr_pipeline.log

mkdir -p "$ANA"

# Skip GPU 1 (owned by another user). Remaining 7 GPUs are 0, 2, 3, 4, 5, 6, 7.
# With CUDA_VISIBLE_DEVICES set, scripts see them numbered 0..6.
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7

say() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

say "========== OCR pipeline starting on $(hostname) =========="
say "Using physical GPUs 0,2,3,4,5,6,7 (GPU 1 reserved for another user)."
nvidia-smi --query-gpu=index,memory.free --format=csv | tee -a "$LOG"

cd "$SCRIPTS"

# --- Step 5 (firing on VQA + OCR-Bench) ---
N_VQA=$(ls "$ANA/firing_vqa"/firing_vqa_layer_*.json 2>/dev/null | wc -l)
N_OCR=$(ls "$ANA/firing_ocr"/firing_ocr_layer_*.json 2>/dev/null | wc -l)
if [ "$N_VQA" -eq 26 ] && [ "$N_OCR" -eq 26 ]; then
    say "Step 5 already complete (26/26 VQA, 26/26 OCR)."
else
    say "Step 5: firing (VQA 50K + OCR-Bench 1000 samples, 7 GPUs)..."
    python3 -u local_analysis_textonly_ocr.py --step 5 --gpus 7 2>&1 | tee -a "$LOG"
fi

# --- Steps 6, 7, 8 ---
if [ -f "$ANA/final_features/final_ocr_features.csv" ]; then
    say "Steps 6-8 already complete."
else
    say "Steps 6-8: Fisher + lexical + intersect..."
    python3 -u local_analysis_textonly_ocr.py --step 6 7 8 --gpus 7 2>&1 | tee -a "$LOG"
fi

FINAL=$ANA/final_features/final_ocr_features.csv
if [ ! -f "$FINAL" ]; then
    say "ERROR: $FINAL not produced. Exiting."
    exit 2
fi
N_FINAL=$(tail -n +2 "$FINAL" | wc -l)
say "Step 8 produced $N_FINAL final OCR features."

# --- Ablation (7 GPUs) ---
if [ -f "$ANA/ablation_ocr/ablation_summary.csv" ]; then
    say "Ablation already complete."
else
    say "Running ablation on $N_FINAL features across 7 GPUs..."
    python3 -u ablation_per_relation_ocr.py \
        --features "$FINAL" --gpus 0 1 2 3 4 5 6 2>&1 | tee -a "$LOG"
fi

# --- Baseline CAA (1 GPU) ---
BASE_RESULTS="$ANA/caa_baseline_ocr/results.json"
if [ -f "$BASE_RESULTS" ] && grep -q 'alpha_5' "$BASE_RESULTS"; then
    say "Baseline CAA already complete."
else
    say "Baseline CAA (L13 middle, α sweep)..."
    CUDA_VISIBLE_DEVICES=0 python3 -u caa_baseline_ocr.py 2>&1 | tee -a "$LOG"
fi

# --- MMDiff-CAA boost (1 GPU) ---
MMDIFF_RESULTS="$ANA/caa_mmdiff_boost_ocr/results.json"
if [ -f "$MMDIFF_RESULTS" ]; then
    say "MMDiff-CAA boost already complete."
else
    say "MMDiff-CAA boost (top-10 OCR features, α×β sweep)..."
    CUDA_VISIBLE_DEVICES=0 python3 -u caa_mmdiff_boost_ocr.py \
        --ablation-csv "$ANA/ablation_ocr/ablation_summary.csv" \
        --top-k 10 2>&1 | tee -a "$LOG"
fi

say "========== OCR pipeline complete =========="
