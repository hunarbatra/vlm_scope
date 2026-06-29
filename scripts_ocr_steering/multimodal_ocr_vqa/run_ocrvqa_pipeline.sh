#!/usr/bin/env bash
# OCR-VQA full pipeline orchestrator (firing → Fisher → lexical → intersect → ablation).
# Skips GPU 1 (occupied by another user). Uses GPUs 0, 2, 3, 4, 5, 6, 7.
set -u
set -o pipefail

ROOT=/data1/vlm_scope_sae_mix448_textonly
SCRIPTS=$ROOT/scripts/multimodal_ocr_vqa
ANA=$ROOT/analysis_ocrvqa
LOG=/tmp/ocrvqa_pipeline.log

mkdir -p "$ANA"

# Skip GPUs 1 + 4 (occupied by syang217). Remaining = 0,2,3,5,6,7 → script sees 0..5.
export CUDA_VISIBLE_DEVICES=0,2,3,5,6,7

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "========== OCR-VQA pipeline starting on $(hostname) =========="
say "Using physical GPUs 0,2,3,5,6,7 (6 GPUs; 1+4 occupied)."
nvidia-smi --query-gpu=index,memory.free --format=csv | tee -a "$LOG"

cd "$SCRIPTS"

# Build OCR-VQA indices (cached)
if [ ! -f "$ANA/ocrvqa_indices_train.json" ] || [ ! -f "$ANA/ocrvqa_indices_test.json" ]; then
  say "Building OCR-VQA indices (10K train + 10K test)..."
  python3 -u build_ocrvqa_indices.py 2>&1 | tee -a "$LOG"
fi

# --- Step 5: firing (VQA 50K + OCR-VQA 10K) ---
N_VQA=$(ls "$ANA/firing_vqa_pertoken"/firing_vqa_layer_*.json 2>/dev/null | wc -l)
N_OCR=$(ls "$ANA/firing_ocrvqa_pertoken"/firing_ocrvqa_layer_*.json 2>/dev/null | wc -l)
if [ "$N_VQA" -eq 26 ] && [ "$N_OCR" -eq 26 ]; then
  say "Step 5 already complete."
else
  say "Step 5: firing (VQA 50K + OCR-VQA 10K, 7 GPUs)..."
  python3 -u local_analysis_textonly_ocrvqa.py --step 5 --gpus 6 2>&1 | tee -a "$LOG"
fi

# --- Steps 6, 7, 8 ---
if [ -f "$ANA/final_features/final_ocrvqa_features.csv" ]; then
  say "Steps 6-8 already complete."
else
  say "Steps 6-8: Fisher + lexical + intersect..."
  python3 -u local_analysis_textonly_ocrvqa.py --step 6 7 8 --gpus 6 2>&1 | tee -a "$LOG"
fi

FINAL=$ANA/final_features/final_ocrvqa_features.csv
if [ ! -f "$FINAL" ]; then
  say "ERROR: $FINAL not produced. Exiting."
  exit 2
fi
N_FINAL=$(tail -n +2 "$FINAL" | wc -l)
say "Step 8 produced $N_FINAL final OCR-VQA features."

# --- Ablation (7 GPUs) ---
if [ -f "$ANA/ablation_ocrvqa/ablation_summary.csv" ]; then
  say "Ablation already complete."
else
  say "Running ablation on $N_FINAL features across 7 GPUs..."
  python3 -u ablation_per_relation_ocrvqa.py \
      --features "$FINAL" --gpus 0 1 2 3 4 5 2>&1 | tee -a "$LOG"
fi

say "========== OCR-VQA pipeline (Steps 5-8 + ablation) complete =========="
say "Next: select top 7-10 features by ∆OCRVQA (gated by ∆Ctrl, ∆VQA), then steering."
