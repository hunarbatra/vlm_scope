#!/bin/bash
# Full analysis + ablation pipeline for DOCCI PaliGemma2 JumpReLU SAE
# Resumes from step 3 (steps 1-2 already done)
#
# Usage: bash run_full_pipeline.sh 2>&1 | tee /data1/vlm_scope_sae_docci/pipeline.log

set -e

export HF_HOME=/data1/vlm_scope_sae_docci/hf_cache
export HF_DATASETS_CACHE=/data1/vlm_scope_sae_docci/hf_datasets_cache
export HF_TOKEN=hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

cd "$(dirname "$0")"

echo "============================================================"
echo "Step 3: Visual Energy Ev (8 GPUs)"
echo "============================================================"
python3 local_analysis.py --step 3 --gpus 8

echo "============================================================"
echo "Step 4: Adapted Features (CPU)"
echo "============================================================"
python3 local_analysis.py --step 4

echo "============================================================"
echo "Step 5: Firing Frequencies (8 GPUs)"
echo "============================================================"
python3 local_analysis.py --step 5 --gpus 8

echo "============================================================"
echo "Step 6: Spatial Features - Fisher Test (CPU)"
echo "============================================================"
python3 local_analysis.py --step 6

echo "============================================================"
echo "Step 8 (pre-lexical): Intersection without lexical filter"
echo "============================================================"
python3 local_analysis.py --step 8

echo "============================================================"
echo "Step 7: Lexical Artifact Filtering (8 GPUs)"
echo "============================================================"
python3 local_analysis.py --step 7 --gpus 8

echo "============================================================"
echo "Step 8 (final): Intersection with lexical filter"
echo "============================================================"
python3 local_analysis.py --step 8

echo "============================================================"
echo "Ablation: VSR + VQA + Control"
echo "============================================================"
FEATURES_CSV=/data1/vlm_scope_sae_docci/analysis/final_features/final_spatial_visual_features.csv
if [ -f "$FEATURES_CSV" ]; then
    python3 local_ablation_vsr.py \
        --features-csv "$FEATURES_CSV" \
        --max-vqa 500 --max-vsr 2000 \
        --results-dir /data1/vlm_scope_sae_docci/analysis/ablation
else
    echo "[WARN] No final features CSV found at $FEATURES_CSV"
    echo "Skipping ablation."
fi

echo "============================================================"
echo "PIPELINE COMPLETE"
echo "Results: /data1/vlm_scope_sae_docci/analysis/"
echo "============================================================"
