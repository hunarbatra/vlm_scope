#!/bin/bash
# Wait for extraction to finish, then run merge + auto-interp
set -e

EXTRACT_LOG="/data1/vlm_scope_sae_mix448_textonly/run_extract_multi.log"
SCRIPTS_DIR="/data1/vlm_scope_sae_mix448_textonly/scripts"
FEATURES_CSV="/data1/vlm_scope_sae_mix448_textonly/analysis/final_features/final_spatial_visual_features.csv"
SAMPLES_DIR="/data1/vlm_scope_sae_mix448_textonly/analysis/multidataset_feature_samples"
COMMON_SUMMARY="/data1/vlm_scope_sae_mix448_textonly/analysis/dataset_all_features.json"
API_KEY="sk-REPLACE_ME_WITH_YOUR_OPENAI_KEY_OR_USE_ENV_VAR"

echo "[$(date)] Waiting for extraction to complete..."

# Wait for "Extraction complete" in log
while true; do
    if grep -q "Extraction complete" "$EXTRACT_LOG" 2>/dev/null; then
        echo "[$(date)] Extraction complete detected!"
        break
    fi
    # Also check if all 8 GPUs report done
    n_done=$(grep -c "All layers done" "$EXTRACT_LOG" 2>/dev/null || echo 0)
    if [ "$n_done" -ge 8 ]; then
        echo "[$(date)] All 8 GPUs done!"
        break
    fi
    sleep 30
done

echo ""
echo "=========================================="
echo "[$(date)] Step 1: Merge multi-dataset samples"
echo "=========================================="
cd "$SCRIPTS_DIR"
python3 -u merge_multidataset_samples.py \
    --samples-dir "$SAMPLES_DIR" \
    --features "$FEATURES_CSV" \
    --output "$COMMON_SUMMARY" \
    --top-k 10

echo ""
echo "=========================================="
echo "[$(date)] Step 2: Auto-interp (multi-dataset)"
echo "=========================================="
python3 -u auto_interp_multidataset.py \
    --common-summary "$COMMON_SUMMARY" \
    --api-key "$API_KEY" \
    --samples-per-feature 5 \
    --delay-s 0.3

echo ""
echo "[$(date)] All steps complete!"
