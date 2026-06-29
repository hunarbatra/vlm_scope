#!/bin/bash
# Auto-launcher: monitors pipeline completion, then runs ablation on new features
# Ablation uses full 3-component projection (attn + mlp + residual) across all 26 layers
# Usage: nohup bash auto_ablation_launcher.sh &

PIPELINE_LOG="/data1/vlm_scope_sae_mix448/logs/steps_5678_v8_8gpu.log"
FEATURES_CSV="/data1/vlm_scope_sae_mix448/analysis/final_features/final_spatial_visual_features.csv"
ABLATION_LOG="/data1/vlm_scope_sae_mix448/logs/ablation_new_features.log"
ABLATION_OUT="/data1/vlm_scope_sae_mix448/analysis/ablation_v2"
SCRIPT_DIR="/home/hbatra/vlm_scope_backup/vlm_scope/finetune/paligemma2"

echo "[$(date)] Auto-ablation launcher started."
echo "[$(date)] Monitoring: $PIPELINE_LOG"
echo "[$(date)] Will ablate features from: $FEATURES_CSV"
echo "[$(date)] Ablation output to: $ABLATION_OUT"

while true; do
    # Check if pipeline finished (step 8 intersection writes DONE)
    if grep -q "\[DONE\] Analysis complete" "$PIPELINE_LOG" 2>/dev/null; then
        echo "[$(date)] Pipeline completed! (found DONE in log)"
        break
    fi

    # Also check the v7 log in case it completes there
    if grep -q "\[DONE\] Analysis complete" "/data1/vlm_scope_sae_mix448/logs/steps_5678_v7.log" 2>/dev/null; then
        echo "[$(date)] Pipeline completed! (found DONE in v7 log)"
        break
    fi

    # Check if features CSV already exists (step 8 output)
    if [ -f "$FEATURES_CSV" ]; then
        # Verify it's recent (modified in last 2 hours)
        if find "$FEATURES_CSV" -mmin -120 | grep -q .; then
            echo "[$(date)] Features CSV found and recent, proceeding."
            break
        fi
    fi

    # Check if pipeline process died
    if ! pgrep -f "local_analysis.py.*--step 5 6 7 8" > /dev/null 2>&1; then
        echo "[$(date)] WARNING: Pipeline process not found!"
        if [ -f "$FEATURES_CSV" ]; then
            echo "[$(date)] Features CSV exists, proceeding with ablation."
            break
        else
            echo "[$(date)] Pipeline died without producing features. Waiting 5 min and rechecking..."
            sleep 300
            if ! pgrep -f "local_analysis.py" > /dev/null 2>&1; then
                if [ ! -f "$FEATURES_CSV" ]; then
                    echo "[$(date)] Pipeline truly dead, no features. Exiting."
                    exit 1
                fi
            fi
        fi
    fi

    sleep 120  # Check every 2 minutes
done

sleep 10  # Let files flush

# Verify features CSV
if [ ! -f "$FEATURES_CSV" ]; then
    echo "[$(date)] ERROR: No features CSV at $FEATURES_CSV"
    exit 1
fi

N_FEATURES=$(tail -n +2 "$FEATURES_CSV" | wc -l)
echo "[$(date)] Found $N_FEATURES new features in $FEATURES_CSV"
echo "[$(date)] Feature file: $(head -3 "$FEATURES_CSV")"

# Launch ablation on all 8 GPUs
# Script uses corrected 3-component ablation:
#   - self_attn output projection
#   - MLP output projection
#   - Layer residual stream projection
# Applied across all 26 transformer layers on text tokens
echo "[$(date)] Launching ablation on $N_FEATURES features across 8 GPUs..."
cd "$SCRIPT_DIR"
python3 -u local_ablation.py \
    --features "$FEATURES_CSV" \
    --gpus 0 1 2 3 4 5 6 7 \
    --out-dir "$ABLATION_OUT" \
    > "$ABLATION_LOG" 2>&1

echo "[$(date)] Ablation complete! Results in $ABLATION_LOG"
echo "[$(date)] Summary CSV at $ABLATION_OUT/ablation_summary.csv"
