#!/bin/bash
# Auto-launcher: waits for step 8 to complete, then launches ablation, then CAA.
# Run as: nohup bash auto_launch_mathverse.sh > /tmp/mathverse_logs/auto_launch.log 2>&1 &

FINAL_FEATURES="/data1/vlm_scope_sae_mix448_textonly/analysis_mathverse/final_features/final_math_features.csv"
ABL_DIR="/data1/vlm_scope_sae_mix448_textonly/analysis_mathverse/ablation_mathverse"
ABL_SUMMARY="${ABL_DIR}/ablation_summary.json"
SCRIPTS_DIR="/data1/vlm_scope_sae_mix448_textonly/scripts/multimodal_mathverse"
PIPELINE_LOG="/tmp/mathverse_logs/pipeline_lexical.log"
ABL_LOG="/tmp/mathverse_logs/ablation.log"
CAA_BASE_LOG="/tmp/mathverse_logs/caa_baseline.log"
CAA_MMDIFF_LOG="/tmp/mathverse_logs/caa_mmdiff.log"

mkdir -p /tmp/mathverse_logs "${ABL_DIR}"

echo "[$(date)] Auto-launcher started. Watching for ${FINAL_FEATURES}" | tee -a /tmp/mathverse_logs/auto_launch.log

# ---- Phase 1: Wait for pipeline step 8 ----
while true; do
    if [ -f "${FINAL_FEATURES}" ]; then
        N=$(tail -n +2 "${FINAL_FEATURES}" | wc -l)
        echo "[$(date)] Step 8 done. ${N} final features." | tee -a /tmp/mathverse_logs/auto_launch.log
        break
    fi
    # Check if pipeline process died
    if ! pgrep -f "local_analysis_mathverse.py" > /dev/null 2>&1; then
        echo "[$(date)] Pipeline process not found. Checking if DONE..." | tee -a /tmp/mathverse_logs/auto_launch.log
        if grep -q "DONE" "${PIPELINE_LOG}" 2>/dev/null; then
            echo "[$(date)] Pipeline DONE (in log) but no final_features.csv. Check manually." | tee -a /tmp/mathverse_logs/auto_launch.log
        fi
        break
    fi
    sleep 60
done

if [ ! -f "${FINAL_FEATURES}" ]; then
    echo "[$(date)] ERROR: No final features CSV. Exiting auto-launcher." | tee -a /tmp/mathverse_logs/auto_launch.log
    exit 1
fi

N_FEATS=$(tail -n +2 "${FINAL_FEATURES}" | wc -l)
echo "[$(date)] ${N_FEATS} final features. Launching ablation on all 8 GPUs..." | tee -a /tmp/mathverse_logs/auto_launch.log

# ---- Phase 2: Ablation ----
if [ "${N_FEATS}" -eq 0 ]; then
    echo "[$(date)] WARN: Zero intersection features. Skipping ablation/CAA." | tee -a /tmp/mathverse_logs/auto_launch.log
    exit 0
fi

nohup python3 -u "${SCRIPTS_DIR}/ablation_mathverse.py" \
    --features "${FINAL_FEATURES}" \
    --gpus 0 1 2 3 4 5 6 7 \
    > "${ABL_LOG}" 2>&1 &
ABL_PID=$!
echo "[$(date)] Ablation PID: ${ABL_PID}" | tee -a /tmp/mathverse_logs/auto_launch.log

# Wait for ablation
while kill -0 $ABL_PID 2>/dev/null; do
    N_DONE=$(ls "${ABL_DIR}"/ablation_L*_F*.json 2>/dev/null | wc -l)
    echo "[$(date)] Ablation: ${N_DONE} features done..." | tee -a /tmp/mathverse_logs/auto_launch.log
    sleep 120
done
echo "[$(date)] Ablation process finished." | tee -a /tmp/mathverse_logs/auto_launch.log

# Summarize ablation
python3 -u "${SCRIPTS_DIR}/ablation_mathverse.py" \
    --out-dir "${ABL_DIR}" \
    --summarize \
    >> "${ABL_LOG}" 2>&1
echo "[$(date)] Ablation summary written." | tee -a /tmp/mathverse_logs/auto_launch.log

# ---- Phase 3: CAA steering ----
if [ ! -f "${ABL_SUMMARY}" ]; then
    echo "[$(date)] No ablation_summary.json. Cannot select CAA features." | tee -a /tmp/mathverse_logs/auto_launch.log
    exit 1
fi

echo "[$(date)] Launching baseline CAA (L13 middle, GPU 0)..." | tee -a /tmp/mathverse_logs/auto_launch.log
CUDA_VISIBLE_DEVICES=0 nohup python3 -u "${SCRIPTS_DIR}/caa_mathverse.py" \
    --mode baseline \
    --gpu 0 \
    > "${CAA_BASE_LOG}" 2>&1 &
CAA_BASE_PID=$!

echo "[$(date)] Launching MMDiff CAA (top-5 features, GPU 1)..." | tee -a /tmp/mathverse_logs/auto_launch.log
CUDA_VISIBLE_DEVICES=1 nohup python3 -u "${SCRIPTS_DIR}/caa_mathverse.py" \
    --mode mmdiff \
    --ablation-dir "${ABL_DIR}" \
    --top-k 5 \
    --gpu 1 \
    > "${CAA_MMDIFF_LOG}" 2>&1 &
CAA_MMDIFF_PID=$!

echo "[$(date)] CAA baseline PID: ${CAA_BASE_PID}, MMDiff PID: ${CAA_MMDIFF_PID}" | tee -a /tmp/mathverse_logs/auto_launch.log

wait $CAA_BASE_PID
echo "[$(date)] CAA baseline done." | tee -a /tmp/mathverse_logs/auto_launch.log
wait $CAA_MMDIFF_PID
echo "[$(date)] CAA MMDiff done." | tee -a /tmp/mathverse_logs/auto_launch.log

echo "[$(date)] === ALL DONE ===" | tee -a /tmp/mathverse_logs/auto_launch.log
echo "" | tee -a /tmp/mathverse_logs/auto_launch.log
echo "[BASELINE CAA RESULTS]" | tee -a /tmp/mathverse_logs/auto_launch.log
cat "${SCRIPTS_DIR%/*}/mathverse"/../analysis_mathverse/caa_baseline_mathverse/results.json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Baseline math: {d[\"baseline_math_acc\"]:.2f}%')
for r in sorted(d[\"results\"], key=lambda x: -x[\"delta_math\"])[:5]:
    print(f'  alpha={r[\"alpha\"]}: math={r[\"math_acc\"]:.2f}% (Delta={r[\"delta_math\"]:+.2f}pp)')
" 2>/dev/null | tee -a /tmp/mathverse_logs/auto_launch.log

echo "[MMDIFF CAA RESULTS]" | tee -a /tmp/mathverse_logs/auto_launch.log
cat /data1/vlm_scope_sae_mix448_textonly/analysis_mathverse/caa_mmdiff_mathverse/results.json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Baseline math: {d[\"baseline_math_acc\"]:.2f}%')
for r in sorted(d[\"results\"], key=lambda x: -x[\"delta_math\"])[:10]:
    print(f'  L{r[\"layer\"]}_F{r[\"feature\"]} a={r[\"alpha\"]} b={r[\"beta\"]}: math={r[\"math_acc\"]:.2f}% (Delta={r[\"delta_math\"]:+.2f}pp)')
" 2>/dev/null | tee -a /tmp/mathverse_logs/auto_launch.log
