#!/bin/bash
# Poll until steps 5+6 finish (spatial_pertoken CSV appears), then run 7+8
LOG=/data1/vlm_scope_sae_mix448_textonly/run_firing_pertoken.log
SPATIAL_CSV=/data1/vlm_scope_sae_mix448_textonly/analysis/spatial_pertoken/spatial_features_pertoken.csv
CHAIN_LOG=/data1/vlm_scope_sae_mix448_textonly/run_lexical_intersection.log

echo "[chain] Waiting for steps 5+6 to complete..."
echo "[chain] Watching for: $SPATIAL_CSV"

while true; do
    # Check if spatial CSV exists (step 6 output)
    if [ -f "$SPATIAL_CSV" ]; then
        echo "[chain] Step 6 complete! Spatial CSV found."
        echo "[chain] Launching steps 7+8..."
        cd /data1/vlm_scope_sae_mix448_textonly/scripts
        python3 -u local_analysis_textonly.py --step 7 8 > "$CHAIN_LOG" 2>&1
        echo "[chain] Steps 7+8 finished. Exit code: $?"
        exit 0
    fi
    
    # Also check if the main process died
    if ! ps -p 664518 > /dev/null 2>&1; then
        # Process ended — check if spatial CSV appeared
        if [ -f "$SPATIAL_CSV" ]; then
            echo "[chain] Steps 5+6 process ended, spatial CSV found. Launching 7+8..."
            cd /data1/vlm_scope_sae_mix448_textonly/scripts
            python3 -u local_analysis_textonly.py --step 7 8 > "$CHAIN_LOG" 2>&1
            echo "[chain] Steps 7+8 finished. Exit code: $?"
            exit 0
        else
            echo "[chain] ERROR: Process 664518 died but no spatial CSV found!"
            echo "[chain] Check $LOG for errors."
            exit 1
        fi
    fi
    
    sleep 60
done
