#!/bin/bash

echo "Starting SAE feature firing frequency tracking..."

CUDA_VISIBLE_DEVICES=7 python finetune/vqa/track_firing_vqa.py \
    --from-layer 0 \
    --to-layer 32 \
    --start-sample 0 \
    --end-sample 50000 \
    --caching-batch-size 8 \
    --output-dir "results/feature_firing_analysis/vqa_text_only" \
    --sae-checkpoint-dir "/scratch/local/ssd/lachin/checkpoints_50k/" \
    --methods text-only


python vqa/track_firing_vsr.py \
    --from-layer 0 \
    --to-layer 32 \
    --start-sample 0 \
    --end-sample 3800 \
    --caching-batch-size 32 \
    --output-dir "results/feature_firing_analysis/vsr" \
    --sae-checkpoint-dir "/scratch/local/ssd/lachin/checkpoints_50k/" \
    --methods pretrained text-only


echo "Feature firing analysis complete!" 