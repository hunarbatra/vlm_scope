python orchestrate.py \
   --from-layer 0 \
   --to-layer 32 \
   --n-training-samples 50000 \
   --caching-batch-size 160 \
   --caching-chunk-size 13000 \
   --training-batch-size 128 \
   --val-frac 0.02 \
   --val-every-n-batches 10 \
   --methods pretrained random text-only image-only
   #--resume-from-sample ... \
   #--resume-wandb-run-id ... \