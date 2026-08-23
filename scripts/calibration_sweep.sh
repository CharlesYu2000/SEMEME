#!/usr/bin/env bash
# Sememe-I across calibration sizes, pruning Llama3.1-8B to 25% (8 of 32 blocks).
# Run from the repository root: bash scripts/calibration_sweep.sh
MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
NUM_BLOCKS=${NUM_BLOCKS:-32}
REMOVE=${REMOVE:-8}

for N in 8 16 32 64 128; do
  python sememe.py --model_name "$MODEL" --num_blocks "$NUM_BLOCKS" --num_remove_blocks "$REMOVE" \
    --dataset wikitext2 --nsamples "$N" --loss_batch_size "$N" --prune_method ols --eval_ppl True
done
