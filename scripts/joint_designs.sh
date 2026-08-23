#!/usr/bin/env bash
# Joint interventional designs on Llama2-7B at 25% (8 of 32 blocks): adjacent pairs, and random
# multi-block removals of about 2 and 8 blocks per intervention (remove_frac x 32 blocks).
# Run from the repository root: bash scripts/joint_designs.sh
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
NUM_BLOCKS=${NUM_BLOCKS:-32}
REMOVE=${REMOVE:-8}
COMMON="--model_name $MODEL --num_blocks $NUM_BLOCKS --num_remove_blocks $REMOVE --dataset wikitext2 \
  --nsamples 128 --loss_batch_size 128 --eval_ppl True --eval_zeroshot True --eval_limit 500 --eval_batch_size 32"

python sememe.py $COMMON --prune_method adjacent
for FRAC in 0.0625 0.25; do
  python sememe.py $COMMON --prune_method shapley --shapley_samples 256 --shapley_remove_frac "$FRAC"
done
