#!/usr/bin/env bash
# Sememe-I-Skill: prune Llama2-7B toward each downstream task (7 of 32 blocks), then evaluate on that task.
# Run from the repository root: bash scripts/task_aware.sh
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
NUM_BLOCKS=${NUM_BLOCKS:-32}
REMOVE=${REMOVE:-7}

for TASK in arc_easy arc_challenge winogrande hellaswag piqa; do
  python sememe.py --model_name "$MODEL" --num_blocks "$NUM_BLOCKS" --num_remove_blocks "$REMOVE" \
    --dataset "$TASK" --nsamples 128 --prune_method ols --eval_ppl False --eval_zeroshot True
done
