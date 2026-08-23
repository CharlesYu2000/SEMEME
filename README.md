# Sememe: Causal Modeling for Task-Aware Depth Pruning of Large Language Models

COLM 2026. [[Paper]](https://openreview.net/pdf?id=UTMigobeqk)

Sememe is a depth-pruning framework which removes entire blocks/layers from transformers. Sememe runs only using forward passes and a lightweight CPU estimation step, requiring no gradients. Sememe operates by considering each block removal as an intervention of a structural causal model, and estimates importance as path coefficients based on the calibration losses per example.

The code in this repo builds upon [SLEB](https://github.com/jiwonsong-dev/SLEB) and the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Setup

```bash
conda env create -f environment.yml
conda activate sememe
pip install -e .

# Patch the lm-evaluation-harness
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git apply ../harness/sememe_evaluator.patch
cp -r ../harness/tasks/. lm_eval/tasks/
pip install -e .
cd ..
```

Note: the `lm-evaluation-harness` patch allows us to turn off or remove blocks using our code. We also add some tasks which allow us to calibrate skill-specific models. 


## Usage

For gated models (e.g. Llama family models), make sure to set the `HF_TOKEN` env variable, or write the token to `.hf_token` in this root directory.

Prune Llama3.1-8B to 25% sparsity (remove 8 of 32 blocks) with Sememe-I and 32 calibration examples, then evaluate on WikiText-2/C4 perplexity and zero-shot downstream task accuracy:

```bash
python sememe.py --model_name meta-llama/Llama-3.1-8B --num_blocks 32 --num_remove_blocks 8 \
    --dataset wikitext2 --nsamples 32 --loss_batch_size 32 --prune_method ols \
    --eval_ppl True --eval_zeroshot True
```

The block removal order is printed as well as appended to `sememe_results/sememe_results.txt`. The following are main options for the script:

| Option | Values |
|---|---|
| `--prune_method` | `ols` (Sememe-I), `dowhy` (Sememe-C), `adjacent` (single blocks and adjacent pairs), `shapley` (random multi-block removals; each block is removed with probability `--shapley_remove_frac`) |
| `--dataset` | `wikitext2`, `c4`, or a task for Sememe-I-Skill: `arc_easy`, `arc_challenge`, `winogrande`, `hellaswag`, `piqa` |
| `--nsamples` | calibration set size N |
| `--calib_mode` | `windows` (N random 2048-token windows, default) or `concat` (N documents concatenated and split into windows) |
| `--loss_batch_size` | forward batch size for the calibration losses (`0` = all N at once) |

Sememe-I-Skill on ARC-Easy (Llama2-7B, 7 of 32 blocks):

```bash
python sememe.py --model_name meta-llama/Llama-2-7b-hf --num_blocks 32 --num_remove_blocks 7 \
    --dataset arc_easy --nsamples 128 --prune_method ols --eval_ppl False --eval_zeroshot True
```

You can also directly evaluate a provided removal order (add the `--incremental` flag to evaluate incremntally after each removed block as opposed to only once after all removals, use `--device_map_auto True` for multi-GPU models):

```bash
python evaluate_pruned.py --model_name meta-llama/Llama-3.1-8B \
    --removal_order 10,25,26,11,12,23,19,8 --output_path results/llama3_1_8b.jsonl
```

### Adding a task for Sememe-I-Skill

1. Add `<task>_prune.yaml` to `lm-evaluation-harness/lm_eval/tasks/<task>/`. This task can simply be created as a copy of the task's original YAML whose splits point to training or validation data.
2. Add a loss function to `sememeutils/loss_utils.py` returning the negative accuracy on `<task>_prune` (e.g., see `get_piqa_loss_eval_type()`), and register it in the `TASK_LOSS_FNS` of `sememe.py`.

## Citation

If you found this paper to be relevant or this code to be useful, please consider citing us:

```bibtex
@inproceedings{
yu2026sememe,
title={Sememe: Causal Modeling for Task-Aware Depth Pruning of Large Language Models},
author={Charles Yu and Heng Ji},
booktitle={Third Conference on Language Modeling},
year={2026},
url={https://openreview.net/forum?id=UTMigobeqk}
}
```

