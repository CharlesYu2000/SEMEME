"""
Evaluate a model with a given block removal order (e.g. one produced by sememe.py).
Results are appended to a JSONL file; finished (num_removed, dataset) pairs are skipped on rerun.

    python evaluate_pruned.py --model_name meta-llama/Llama-2-7b-hf \
        --removal_order 12,18,24,11,23,10,29 --output_path results/llama2_7b.jsonl

    # after removing 1, 2, ..., 7 blocks, perplexity only
    python evaluate_pruned.py --model_name meta-llama/Llama-2-7b-hf \
        --removal_order 12,18,24,11,23,10,29 --incremental --eval_datasets wikitext2,c4 \
        --output_path results/llama2_7b_ppl_curve.jsonl
"""

import fire
import json
import copy
import time
import os
import gc
import logging

import torch

from sememeutils.logger_utils import setup_logger
from sememeutils.model_utils import get_llm
from sememeutils.onoff_utils.onoff import block_replace, turn_off
from sememeutils.eval_utils import load_and_eval_ppl, eval_zero_shot
from sememeutils.hf_utils import set_hf_token_env

logger_name='sememe'
setup_logger(log_dir="evaluate_pruned_logs", log_prefix="evaluate_pruned", logger_name=logger_name)
logger = logging.getLogger(logger_name)

PERPLEXITY_DATASETS = {'wikitext2', 'c4'}
TASK_DATASETS = {'piqa', 'winogrande', 'hellaswag', 'arc_challenge', 'arc_easy'}
ALL_DATASETS = PERPLEXITY_DATASETS | TASK_DATASETS


def evaluate_pruned(
    model_name: str,
    removal_order: str,
    output_path: str = 'eval_results.jsonl',
    device_map_auto: bool = False,
    eval_datasets: str = 'wikitext2,c4,piqa,winogrande,hellaswag,arc_challenge,arc_easy',
    num_to_remove: int = None,
    incremental: bool = False,
    eval_limit: int = 1000,
    eval_batch_size: int = 32,
    ppl_batch_size: int = 32,
):
    """
    removal_order: comma-separated block indices in pruning order
    device_map_auto: load with device_map="auto" (multi-GPU) instead of cuda:0
    num_to_remove: number of blocks from removal_order to remove (default: all)
    incremental: evaluate after each removal (1, ..., num_to_remove) instead of only at the end
    eval_limit: max docs per zero-shot task (lm-eval `limit`); None for the full test sets
    eval_batch_size / ppl_batch_size: batch sizes for zero-shot and perplexity evaluation
    """
    set_hf_token_env()

    # Fire may parse comma-separated values as tuples
    if isinstance(removal_order, (tuple, list)):
        removal_list = [int(x) for x in removal_order]
    else:
        removal_list = [int(x.strip()) for x in str(removal_order).split(',') if x.strip()]

    if isinstance(eval_datasets, (tuple, list)):
        datasets = [str(d).strip() for d in eval_datasets]
    else:
        datasets = [d.strip() for d in str(eval_datasets).split(',') if d.strip()]

    if num_to_remove is None:
        num_to_remove = len(removal_list)
    if num_to_remove > len(removal_list):
        raise ValueError(
            f"num_to_remove ({num_to_remove}) exceeds removal_order length ({len(removal_list)})"
        )

    ppl_datasets = [d for d in datasets if d in PERPLEXITY_DATASETS]
    task_datasets = [d for d in datasets if d in TASK_DATASETS]
    invalid = [d for d in datasets if d not in ALL_DATASETS]
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}")
    if not ppl_datasets and not task_datasets:
        raise ValueError("No valid datasets specified in eval_datasets")

    if incremental:
        steps = list(range(1, num_to_remove + 1))
    else:
        steps = [num_to_remove]

    device_map = "auto" if device_map_auto else "cuda:0"
    device = torch.device("cuda:0")

    logger.info(
        f"Configuration:\n"
        f"  Model: {model_name}\n"
        f"  Removal order ({len(removal_list)} blocks): {removal_list}\n"
        f"  Evaluation steps: {steps}\n"
        f"  Perplexity datasets: {ppl_datasets}\n"
        f"  Task datasets: {task_datasets}\n"
        f"  Device map: {device_map}\n"
        f"  Zero-shot eval: limit={eval_limit}/task, batch_size={eval_batch_size}\n"
        f"  Output: {output_path}"
    )

    completed = _load_completed(output_path)
    if completed:
        logger.info(f"Resuming: found {len(completed)} existing results in {output_path}")

    model = None
    if ppl_datasets:
        ppl_needed = any(
            (step, d) not in completed for step in steps for d in ppl_datasets
        )
        if ppl_needed:
            model = get_llm(model_name, device_map=device_map)
            model.config.use_cache = False
            model = block_replace(model)
            model.eval()

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Perplexity first, then zero-shot. lm-eval loads its own copy of the model, so the perplexity
    # model is freed in between (two copies of a 70B model do not fit).
    if model is not None:
        blocks_turned_off = 0
        for step in steps:
            current_removal = removal_list[:step]
            logger.info(f"=== [PPL] Step {step}/{num_to_remove}: {step} block(s) removed: {current_removal} ===")

            while blocks_turned_off < step:
                turn_off(model, removal_list[blocks_turned_off])
                blocks_turned_off += 1

            for dataset in ppl_datasets:
                if (step, dataset) in completed:
                    logger.info(f"Skipping {dataset} with {step} blocks removed (already computed)")
                    continue

                start_time = time.time()
                ppl = load_and_eval_ppl(model, device=device, dataset=dataset, bs=ppl_batch_size)
                elapsed = time.time() - start_time

                result = {
                    "model_name": model_name,
                    "num_removed": step,
                    "removal_list": current_removal,
                    "dataset": dataset,
                    "metric": "perplexity",
                    "value": ppl,
                    "elapsed_seconds": round(elapsed, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                _log_result(output_path, result)
                logger.info(f"{dataset} perplexity = {ppl:.2f} ({step} blocks removed, {elapsed:.1f}s)")

        del model
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if task_datasets:
        for step in steps:
            current_removal = removal_list[:step]
            logger.info(f"=== [zero-shot] Step {step}/{num_to_remove}: {step} block(s) removed: {current_removal} ===")
            remaining_tasks = [t for t in task_datasets if (step, t) not in completed]
            if not remaining_tasks:
                logger.info(f"Skipping task eval with {step} blocks removed (all already computed)")
                continue

            start_time = time.time()
            results = eval_zero_shot(
                model_name,
                copy.deepcopy(current_removal),
                task_list=remaining_tasks,
                parallelize=device_map_auto,
                eval_limit=eval_limit,
                eval_batch_size=eval_batch_size,
            )
            elapsed = time.time() - start_time
            task_results = results['results']

            for task in remaining_tasks:
                if task in task_results:
                    result = {
                        "model_name": model_name,
                        "num_removed": step,
                        "removal_list": current_removal,
                        "dataset": task,
                        "metric": "zero_shot",
                        "value": task_results[task],
                        "elapsed_seconds": round(elapsed, 2),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    _log_result(output_path, result)
                    logger.info(f"{task}: {task_results[task]} ({step} blocks removed)")
                else:
                    logger.warning(f"Task {task} not found in results for step {step}")

    logger.info(f"All evaluations complete. Results saved to {output_path}")


def _load_completed(output_path):
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    completed.add((r['num_removed'], r['dataset']))
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def _log_result(output_path, result):
    with open(output_path, 'a') as f:
        f.write(json.dumps(result) + '\n')
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    fire.Fire(evaluate_pruned)
