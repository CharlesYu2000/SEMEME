import fire
import copy
import time
import json
import logging
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from sememeutils.logger_utils import setup_logger, add_run_file_handler
from sememeutils.model_utils import get_llm
from sememeutils.onoff_utils.onoff import block_replace, turn_off, turn_on
from sememeutils.data_utils import get_trainloaders, get_tokenizer, get_calibration_windows
from sememeutils.block_remove import block_remove
from sememeutils.eval_utils import load_and_eval_ppl, eval_zero_shot
from sememeutils.loss_utils import (get_per_example_lm_loss, get_arc_easy_loss_eval_type, get_arc_challenge_loss_eval_type,
                                    get_winogrande_loss_eval_type, get_hellaswag_loss_eval_type, get_piqa_loss_eval_type)
from sememeutils.dowhy_utils import get_next_block_to_prune
from sememeutils.hf_utils import set_hf_token_env

setup_logger(log_dir="logs", log_prefix="sememe")
logger = logging.getLogger("sememe")

# Sememe-I-Skill: the loss is the (negative) accuracy on the corresponding `<task>_prune` harness task
TASK_LOSS_FNS = {
    'arc_easy': get_arc_easy_loss_eval_type,
    'arc_challenge': get_arc_challenge_loss_eval_type,
    'winogrande': get_winogrande_loss_eval_type,
    'hellaswag': get_hellaswag_loss_eval_type,
    'piqa': get_piqa_loss_eval_type,
}
TASK_DATASETS = list(TASK_LOSS_FNS)
VALID_DATASETS = ['wikitext2', 'c4'] + TASK_DATASETS

@torch.no_grad()
def get_loss(model, testenc, bs=1, device=None, dataset=None, eval_prune_type_args=None):
    if dataset in TASK_LOSS_FNS:
        args = eval_prune_type_args
        return TASK_LOSS_FNS[dataset](args['model_name'], args['removal_list'], args['total_blocks'], args['num_samples'], device=None)
    if dataset in ('wikitext2', 'c4'):
        return get_per_example_lm_loss(model, testenc, batch_size=bs, device=device)
    raise ValueError(f"Unknown dataset: {dataset}")

def solve_structural_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # OLS fit of Y = sum_i gamma_i * B_i + zeta.
    # With one-at-a-time interventions the intercept is already in the column space of X,
    # so fitting with or without it gives the same block ranking.
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    return model.coef_

def standardize_coefficients(X: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    # gamma* = gamma * sd(B) / sd(Y)
    sigma_X = np.std(X, axis=0, ddof=1)
    sigma_y = np.std(X.dot(gamma), ddof=1)
    sigma_y = np.maximum(sigma_y, 1e-8)
    return gamma * (sigma_X / sigma_y)

def partition_into_heats(n_blocks: int, heat_size: int = 500) -> List[List[int]]:
    # Optionally split blocks into groups that are fit separately. The default (500) puts
    # every block in a single group, i.e. one joint fit as in the paper.
    heats = []
    for i in range(0, n_blocks, heat_size):
        heats.append(list(range(i, min(i + heat_size, n_blocks))))
    return heats

def sememe(
        model_name: str = 'meta-llama/Llama-2-7b-hf',
        num_blocks: int = 32,
        num_remove_blocks: int = 7,
        heat_size: int = 500,
        seed: int = 0,
        nsamples: int = 128,
        result_folder: str = 'sememe_results',
        result_file: str = 'sememe_results.txt',
        dataset: str = 'wikitext2',
        eval_ppl: bool = True,
        eval_zeroshot: bool = False,
        eval_limit: int = None,
        eval_batch_size='auto',
        model=None,
        use_aggregated_loss: bool = False,
        eval_removal_list: List[int] = None,
        blocks_to_kickstart_removal: List[int] = None,
        prune_method: str = 'ols',
        run_log_dir: str = 'logs',
        loss_batch_size: int = 1,
        calib_mode: str = 'windows',
        timing_dir: str = 'timings',
        shapley_samples: int = 256,
        shapley_remove_frac: float = 0.25,
):
    """
    model_name: HuggingFace model to prune
    num_blocks / num_remove_blocks: total number of transformer blocks / number to remove
    dataset: calibration data, 'wikitext2' or 'c4', or a task name (arc_easy, arc_challenge,
        winogrande, hellaswag, piqa) for Sememe-I-Skill
    nsamples: calibration set size N
    prune_method: 'ols' (Sememe-I), 'dowhy' (Sememe-C), 'shapley' (random multi-block
        interventions), 'adjacent' (single blocks + adjacent pairs)
    calib_mode: 'windows' = N random seqlen windows; 'concat' = N documents concatenated and chunked
    loss_batch_size: forward batch size for calibration losses (<= 0 batches all N at once)
    eval_removal_list: skip pruning and only evaluate this removal order
    blocks_to_kickstart_removal: remove these blocks first, then continue pruning
    eval_limit / eval_batch_size: lm-eval `limit` and batch size for the zero-shot evaluation
    """
    set_hf_token_env()

    if dataset not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}. Must be one of {VALID_DATASETS}.")

    if eval_removal_list is not None and isinstance(eval_removal_list, tuple):
        eval_removal_list = list(eval_removal_list)
    if blocks_to_kickstart_removal is not None and isinstance(blocks_to_kickstart_removal, tuple):
        blocks_to_kickstart_removal = list(blocks_to_kickstart_removal)

    add_run_file_handler(run_log_dir, model_name, prune_method, num_remove_blocks, nsamples,
                         seed=seed, heat_size=heat_size, dataset=dataset)

    logger.info(f"Arguments: model_name={model_name}, num_blocks={num_blocks}, num_remove_blocks={num_remove_blocks}, heat_size={heat_size}, seed={seed}, "
                f"nsamples={nsamples}, result_folder={result_folder}, result_file={result_file}, dataset={dataset}, eval_ppl={eval_ppl}, eval_zeroshot={eval_zeroshot}, "
                f"use_aggregated_loss={use_aggregated_loss}, eval_removal_list={eval_removal_list}; "
                f"blocks_to_kickstart_removal={blocks_to_kickstart_removal}; prune_method={prune_method}; "
                f"model={'provided' if model is not None else 'None'}")
    alive_list = [i for i in range(num_blocks)]
    removal_list = []

    if model is None:
        model = get_llm(model_name)
        use_cache = model.config.use_cache
        model.config.use_cache = False
        logger.info(f"Loaded Model: {model.name}")

        model = block_replace(model)
        model.eval()
    else:
        use_cache = True
        logger.info("Using provided model, assuming blocks are already replaced.")

    if eval_removal_list is None: # Then need to figure out what to prune!
        blocks_to_pre_remove = blocks_to_kickstart_removal if blocks_to_kickstart_removal is not None else []

        tokenizer = get_tokenizer(model_name)
        if dataset in ('wikitext2', 'c4'):
            if calib_mode == 'concat':
                # N documents concatenated and chunked into seqlen windows (fewer than N examples)
                _enc = get_trainloaders(dataset, nsamples=nsamples, seed=seed, model=model_name, tokenizer=tokenizer)
                _ids = _enc.input_ids
                _m = _ids.shape[1] // model.seqlen
                if _m == 0:
                    raise ValueError("concat calibration produced 0 full windows; increase nsamples")
                dataloader = _ids[:, :_m * model.seqlen].reshape(_m, model.seqlen)  # [num_chunks, seqlen]
            else:
                # exactly N random seqlen windows
                dataloader = get_calibration_windows(dataset, nsamples, seed, model.seqlen, model_name)  # [N, seqlen]
            logger.info(f"Calibration[{calib_mode}]: {tuple(dataloader.shape)} = {dataloader.shape[0]} examples "
                        f"x {model.seqlen} tokens (loss_batch_size={loss_batch_size}).")
        else:
            # task losses are computed by the harness on the `<task>_prune` task
            dataloader = None
        logger.info(f"Dataloader({dataset}) loaded.")

        def calib_losses(extra_removed):
            # per-example calibration losses with `extra_removed` blocks (on top of removal_list) turned off
            loss, losses = get_loss(model, dataloader, bs=loss_batch_size, device=torch.device("cuda:0"), dataset=dataset,
                                    eval_prune_type_args={'model_name': model_name,
                                                          'removal_list': removal_list + extra_removed,
                                                          'total_blocks': num_blocks,
                                                          'num_samples': nsamples})
            return loss, [l.item() if torch.is_tensor(l) else l for l in losses]

        start_point = time.time()
        fit_time_total = 0.0   # estimator fit time (OLS / GMM); the rest is forward passes
        phase_times = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        pending_adjacent = []  # 'adjacent' only: second block of a chosen pair, removed in the next step

        for t in tqdm(range(num_remove_blocks), desc="Pruning Blocks", leave=True):
            phase_start_point = time.time()

            current_blocks = len(alive_list)
            logger.debug(f"Phase {t+1}/{num_remove_blocks}: {current_blocks} blocks remaining.")

            if len(blocks_to_pre_remove) == 0:
                if prune_method == 'dowhy':
                    # Sememe-C: fit a GMM per block and query the counterfactual expected loss
                    losses_per_block = {}
                    for j in tqdm(range(current_blocks), desc="Testing Individual Block Removal", leave=False):
                        block_idx = alive_list[j]
                        turn_off(model, block_idx)
                        loss, losses = calib_losses([block_idx])
                        losses_per_block[block_idx] = [loss] if use_aggregated_loss else losses
                        logger.debug(f"Losses after removing block {block_idx}: {losses}")
                        turn_on(model, block_idx)
                    _fit_t0 = time.time()
                    min_impact_block_idx = get_next_block_to_prune(losses_per_block)
                    fit_time_total += time.time() - _fit_t0
                    p = alive_list.index(min_impact_block_idx)
                    logger.debug(f"[DoWhy] Selected block {min_impact_block_idx} (index {p}) for removal.")
                elif prune_method == 'shapley':
                    # Multi-block interventions: each of `shapley_samples` configurations removes every
                    # remaining block with probability `shapley_remove_frac`; all are fit jointly.
                    samples = []
                    _rng = np.random.default_rng(seed + t)
                    for _m in tqdm(range(shapley_samples), desc="Multi-block interventions", leave=False):
                        remove_mask = _rng.random(current_blocks) < shapley_remove_frac
                        if remove_mask.all():
                            remove_mask[_rng.integers(current_blocks)] = False  # keep at least one block alive
                        removed = [alive_list[j] for j in range(current_blocks) if remove_mask[j]]
                        for bi in removed:
                            turn_off(model, bi)
                        loss, losses = calib_losses(removed)
                        do_vector = np.ones(current_blocks)
                        do_vector[remove_mask] = 0
                        if use_aggregated_loss:
                            samples.append([do_vector, loss])
                        else:
                            for l in losses:
                                samples.append([do_vector, l])
                        for bi in removed:
                            turn_on(model, bi)
                    if not samples:
                        raise ValueError("No samples collected for multi-block estimation")
                    B_matrix = np.array([s[0] for s in samples])
                    Y_vector = np.array([s[1] for s in samples])
                    _fit_t0 = time.time()
                    all_gamma = solve_structural_equations(B_matrix, Y_vector)
                    all_gamma_std = standardize_coefficients(B_matrix, all_gamma)
                    fit_time_total += time.time() - _fit_t0
                    p = int(np.argmax(all_gamma_std))
                    min_impact_block_idx = alive_list[p]
                    logger.debug(f"[Shapley] Selected block {min_impact_block_idx} (index {p}); "
                                 f"M={shapley_samples}, remove_frac={shapley_remove_frac}, gamma*={all_gamma_std[p]:.6f}")
                elif prune_method == 'adjacent':
                    # Local design: baseline + every single-block removal + every adjacent-pair removal.
                    # Remove the intervention with the lowest loss increase per removed block.
                    if pending_adjacent:
                        min_impact_block_idx = pending_adjacent.pop(0)
                        p = alive_list.index(min_impact_block_idx)
                        logger.debug(f"[Adjacent] Removing second block of pair {min_impact_block_idx} (index {p}).")
                    else:
                        n = current_blocks
                        allow_pairs = (num_remove_blocks - len(removal_list)) >= 2  # only start a pair if both fit in the budget
                        def _cfg_loss(extra_removed):
                            _, vals = calib_losses(extra_removed)
                            return float(np.mean(vals)) if vals else 1e12
                        base_loss = _cfg_loss([])
                        single_loss = np.zeros(n)
                        for j in tqdm(range(n), desc="Adjacent: singles", leave=False):
                            bi = alive_list[j]
                            turn_off(model, bi)
                            single_loss[j] = _cfg_loss([bi])
                            turn_on(model, bi)
                        pair_loss = np.full(max(n - 1, 0), np.nan)
                        if allow_pairs:
                            for j in tqdm(range(n - 1), desc="Adjacent: pairs", leave=False):
                                a, b = alive_list[j], alive_list[j + 1]
                                turn_off(model, a); turn_off(model, b)
                                pair_loss[j] = _cfg_loss([a, b])
                                turn_on(model, a); turn_on(model, b)
                        _fit_t0 = time.time()
                        main = single_loss - base_loss  # loss increase from removing each block
                        best_cost, best_positions, best_kind = None, None, None
                        for j in range(n):
                            if best_cost is None or main[j] < best_cost:
                                best_cost, best_positions, best_kind = float(main[j]), [j], 'single'
                        inter = np.full(max(n - 1, 0), np.nan)
                        if allow_pairs:
                            for j in range(n - 1):
                                inter[j] = pair_loss[j] - single_loss[j] - single_loss[j + 1] + base_loss  # pair interaction (<0 = redundant)
                                pc = (pair_loss[j] - base_loss) / 2.0  # per-block cost of removing the pair
                                if pc < best_cost:
                                    best_cost, best_positions, best_kind = float(pc), [j, j + 1], 'pair'
                        fit_time_total += time.time() - _fit_t0
                        j = best_positions[0]
                        min_impact_block_idx = alive_list[j]
                        p = j
                        if best_kind == 'pair':
                            pending_adjacent.append(alive_list[j + 1])
                            logger.debug(f"[Adjacent] Pair ({alive_list[j]},{alive_list[j+1]}) per-block cost={best_cost:.6f} "
                                         f"interaction={inter[j]:.6f}; removing first now, second next step.")
                        else:
                            logger.debug(f"[Adjacent] Single block {alive_list[j]} (index {j}) cost={best_cost:.6f}.")
                        logger.debug(f"[Adjacent] main effects: {np.array2string(main, precision=4, max_line_width=200)}")
                        if allow_pairs:
                            logger.debug(f"[Adjacent] pair interactions: {np.array2string(inter, precision=4, max_line_width=200)}")
                else:
                    # Sememe-I: one-at-a-time interventions + OLS on the structural equation
                    samples = []  # [do_vector, loss] pairs
                    for j in tqdm(range(current_blocks), desc="Testing Individual Block Removal", leave=False):
                        block_idx = alive_list[j]
                        turn_off(model, block_idx)
                        loss, losses = calib_losses([block_idx])
                        logger.debug(f"Losses after removing block {block_idx}: {losses}")
                        do_vector = np.ones(current_blocks)
                        do_vector[j] = 0  # do(B_j = 0)
                        if use_aggregated_loss:
                            samples.append([do_vector, loss])
                        else:
                            for l in losses:
                                samples.append([do_vector, l])
                        turn_on(model, block_idx)
                    if not samples:
                        raise ValueError("No samples collected for causal modeling")
                    B_matrix = np.array([sample[0] for sample in samples])
                    Y_vector = np.array([sample[1] for sample in samples])
                    _fit_t0 = time.time()
                    heats = partition_into_heats(current_blocks, heat_size)
                    all_gamma = np.zeros(current_blocks)
                    all_gamma_std = np.zeros(current_blocks)
                    for heat in heats:
                        B_heat = B_matrix[:, heat]
                        gamma_heat = solve_structural_equations(B_heat, Y_vector)
                        gamma_heat_std = standardize_coefficients(B_heat, gamma_heat)
                        for local_idx, global_idx in enumerate(heat):
                            all_gamma[global_idx] = gamma_heat[local_idx]
                            all_gamma_std[global_idx] = gamma_heat_std[local_idx]
                    p = np.argmax(all_gamma_std)
                    min_impact_block_idx = alive_list[p]
                    fit_time_total += time.time() - _fit_t0
                    logger.debug(f"Selected block {min_impact_block_idx} (index {p}) for removal. Standardized Impact={all_gamma_std[p]:.6f}")
                    logger.debug(f"Gamma coefficients list: [{', '.join([str(g) for g in all_gamma])}]")
                    logger.debug(f"Standardized gamma list: [{', '.join([str(g) for g in all_gamma_std])}]")
            else:
                # Pre-remove specified blocks without causal modeling
                min_impact_block_idx = blocks_to_pre_remove.pop(0)
                p = alive_list.index(min_impact_block_idx)
                logger.debug(f"Pre-removed block {min_impact_block_idx} (index {p}) without causal modeling.")

            # Remove the selected block
            turn_off(model, min_impact_block_idx)
            removal_list.append(min_impact_block_idx)
            logger.debug(f"Current Block Removal List: {removal_list}")

            del alive_list[p]
            phase_times.append(time.time() - phase_start_point)

        finish_point = time.time()
        time_elapsed = finish_point - start_point

        peak_gpu_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None
        eval_time_total = max(time_elapsed - fit_time_total, 0.0)
        peak_str = f" peak_gpu={peak_gpu_gb:.2f}GB" if peak_gpu_gb is not None else ""
        logger.info(f"Prune timing: total={time_elapsed:.1f}s eval~={eval_time_total:.1f}s fit={fit_time_total:.3f}s{peak_str}")
        try:
            os.makedirs(timing_dir, exist_ok=True)
            with open(os.path.join(timing_dir, 'prune_timings.jsonl'), 'a') as _tf:
                _tf.write(json.dumps({
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    'model_name': model_name, 'prune_method': prune_method,
                    'num_blocks': num_blocks, 'num_remove_blocks': num_remove_blocks,
                    'nsamples': nsamples, 'seed': seed, 'heat_size': heat_size, 'dataset': dataset,
                    'prune_time_s': time_elapsed, 'fit_time_s': fit_time_total,
                    'eval_time_s': eval_time_total, 'phase_times_s': phase_times,
                    'peak_gpu_gb': peak_gpu_gb, 'removal_order': [int(x) for x in removal_list],
                }) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write prune timing log: {e}")

        logger.info(
            f"Time_Elapsed: {time_elapsed}\n"
            f"Model Name: {model_name}\n"
            f"# Total Blocks: {num_blocks}\n"
            f"# Remove Blocks: {num_remove_blocks}\n"
            f"Dataset: {dataset}\n"
            f"Seed: {seed}\n"
            f"Heat Size: {heat_size}\n"
            f"Block Removal Order: {removal_list}\n"
        )
    else:
        logger.info(f"Using provided removal list for eval only: {eval_removal_list}")
        removal_list = eval_removal_list

    if eval_ppl:
        logger.info("Starting PPL evaluation...")
        model = block_remove(model, copy.deepcopy(removal_list))
        model.config.use_cache = use_cache

        w2_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2')
        logger.info(f"WikiText-2 PPL = {w2_ppl:.2f}")

        c4_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='c4')
        logger.info(f"C4 PPL = {c4_ppl:.2f}")

    if eval_zeroshot:
        logger.info("Starting Zero-shot tasks evaluation...")
        if '30b' in model_name or '66b' in model_name or '70b' in model_name:
            parallelize = True
        else:
            parallelize = False

        tasks = ['piqa','winogrande','hellaswag','arc_challenge','arc_easy']
        if dataset in TASK_DATASETS:
            tasks = [dataset] # Only eval on the skill-localized task
        results = eval_zero_shot(model_name, copy.deepcopy(removal_list), tasks, parallelize=parallelize, eval_limit=eval_limit, eval_batch_size=eval_batch_size)
        results = results['results']

        for task in tasks:
            logger.info(f"{task}: {results[task]}")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_path = os.path.join(result_folder, result_file)

    with open(result_path, 'a') as file:
        sentences = []
        if eval_removal_list is None:
            if blocks_to_kickstart_removal is not None:
                sentences.append(f"Blocks to Kickstart Removal: {blocks_to_kickstart_removal}\n")
            sentences.append(f"Time Elapsed: {time_elapsed}\n")
            sentences.append(f"Model Name: {model_name}\n")
            sentences.append(f"# Total Blocks: {num_blocks}\n")
            sentences.append(f"# Remove Blocks: {num_remove_blocks}\n")
            sentences.append(f"Dataset: {dataset}\n")
            sentences.append(f"Seed: {seed}\n")
            sentences.append(f"Method: {prune_method}\n")
            sentences.append(f"Heat Size: {heat_size}\n")
            sentences.append(f"NSamples: {nsamples}\n")
            sentences.append(f"Use Aggregated Loss: {use_aggregated_loss}\n")
            sentences.append(f"Block Removal Order: {removal_list}\n")
        else:
            sentences.append(f"Evaluation with provided removal list: {eval_removal_list}\n")

        if eval_ppl:
            sentences.append(f"WikiText-2 PPL = {w2_ppl:.2f}\n")
            sentences.append(f"C4 PPL = {c4_ppl:.2f}\n")

        if eval_zeroshot:
            sentences.append("Zero-shot results: \n")
            for task in tasks:
                sentences.append(f"{task}: {results[task]}\n")
        sentences.append("\n")

        for sentence in sentences:
            file.write(sentence)

if __name__ == "__main__":
    fire.Fire(sememe)
