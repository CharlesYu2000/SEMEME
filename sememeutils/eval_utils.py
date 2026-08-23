# Import necessary modules
import time
from tqdm import tqdm
import os
import gc

import torch
import torch.nn as nn

# Import get_loaders function from data module within the same director
from sememeutils.data_utils import *
import fnmatch

import logging

from sememeutils.onoff_utils.onoff import turn_on_all

logger = logging.getLogger("sememe")


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
@torch.no_grad()
def load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2', testloader=None, tokenizer=None, bs=32):
    # Print status
    logger.info(f"Evaluating on {dataset}")

    # Get the test loader
    if testloader is None:
        if tokenizer is None:
            tokenizer = get_tokenizer(model.name)

        _, testloader = get_loaders(
            dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer
        )
        logger.info(f"Dataset Loaded.")

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl(model, testloader, bs, device)
    return ppl_test

@torch.no_grad()
def eval_ppl(model, testenc, bs=32, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    logger.info(f"nsamples {nsamples}")

    loss_fct = nn.CrossEntropyLoss()
    # Batched (halving bs on CUDA OOM); gives the same perplexity as bs=1
    pbar = tqdm(total=nsamples)
    i = 0
    while i < nsamples:
        b = min(bs, nsamples - i)
        try:
            inputs = testenc[:, (i * model.seqlen):((i + b) * model.seqlen)].to(device)
            inputs = inputs.reshape(b, model.seqlen)
            lm_logits = model(inputs).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            nlls.append(loss.float() * model.seqlen * b)
            del lm_logits, shift_logits, inputs
            i += b
            pbar.update(b)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or bs == 1:
                raise
            try:
                del inputs
            except NameError:
                pass
            try:
                del lm_logits
            except NameError:
                pass
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            logger.warning(f"CUDA OOM in eval_ppl; backing off to bs={bs}")
    pbar.close()

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl.item()

harness_lm_cache = {} # model_args -> harness lm, reused across calls during task-aware pruning

torch.no_grad()
def eval_zero_shot(model_name, removal_list, task_list=['piqa','winogrande','hellaswag','arc_challenge','arc_easy'],
        num_fewshot=0, parallelize=False, prune_version=False, total_blocks=None, num_samples=None, eval_limit=None, eval_batch_size='auto'):
    # prune_version=True (with total_blocks and num_samples) is used during task-aware pruning:
    # blocks are turned off in a cached harness model instead of being removed.
    logger.info(f"eval_zero_shot: model_name={model_name}, removal_list={removal_list}, task_list={task_list}, prune_version={prune_version}")

    if prune_version and (total_blocks is None or num_samples is None):
        raise ValueError("total_blocks and num_samples must be provided when prune_version is True.")

    from lm_eval import tasks, evaluator, utils
    task_manager = tasks.TaskManager(include_path='lm-evaluation-harness/lm_eval/tasks')

    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
        if os.path.isfile(task):
            config = utils.load_yaml_config(task)
            task_names.append(config)
    task_missing = [
        task
        for task in task_list
        if task not in task_names and "*" not in task
    ]  # we don't want errors if a wildcard ("*") task name was used
    if task_missing:
        logger.warning(f"Some tasks could not be matched: {task_missing}")

    model_args = f"pretrained={model_name},"
    if parallelize:
        model_args = f"pretrained={model_name},parallelize=True"

    if len(removal_list)>0:
        remove = True
    else:
        remove = False

    if prune_version:
        model = 'hf'
        if model_args in harness_lm_cache:
            model = harness_lm_cache[model_args]
            turn_on_all(model.model, total_blocks, model_name=model_name)

        results, harness_lm = evaluator.simple_evaluate(
            model=model,
            sememe_model_name=model_name,
            model_args=model_args,
            tasks=task_list,
            limit=num_samples,
            num_fewshot=num_fewshot,
            batch_size=eval_batch_size,
            max_batch_size=None,
            device='cuda:0',
            use_cache=None,
            check_integrity=False,
            write_out=False,
            gen_kwargs=None,
            task_manager=task_manager,
            remove = remove,
            removal_list = removal_list,
            return_lm = True,
            pruning_version = True,
        )
        harness_lm_cache[model_args] = harness_lm
        return results
    else:
        # lm-eval does not back off on a fixed int batch_size, so halve it on CUDA OOM and retry
        bs = eval_batch_size
        while True:
            try:
                results = evaluator.simple_evaluate(
                    model='hf',
                    sememe_model_name=model_name,
                    model_args=model_args,
                    tasks=task_list,
                    num_fewshot=num_fewshot,
                    batch_size=bs,
                    max_batch_size=None,
                    device='cuda:0',
                    use_cache=None,
                    limit=eval_limit,
                    check_integrity=False,
                    write_out=False,
                    gen_kwargs=None,
                    task_manager=task_manager,
                    remove = remove,
                    removal_list = removal_list,
                )
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not isinstance(bs, int) or bs <= 1:
                    logger.error(f"CUDA OOM in zero-shot eval at batch_size={bs}; cannot back off further.")
                    raise
                new_bs = max(1, bs // 2)
                logger.warning(f"CUDA OOM in zero-shot eval; backing off batch_size {bs} -> {new_bs} and retrying.")
                bs = new_bs
        return results
