import torch
import torch.nn as nn
from sememeutils.eval_utils import eval_zero_shot
import logging

logger = logging.getLogger("sememe")

@torch.no_grad()
def get_per_example_lm_loss(model, samples, batch_size=1, device=None):
    # samples: [N, seqlen] calibration windows. Returns (loss_sum, losses) where each per-example
    # loss is mean token cross-entropy * seqlen. batch_size <= 0 starts from all N at once;
    # the batch size is halved on CUDA OOM and the largest size that fit is cached on the model.
    n = samples.shape[0]
    requested = n if (batch_size is None or batch_size <= 0) else min(batch_size, n)
    bs = min(requested, getattr(model, "_sememe_loss_bs", requested))
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    losses = []
    i = 0
    while i < n:
        b = min(bs, n - i)
        batch = None
        try:
            batch = samples[i:i + b].to(device)
            logits = model(batch).logits
            tok = loss_fct(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                           batch[:, 1:].reshape(-1)).view(b, -1)       # [b, seqlen-1] per-token CE
            seq = (tok.mean(dim=1) * model.seqlen).float()             # [b] per-sequence loss
            for s in seq:
                if not bool(torch.isnan(s)):
                    losses.append(s.detach())
            del logits, tok, seq, batch
            i += b
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            del batch
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            logger.warning(f"CUDA OOM at loss batch {b}; backing off to loss_batch_size={bs}")
    model._sememe_loss_bs = bs
    loss_sum = torch.stack(losses).sum().item() if losses else 999999999.0
    return loss_sum, losses

# Task losses for Sememe-I-Skill: negative accuracy on the `<task>_prune` harness task, so that
# the block whose removal hurts accuracy the _least_ is selected.

@torch.no_grad()
def get_arc_easy_loss_eval_type(model_name, removal_list, total_blocks, num_samples, device=None):
    results = eval_zero_shot(model_name, removal_list, task_list=['arc_easy_prune'],
        num_fewshot=0, parallelize=False, prune_version=True, total_blocks=total_blocks, num_samples=num_samples)

    logger.debug(f"ARC-Easy eval_type results: {results}")
    acc_value = results['results']['arc_easy_prune']['acc_norm,none']
    losses = [-acc_value]
    loss_sum = sum(losses)

    return loss_sum, losses

@torch.no_grad()
def get_arc_challenge_loss_eval_type(model_name, removal_list, total_blocks, num_samples, device=None):
    results = eval_zero_shot(model_name, removal_list, task_list=['arc_challenge_prune'],
        num_fewshot=0, parallelize=False, prune_version=True, total_blocks=total_blocks, num_samples=num_samples)

    logger.debug(f"ARC-Challenge eval_type results: {results}")
    acc_value = results['results']['arc_challenge_prune']['acc_norm,none']
    losses = [-acc_value]
    loss_sum = sum(losses)

    return loss_sum, losses

@torch.no_grad()
def get_winogrande_loss_eval_type(model_name, removal_list, total_blocks, num_samples, device=None):
    results = eval_zero_shot(model_name, removal_list, task_list=['winogrande_prune'],
        num_fewshot=0, parallelize=False, prune_version=True, total_blocks=total_blocks, num_samples=num_samples)

    logger.debug(f"Winogrande eval_type results: {results}")
    acc_value = results['results']['winogrande_prune']['acc,none']
    losses = [-acc_value]
    loss_sum = sum(losses)

    return loss_sum, losses

@torch.no_grad()
def get_hellaswag_loss_eval_type(model_name, removal_list, total_blocks, num_samples, device=None):
    results = eval_zero_shot(model_name, removal_list, task_list=['hellaswag_prune'],
        num_fewshot=0, parallelize=False, prune_version=True, total_blocks=total_blocks, num_samples=num_samples)

    logger.debug(f"Hellaswag eval_type results: {results}")
    acc_value = results['results']['hellaswag_prune']['acc_norm,none']
    losses = [-acc_value]
    loss_sum = sum(losses)

    return loss_sum, losses

@torch.no_grad()
def get_piqa_loss_eval_type(model_name, removal_list, total_blocks, num_samples, device=None):
    results = eval_zero_shot(model_name, removal_list, task_list=['piqa_prune'],
        num_fewshot=0, parallelize=False, prune_version=True, total_blocks=total_blocks, num_samples=num_samples)

    logger.debug(f"PIQA eval_type results: {results}")
    acc_value = results['results']['piqa_prune']['acc_norm,none']
    losses = [-acc_value]
    loss_sum = sum(losses)

    return loss_sum, losses
