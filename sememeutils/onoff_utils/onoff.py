from sememeutils.onoff_utils import onoff_opt, onoff_llama, onoff_qwen, onoff_mistral

import logging

logger = logging.getLogger("sememe")

def block_replace(model, model_name=None):
    logger.debug(f"Replacing all blocks: {model_name or model.name}")
    if 'opt' in (model_name or model.name.lower()):
        model = onoff_opt.block_replace(model)
    elif 'llama' in (model_name or model.name.lower()):
        model = onoff_llama.block_replace(model)
    elif 'qwen' in (model_name or model.name.lower()):
        model = onoff_qwen.block_replace(model)
    elif 'mistral' in (model_name or model.name.lower()):
        model = onoff_mistral.block_replace(model)
    else:
        raise ValueError(f"Unsupported model type for block replacement: {model.name}")

    return model

def turn_on_all(model, total_blocks, model_name=None):
    logger.debug(f"Turning on all blocks: {total_blocks}")
    for block in range(total_blocks):
        turn_on(model, block, model_name=model_name)

def turn_off(model, block_idx, model_name=None):
    if 'opt' in (model_name or model.name.lower()):
        onoff_opt.turn_off(model, block_idx)
    elif 'llama' in (model_name or model.name.lower()):
        onoff_llama.turn_off(model, block_idx)
    elif 'qwen' in (model_name or model.name.lower()):
        onoff_qwen.turn_off(model, block_idx)
    elif 'mistral' in (model_name or model.name.lower()):
        onoff_mistral.turn_off(model, block_idx)
    else:
        raise ValueError(f"Unsupported model type for turn_off: {model.name}")

def turn_on(model, block_idx, model_name=None):
    if 'opt' in (model_name or model.name.lower()):
        onoff_opt.turn_on(model, block_idx)
    elif 'llama' in (model_name or model.name.lower()):
        onoff_llama.turn_on(model, block_idx)
    elif 'qwen' in (model_name or model.name.lower()):
        onoff_qwen.turn_on(model, block_idx)
    elif 'mistral' in (model_name or model.name.lower()):
        onoff_mistral.turn_on(model, block_idx)
    else:
        raise ValueError(f"Unsupported model type for turn_on: {model.name}")

def scan(model, num_blocks):

    if 'opt' in model.name.lower():
        onoff_opt.scan(model, num_blocks)
    elif 'llama' in model.name.lower():
        onoff_llama.scan(model, num_blocks)
    elif 'qwen' in model.name.lower():
        onoff_qwen.scan(model, num_blocks)
    elif 'mistral' in model.name.lower():
        onoff_mistral.scan(model, num_blocks)
    else:
        raise ValueError(f"Unsupported model type for scan: {model.name}")
