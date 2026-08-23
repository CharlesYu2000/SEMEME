import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

import logging

logger = logging.getLogger("sememe")

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
        tokenizer.pad_token = tokenizer.eos_token # since we're only doing inference, this should impact equally/get masked out
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer, batch_size, return_individual_samples=False):

    traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    if return_individual_samples:
        # Return the list of (inp, tar) pairs, not batched
        return trainloader, testenc
    else:
        new_trainloader = []
        num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
        for i in range(0, num_batches):
            start =  i * batch_size
            end = min(start + batch_size, nsamples)
            batched_inp = []
            batched_tar = []
            for j in range(start, end):
                batched_inp.append(trainloader[j][0])
                batched_tar.append(trainloader[j][1])
            batched_inp = torch.cat(batched_inp)
            batched_tar = torch.cat(batched_tar)
            new_trainloader.append((batched_inp, batched_tar))
        del trainloader
        trainloader = new_trainloader
        del new_trainloader
        return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer, batch_size, return_individual_samples=False):

    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    if return_individual_samples:
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return trainloader, valenc
    else:
        new_trainloader = []
        num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
        for i in range(0, num_batches):
            start =  i * batch_size
            end = min(start + batch_size, nsamples)
            batched_inp = []
            batched_tar = []
            for j in range(start, end):
                batched_inp.append(trainloader[j][0])
                batched_tar.append(trainloader[j][1])
            batched_inp = torch.cat(batched_inp)
            batched_tar = torch.cat(batched_tar)
            new_trainloader.append((batched_inp, batched_tar))
        del trainloader
        trainloader = new_trainloader
        del new_trainloader
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, model='', batch_size=1, return_individual_samples=False):
    if tokenizer is None:
        tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer, batch_size, return_individual_samples=return_individual_samples)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer, batch_size, return_individual_samples=return_individual_samples)

def get_calibration_windows(name, nsamples, seed, seqlen, model_name):
    # Exactly `nsamples` random seqlen windows from the train split, as a LongTensor [nsamples, seqlen].
    # The corpus is tokenized once with a fast tokenizer (same token IDs as the slow one).
    if 'wikitext2' in name:
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(data['text'])
    elif 'c4' in name:
        data = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        text = " ".join(data[:max(2000, nsamples * 8)]['text'])
    else:
        raise ValueError(f"No calibration-window loader for dataset '{name}'")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name)
    enc = tok(text, return_tensors='pt').input_ids
    if enc.shape[1] <= seqlen + 1:
        raise ValueError(f"Calibration corpus has only {enc.shape[1]} tokens; need > {seqlen + 1}")
    random.seed(seed)
    rows = [enc[:, (start := random.randint(0, enc.shape[1] - seqlen - 1)):start + seqlen]
            for _ in range(nsamples)]
    return torch.cat(rows, dim=0)

def get_wikitext2_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size, return_individual_samples=False):

    traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split='train')
    traindata = traindata.shuffle(seed=seed)
    if return_individual_samples:
        # Return a list of tokenized samples, each of length seqlen
        samples = []
        for text in traindata[:nsamples]['text']:
            enc = tokenizer(text, return_tensors='pt')
            samples.append(enc.input_ids)
        return samples
    else:
        trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')
        return trainenc

def get_c4_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size, return_individual_samples=False):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    traindata = traindata.shuffle(seed=seed)

    if return_individual_samples:
        samples = []
        for text in traindata[:nsamples]['text']:
            enc = tokenizer(text, return_tensors='pt')
            samples.append(enc.input_ids)
        return samples
    else:
        trainenc = tokenizer(' '.join(traindata[:nsamples]['text']), return_tensors='pt')
        trainenc = trainenc.input_ids
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        trainenc = TokenizerWrapper(trainenc)
        return trainenc

def get_trainloaders(name, nsamples=128, seed=0, seqlen=2048, model='', batch_size=1, tokenizer=None, return_individual_samples=False):
    if tokenizer is None:
        tokenizer = get_tokenizer(model)

    if 'wikitext2' in name:
        return get_wikitext2_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size, return_individual_samples=return_individual_samples)
    if 'c4' in name:
        return get_c4_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size, return_individual_samples=return_individual_samples)
    raise ValueError(f"No calibration loader for dataset '{name}'")
