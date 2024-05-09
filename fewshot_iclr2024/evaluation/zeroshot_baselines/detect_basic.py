# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

def get_rank(text, base_model, base_tokenizer, log=False):
    """From: https://github.com/eric-mitchell/detect-gpt/blob/main/run.py#L298C1-L320C43
    """
    with torch.no_grad():
        tokenized = base_tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(base_model.device)
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized["input_ids"][:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()

def get_entropy(text, base_model, base_tokenizer):
    """From: https://github.com/eric-mitchell/detect-gpt/blob/main/run.py#L324C1-L332C1
    """
    with torch.no_grad():
        tokenized = base_tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(base_model.device)
        logits = base_model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()

def detect_basic(generation, base_model, base_tokenizer, method_name="rank"):
    """Detection function for the simple ZeroShot baselines.
    """
    assert method_name in ["rank", "logrank", "entropy"]
    
    if method_name == "rank":
        return -get_rank(generation, base_model, base_tokenizer, log=False)
    elif method_name == "logrank":
        return -get_rank(generation, base_model, base_tokenizer, log=True)
    else:
        return get_entropy(generation, base_model, base_tokenizer)