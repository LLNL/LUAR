# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

HUGGINGFACE_NEG_VALUE = -10000.

class SelfAttention(nn.Module):
    """Implements Dot-Product Self-Attention as used in "Attention is all You Need".
    """
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, k, q, v):
        d_k = q.size(-1)
        scores = torch.matmul(k, q.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, v)
    
############################################################
# Self-Attention Mechanisms
############################################################

# Adapted LucidRains impl. of Memory Efficient Attention
# https://github.com/lucidrains/memory-efficient-attention-pytorch

def exists(val):
    return val is not None

def summarize_qkv_chunk(
    q, k, v, 
    mask
):
    """Dot-Product Attention for a chunk of queries, keys, and values.
    """
    weight = torch.einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(mask):
        # HuggingFace masks have to be added:
        weight += mask

    weight_max = weight.amax(dim = -1, keepdim = True).detach()
    weight = weight - weight_max

    exp_weight = weight.exp()
    weighted_value = torch.einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim = -1), weighted_value, rearrange(weight_max, '... 1 -> ...')

checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)

def memory_efficient_attention(
    q, k, v,
    mask = None,
    q_bucket_size = 512,
    k_bucket_size = 1024,
    eps = 1e-8
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # function
    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

    # chunk all the inputs
    q_chunks = q.split(q_bucket_size, dim = -2)
    k_chunks = k.split(k_bucket_size, dim = -2)
    v_chunks = v.split(k_bucket_size, dim = -2)
    mask_chunks = mask.split(k_bucket_size, dim = -1) if exists(mask) else ((None,) * len(k_chunks))

    # loop through all chunks and accumulate
    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []
        
        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):

            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
            )

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        exp_weights = torch.stack(exp_weights, dim = -1)
        weighted_values = torch.stack(weighted_values, dim = -1)
        weight_maxes = torch.stack(weight_maxes, dim = -1)

        global_max = weight_maxes.amax(dim = -1, keepdim = True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim = -1)
        all_weights = exp_weights.sum(dim = -1)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)

    return torch.cat(out, dim = -2)


class MemoryEfficientAttention(nn.Module):
    """Memory Efficient Attention: https://arxiv.org/abs/2112.05682
    
       Memory Complexity - O(log n)
       Time Complexity - O(n^2)
    """
    def __init__(
        self,
        *,
        dim = 768,
        heads = 12,
        dim_head = 64,
        memory_efficient = False,
        q_bucket_size = 512,
        k_bucket_size = 1024
    ):
        super().__init__()
        self.heads = heads

        inner_dim = heads * dim_head

        self.key = nn.Linear(dim, inner_dim)
        self.query = nn.Linear(dim, inner_dim)
        self.value = nn.Linear(dim, inner_dim)

        self.memory_efficient = memory_efficient
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        # the following parameters are expected by the HuggingFace
        # implementation of Attention but not used here:
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        h = self.heads

        k = self.key(hidden_states)
        q = self.query(hidden_states)
        v = self.value(hidden_states)

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), 
            (q, k, v)
        )

        out = memory_efficient_attention(
            q, k, v, 
            mask=attention_mask, 
            q_bucket_size=self.q_bucket_size, 
            k_bucket_size=self.k_bucket_size
        )

        out = rearrange(out, 'b h n d -> b n (h d)')

        return (out,)