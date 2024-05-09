# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Prototypical Networks
"""

import torch
from einops import reduce, repeat
import torch.nn.functional as F

def mean_pooling(hidden_size, token_embeddings, attention_mask):
    """Mean Pooling as described in the SBERT paper.
    """
    input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=hidden_size).float()
    sum_embeddings = reduce(token_embeddings * input_mask_expanded, 'b l d -> b d', 'sum')
    sum_mask = torch.clamp(reduce(input_mask_expanded, 'b l d -> b d', 'sum'), min=1e-9)
    return sum_embeddings / sum_mask

def distance_metric(a, b, metric="euclidean"):
    # https://github.com/yinboc/prototypical-network-pytorch/blob/master/utils.py#L46
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    
    if metric == "euclidean":
        logits = -((a - b)**2).sum(dim=2)
    elif metric == "cosine":
        logits = F.cosine_similarity(a, b, dim=2)
    else:
        raise NotImplementedError

    return logits

def prototypical_step(model, X_support, y_support, X_query, y_query, task_idx, metric):
    """One step of Prototypical Networks, evaluated on one task.
    """
    support_out = model(
        input_ids=X_support["input_ids"][task_idx],
        attention_mask=X_support["attention_mask"][task_idx],
        labels=y_support[task_idx],
        output_hidden_states=True,
    ).hidden_states[-1]
    support_out = mean_pooling(support_out.size(-1), support_out, X_support["attention_mask"][task_idx])
    proto = support_out.reshape(-1, 2, support_out.size(-1)).mean(dim=0)
    
    query_out = model(
        input_ids=X_query["input_ids"][task_idx],
        attention_mask=X_query["attention_mask"][task_idx],
        labels=y_query[task_idx],
        output_hidden_states=True,
    ).hidden_states[-1]
    query_out = mean_pooling(query_out.size(-1), query_out, X_query["attention_mask"][task_idx])
    
    logits = distance_metric(query_out, proto, metric=metric)
    loss = F.cross_entropy(logits, y_query[task_idx])

    return loss, logits

def prototypical_loop(model, X_support, y_support, X_query, y_query, metric):
    """Prototypical Networks loop, evaluated on all tasks.
    """
    num_tasks = y_support.shape[0]
    query_accuracies = []
    query_losses = []
    loss_agg = 0.0

    for task_idx in range(num_tasks):
        loss, logits = prototypical_step(model, X_support, y_support, X_query, y_query, task_idx, metric)

        loss_agg += loss
        query_losses.append(loss.item())
        query_accuracies.append((logits.argmax(dim=1) == y_query[task_idx]).float().mean().item())
    
    return loss_agg, query_losses, query_accuracies