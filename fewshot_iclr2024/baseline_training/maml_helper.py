# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for MAML
"""

def maml_step(model, X_support, y_support, X_query, y_query, task_idx, n_inner_loop):
    """One step of MAML, evaluated on one task.
    """
    learner = model.clone()
    
    for _ in range(n_inner_loop):
        support_out = learner(
            input_ids=X_support["input_ids"][task_idx],
            attention_mask=X_support["attention_mask"][task_idx],
            labels=y_support[task_idx],
        )
        
        learner.adapt(support_out.loss)
    
    query_out = learner(
        input_ids=X_query["input_ids"][task_idx],
        attention_mask=X_query["attention_mask"][task_idx],
        labels=y_query[task_idx],
    )
    loss = query_out.loss
    logits = query_out.logits

    return loss, logits

def maml_loop(model, X_support, y_support, X_query, y_query, n_inner_loop):
    """MAML loop, evaluated on all tasks.
    """
    num_tasks = y_support.shape[0]
    query_accuracies = []
    query_losses = []
    loss_agg = 0.0

    for task_idx in range(num_tasks):
        loss, logits = maml_step(model, X_support, y_support, X_query, y_query, task_idx, n_inner_loop)
        loss_agg += loss
        query_losses.append(loss.item())
        query_accuracies.append((logits.argmax(-1) == y_query[task_idx]).float().mean().item())

    return loss_agg, query_losses, query_accuracies