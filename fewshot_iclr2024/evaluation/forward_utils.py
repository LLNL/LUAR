# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

"""
Some utilities to help with the forward pass of Few-Shot methods like MAML / PROTONET / etc.
"""

import os
import sys

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath("../"))
from baseline_training.prototypical_helper import mean_pooling

##### MAML

def maml_adapt(model, query, n_inner_loop=5):
    """Adapts the model to the query (support set).
    """
    learner = model.clone()
    
    input_ids = query["input_ids"][:, :64].to(model.device)
    attention_mask = query["attention_mask"][:, :64].to(model.device)
    labels = query["labels"].to(model.device)
    for _ in range(n_inner_loop):
        support_out = learner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        learner.adapt(support_out.loss)

    return learner

def maml_evaluate(learner, targets, batch_size):
    """Evaluates a MAML learner on the targets (human background + machine targets)
    """
    learner.eval()
    data_loader = DataLoader(targets, batch_size=batch_size)
    probs = []

    with torch.inference_mode():
        for batch in data_loader:
            input_ids = batch["input_ids"].squeeze(1).to(learner.device)
            attention_mask = batch["attention_mask"].squeeze(1).to(learner.device)
            out = learner(
                input_ids, attention_mask, 
            )
            
            prob = out.logits.softmax(dim=-1)
            prob = prob[:, 1]
            probs.append(prob)

    probs = torch.cat(probs, axis=0).cpu().tolist()
    return probs

##### PROTONET

def get_prototypes(model, dataset, batch_size):
    """Gets the prototypes for the dataset.
    """
    all_proto = []
    data_loader = DataLoader(dataset, batch_size=batch_size)

    with torch.inference_mode():
        for batch in tqdm(data_loader):
            num_proto = batch["input_ids"].size(0)
            num_tokens = batch["input_ids"].size(-1)
            input_ids = batch["input_ids"].reshape(-1, num_tokens).to(model.device)
            attention_mask = batch["attention_mask"].reshape(-1, num_tokens).to(model.device)
            proto = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states[-1]
            proto = mean_pooling(proto.size(-1), proto, attention_mask)
            proto = proto.reshape(num_proto, -1, proto.size(-1)).mean(dim=1)
            all_proto.append(proto)
    
    return torch.cat(all_proto, axis=0).cpu()

##### SBERT / Anna Wegmann

def extract_sbert_embeddings(episodes, model, tokenizer, num_tokens=512, batch_size=128):
    """Takes any model with SBERT-like forward pass and extracts the embeddings.
    """
    embeddings = []
    for i in tqdm(range(0, len(episodes), batch_size)):
        batch = episodes[i:i+batch_size]
        B, E = len(batch), len(batch[0])

        batch = [j for i in batch for j in i]
        batch = tokenizer.batch_encode_plus(
            batch,
            max_length=num_tokens,
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        if torch.cuda.is_available():
            batch["input_ids"] = batch["input_ids"].cuda()
            batch["attention_mask"] = batch["attention_mask"].cuda()
        
        with torch.inference_mode():
            output = model(input_ids=batch["input_ids"], 
                           attention_mask=batch["attention_mask"], 
                           output_hidden_states=True)["hidden_states"][-1]
            output = mean_pooling(output.size(-1), output, batch["attention_mask"])
            output = output.reshape(B, E, -1).mean(dim=1)

        embeddings.append(output.detach().cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)

##### LUAR

def extract_luar_embeddings(episodes, model, tokenizer, num_tokens=512, batch_size=128):
    """Extract the LUAR embeddings for the data.
    """
    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(episodes), batch_size)):
        batch = episodes[i:i+batch_size]
        E = len(batch[0])

        batch = [j for i in batch for j in i]
        batch = tokenizer.batch_encode_plus(
            batch,
            max_length=num_tokens,
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        batch["input_ids"] = rearrange(batch["input_ids"], "(b e) d -> b e d", e=E)
        batch["attention_mask"] = rearrange(batch["attention_mask"], "(b e) d -> b e d", e=E)

        if torch.cuda.is_available():
            batch["input_ids"] = batch["input_ids"].cuda()
            batch["attention_mask"] = batch["attention_mask"].cuda()
        
        with torch.inference_mode():
            output = model(**batch)

        embeddings.append(output.detach().cpu().numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings
