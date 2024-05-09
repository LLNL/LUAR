# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

"""Implements MAML and ProtoNet for RoBERTa / SBERT / LUAR.

MAML: https://arxiv.org/pdf/1703.03400.pdf
ProtoNet: https://arxiv.org/pdf/1703.05175.pdf

- Meta Optimizer = Adam
- Inner Optimizer = SGD
"""
import os
import sys
from argparse import ArgumentParser

import learn2learn as l2l
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    set_seed
)
from tqdm import tqdm

from fewshot_dataset import MetaDataset
from maml_helper import maml_loop
from prototypical_helper import prototypical_loop

parser = ArgumentParser()

parser.add_argument("--experiment_id", type=str, default="debug",
                    help="Experiment ID, used for logging and saving checkpoints.")
parser.add_argument("--fewshot_method", type=str, default="MAML",
                    choices=["MAML", "PROTONET"])
parser.add_argument("--metric", type=str, default="euclidean",
                    choices=["euclidean", "cosine"],
                    help="Which metric to use for Prototypical Networks.")
parser.add_argument("--num_train_iterations", type=int, default=1000,
                    help="Number of training iterations.")
parser.add_argument("--evaluate_every", type=int, default=50,
                    help="How often to evaluate the model on the validation set.")
parser.add_argument("--meta_batch_size", type=int, default=1,
                    help="Number of tasks per meta-batch.")
parser.add_argument("--num_tokens", type=int, default=64,
                    help="Number of tokens to use for each example.")
parser.add_argument("--seed", type=int, default=43)

parser.add_argument("--dont_filter_lengths", default=False, action="store_true",
                    help="If True, will not filter the human data by length.")
parser.add_argument("--use_many_topics", default=False, action="store_true",
                    help="If True, will use many topics for the humans.")

parser.add_argument("--n_inner_loop", type=int, default=5,
                    help="Number of inner loop iterations.")
parser.add_argument("--k_support", type=int, default=16,
                    help="Number of examples for the support class in each batch.")
parser.add_argument("--k_query", type=int, default=16,
                    help="Number of examples for the support set in each batch.")
parser.add_argument("--inner_lr", type=float, default=2e-5,
                    help="Learning rate for inner loop.")
parser.add_argument("--k_support_randomize", default=False, action="store_true",
                    help="If True, will randomize the number of elements in the support set.")

args = parser.parse_args()

args.meta_lr = args.inner_lr / 10.0

def to_device(inputs, device):
    """Move inputs to the specified device.
    """    
    for i, elem in enumerate(inputs):
        if isinstance(elem, dict):
            inputs[i]["input_ids"] = elem["input_ids"].to(device)
            inputs[i]["attention_mask"] = elem["attention_mask"].to(device)
        else:
            inputs[i] = elem.to(device)
    return inputs

def train(data_loader, model, optimizer, epoch, log):
    """Train the model for one epoch.
    """
    model.train()
    
    for X_support, y_support, X_query, y_query in data_loader:

        device = next(model.parameters()).device
        X_support, y_support, X_query, y_query = \
            to_device([X_support, y_support, X_query, y_query], device)

        if args.fewshot_method == "MAML":
            loss, query_losses, query_accuracies = maml_loop(
                model, X_support, y_support, X_query, y_query, args.n_inner_loop
            )
        else:
            loss, query_losses, query_accuracies = prototypical_loop(
                model, X_support, y_support, X_query, y_query, args.metric
            )

        optimizer.zero_grad()
        loss /= y_support.shape[0]
        loss.backward()
        optimizer.step()
        
        log.append({
            "epoch": epoch,
            "epoch_iteration": len(log),
            "query_loss": np.mean(query_losses),
            "query_accuracy": np.mean(query_accuracies),
        }),

def evaluate(data_loader, model, epoch, log):
    """Evaluates the model on the validation set.
    """
    model.train()
    
    for X_support, y_support, X_query, y_query in data_loader:
        
        device = next(model.parameters()).device
        X_support, y_support, X_query, y_query = \
            to_device([X_support, y_support, X_query, y_query], device)

        if args.fewshot_method == "MAML":
            _, query_losses, query_accuracies = maml_loop(
                model, X_support, y_support, X_query, y_query, args.n_inner_loop
            )
        else:
            _, query_losses, query_accuracies = prototypical_loop(
                model, X_support, y_support, X_query, y_query, args.metric
            )

        log.append({
            "epoch": epoch,
            "epoch_iteration": len(log),
            "query_loss": np.mean(query_losses),
            "query_accuracy": np.mean(query_accuracies),
        })

def save_state(output_path, model, optimizer, epoch, train_log, eval_log, is_best=False):
    """Saves the model and optimizer state.
    """
    state = {
        "state_dict": model.state_dict(),
        "meta_optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "train_log": train_log,
        "eval_log": eval_log,
    }
    if is_best:
        torch.save(state, os.path.join(output_path, "best.pt"))
    
    # write it as epoch_0001.pt
    torch.save(state, os.path.join(output_path, f"epoch_{epoch:05d}.pt"))
    
def main():
    set_seed(args.seed)

    output_path = os.path.join(utils.weight_path, args.fewshot_method)
    experiment_dir = os.path.join(output_path, args.experiment_id)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    print("loading roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    if args.fewshot_method == "MAML":
        model = l2l.algorithms.MAML(
            model, 
            lr=args.inner_lr, 
            first_order=False, 
            allow_nograd=True,
        )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    if torch.cuda.is_available():
        model.cuda()
    
    train_dataset = MetaDataset(tokenizer, split="train", 
                                k_support=args.k_support, k_query=args.k_query,
                                num_tokens=args.num_tokens,
                                filter_lengths=not args.dont_filter_lengths,
                                use_many_topics=args.use_many_topics,
    )
    valid_dataset = MetaDataset(tokenizer, split="valid", 
                                model_to_label=train_dataset.model_to_label,
                                k_support=args.k_support, k_query=args.k_query,
                                num_tokens=args.num_tokens,
                                filter_lengths=not args.dont_filter_lengths,
                                use_many_topics=args.use_many_topics,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.meta_batch_size, 
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.meta_batch_size, 
        shuffle=False, 
    )

    best_model = None
    train_log, eval_log = [], []
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.meta_lr if args.fewshot_method in ["MAML"] else args.inner_lr
    )
    for it in tqdm(range(args.num_train_iterations)):
        train(train_dataloader, model, optimizer, it, train_log)
        print(f"Epoch {it} | Train Query Loss: {train_log[-1]['query_loss']:.2f} | Train Query Accuracy: {train_log[-1]['query_accuracy']:.2f}")
    
        if it % args.evaluate_every == 0 or it == args.num_train_iterations - 1:
            evaluate(valid_dataloader, model, it, eval_log)
            print(f"\tEpoch {it} | Eval Query Loss: {eval_log[-1]['query_loss']:.2f} | Eval Query Accuracy: {eval_log[-1]['query_accuracy']:.2f}")

            if best_model is None or eval_log[-1]["query_loss"] < best_model["query_loss"]:
                best_model = eval_log[-1]
                save_state(experiment_dir, model, optimizer, it, train_log, eval_log, is_best=True)
            else:
                save_state(experiment_dir, model, optimizer, it, train_log, eval_log)

    return 0

if __name__ == "__main__":
    sys.exit(main())