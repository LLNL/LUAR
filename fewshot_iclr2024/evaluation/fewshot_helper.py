# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

"""
FewShot helper functions.
"""
import logging; logging.basicConfig(level=logging.INFO)
import os
import random; random.seed(43)
import sys
from collections import defaultdict
from datetime import datetime

import learn2learn as l2l
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
)
from sklearn.metrics import (
    pairwise_distances,
    roc_auc_score,
    roc_curve,
)

sys.path.append(os.path.abspath("../"))
from baseline_training.prototypical_helper import distance_metric
from file_utils import Utils as utils
from evaluation.forward_utils import maml_adapt, maml_evaluate
from evaluation.roberta_detector import Detector

def load_state_dict(checkpoint_name_or_path):
    """Loads the state dict from the given checkpoint.
    """
    state_dict = torch.load(checkpoint_name_or_path)["state_dict"]
    if "roberta.embeddings.position_ids" in state_dict:
        del state_dict["roberta.embeddings.position_ids"]
    if "roberta.embeddings.token_type_ids" in state_dict:
        del state_dict["roberta.embeddings.token_type_ids"]
    return state_dict

def to_gpu(model):
    """Moves the model to the GPU if available.
    """
    if torch.cuda.is_available(): 
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        else:
            model = model.cuda()
    return model

def load_model_and_tokenizer(model_name, adaptation_lr=None):
    """Loads the model and tokenizer for all possible model variations.
    """
    if model_name in ["MUD", "CRUD", "Multi-LLM", "Multi-domain"]:
        hfid_or_path = {
            "MUD": "rrivera1849/LUAR-MUD",
            "CRUD": "rrivera1849/LUAR-CRUD",
            "Multi-LLM": os.path.join(utils.weights_path, "Multi-LLM"),
            "Multi-domain": os.path.join(utils.weights_path, "Multi-domain"),
        }[model_name]
        model = AutoModel.from_pretrained(hfid_or_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hfid_or_path)
        model = to_gpu(model)

    elif model_name == "roberta_finetuned" or model_name=="openAI":
        model = Detector(model_name)
        model.to_gpu()
        tokenizer = model.get_tokenizer()

    elif model_name == "MAML":
        assert adaptation_lr is not None, "Must provide a fewshot_lr for MAML"
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        model = l2l.algorithms.MAML(model, lr=adaptation_lr, allow_nograd=True)
        state_dict = load_state_dict(os.path.join(utils.weights_path, "MAML/from_scratch/best.pt"))
        model.load_state_dict(state_dict, strict=True)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = to_gpu(model)

    elif model_name == "PROTONET":
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        state_dict = load_state_dict(os.path.join(utils.weights_path, "PROTONET/from_scratch/best.pt"))
        model.load_state_dict(state_dict, strict=True)
        model = to_gpu(model)

    elif model_name == "SBERT":
        model = AutoModel.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v1")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v1")
        model = to_gpu(model)
        
    elif model_name == "WEGMANN":
        model = AutoModel.from_pretrained("AnnaWegmann/Style-Embedding")
        tokenizer = AutoTokenizer.from_pretrained("AnnaWegmann/Style-Embedding")
        model = to_gpu(model)
        
    elif model_name in ["rank", "logrank", "entropy"]:
        tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
        model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
        model = to_gpu(model)
        model.eval()

    return model, tokenizer

def random_combinations(lists, k=10):
    """Returns k random combinations of len(lists) where we randomly draw
       one element from each list.
    """
    i = 0
    seen = {}
    result = []
    while i < k:
        sample = []
        for l in lists: 
            sample.append(random.choice(l))

        key = "_".join([str(s) for s in sample])
        if key in seen: 
            continue

        seen[key] = True
        result.append(sample)
        i += 1
    return result

def index_generator(
    INDEX_TO_AUTHOR, 
    machine_support_labels,
    machine_queries_labels,
    support_to_queries,
    max_trials,
    mode
):
    """Generate the support and query indices for the given model.
    """
    if mode == "multiple_target":
        lm_indices = [np.where(machine_support_labels == lm_index)[0].tolist() for lm_index, name in INDEX_TO_AUTHOR.items() if name not in ["opt", "gpt2"]]
        lm_indices_comb = random_combinations(lm_indices, k=max_trials)
        
        for trial, support_indices in enumerate(lm_indices_comb):
            to_delete = [support_to_queries[support_index] for support_index in support_indices]
            to_delete = [item for sublist in to_delete for item in sublist]
            queries_index = np.delete(
                np.arange(len(machine_queries_labels)),
                to_delete,
            )
            yield "ALL", trial, support_indices, queries_index
    else:
        for lm_index, LM in INDEX_TO_AUTHOR.items():
            if LM in ["opt", "gpt2"]: continue
            trial = 0
            
            for support_index in np.where(machine_support_labels == lm_index)[0]:

                try:
                    queries_index = np.where(machine_queries_labels == lm_index)[0]
                    to_delete = np.argwhere(queries_index == support_to_queries[support_index])
                    queries_index = np.delete(
                        queries_index,
                        to_delete
                    )
                except:
                    # this shouldn't hurt too much
                    queries_index = np.where(machine_queries_labels == lm_index)[0]

                yield LM, trial, support_index, queries_index
                
                trial += 1
                if trial >= max_trials:
                    break

def calculate_roc_metrics(
        labels, 
        scores,
    ):
    """Calculates the Detection metrics for the given labels and scores.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    roc_auc = roc_auc_score(labels, scores)
    roc_auc_cutoff = roc_auc_score(labels, scores, max_fpr=10**-2)
    
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": roc_auc,
        "roc_auc_cutoff": roc_auc_cutoff,
    }

def calculate_detector_metrics(
    INDEX_TO_AUTHOR,
    machine_episodes_labels,
    machine_episodes_probs,
    background_probs,
):
    """Calculates the ROC metrics for a generic detector.
    """
    metrics = defaultdict(dict)
    
    all_roc_metrics = []
    for lm_index, LM in INDEX_TO_AUTHOR.items():
        if LM in ["opt", "gpt2"]: continue
        indices = np.where(machine_episodes_labels == lm_index)[0]
        roc_labels = [1 for _ in range(len(indices))] + [0 for _ in range(len(background_probs))]
        lm_probs = np.concatenate((machine_episodes_probs[indices], background_probs), axis=0)
        metrics[LM]["trial=000"] = calculate_roc_metrics(roc_labels, lm_probs)
        all_roc_metrics.append(metrics[LM]["trial=000"])
    metrics["global"]["roc"] = all_roc_metrics

    return metrics

def calculate_nn_metrics(
        INDEX_TO_AUTHOR,
        machine_support_embeddings, machine_support_labels,
        machine_queries_embeddings, machine_queries_labels,
        background_embeddings,
        support_to_queries,
        max_trials,
        mode,
    ):
    """Calculates the LUAR metrics for the given embeddings.
    """
    background_distances = pairwise_distances(
        machine_support_embeddings, 
        Y=background_embeddings,
        metric="cosine", 
        n_jobs=-1
    )

    metrics = defaultdict(dict)
    metrics["global"] = defaultdict(float)
    all_roc_metrics = []
    generator = index_generator(
        INDEX_TO_AUTHOR, 
        machine_support_labels, 
        machine_queries_labels, 
        support_to_queries, 
        max_trials,
        mode,
    )
    
    for index in generator:
        LM, trial, support_index, query_indices = index

        if mode == "multiple_target" or mode == "multiple_target_paraphrase":
            if mode == "multiple_target_paraphrase":
                offset = machine_support_embeddings.shape[0] // 2
                support_index = [support_index, support_index + offset]
            
            distances = []
            for sindex in support_index:
                dist_background = background_distances[sindex]
                dist_target = pairwise_distances(
                    machine_support_embeddings[sindex].reshape(1, -1),
                    Y=machine_queries_embeddings[query_indices],
                    metric="cosine",
                ).flatten()
                dist = np.concatenate((dist_target, dist_background), axis=0)
                distances.append(dist)
            dist = np.vstack(distances).min(axis=0)
        else:
            dist_background = background_distances[support_index]
            dist_target = pairwise_distances(
                machine_support_embeddings[support_index].reshape(1, -1), 
                Y=machine_queries_embeddings[query_indices], 
                metric="cosine",
            ).flatten()
            dist = np.concatenate((dist_target, dist_background), axis=0)

        labels = [1 for _ in range(len(dist_target))] + [0 for _ in range(len(dist_background))]
        roc_metrics = calculate_roc_metrics(labels, (-dist).tolist())
        metrics[LM]["trial_{:03d}".format(trial)] = roc_metrics
        all_roc_metrics.append(roc_metrics)

    metrics["global"]["roc"] = all_roc_metrics
    return metrics

def tokenize_function(examples, tokenizer, max_length=512):
    tokenization = tokenizer(examples["text"], 
                            truncation=True, 
                            padding="max_length", 
                            max_length=max_length, 
                            return_tensors="pt"
                            )
    return tokenization

def create_dataset(dictionary, tokenizer=False, max_token_length=512):
    """ Create dataset. Optionally, tokenize dataset."""
    dataset = Dataset.from_dict(dictionary)
    if tokenizer:
        dataset = dataset.map(lambda x : tokenize_function(x, tokenizer, max_token_length))
        dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    return dataset

def tokenize_dataset(text, label, model, tokenizer, concat=False):
    # Doesn't feel like this belongs here...
    if concat:
        labels = [label for _ in range(len(text))]
    else:
        labels = [[label for _ in range(len((t)))] for t in text] 
    dictionary = {
        "text": text,
        "labels": labels 
        }
    tokenized_dataset = create_dataset(dictionary, tokenizer)
    return tokenized_dataset

def calculate_adaptation_metrics(
        INDEX_TO_AUTHOR,
        model, tokenizer,
        machine_support, machine_support_labels,
        machine_queries, machine_queries_labels,
        background,
        support_to_queries,
        max_trials,
        num_few_shot_epochs,
        mode,
):
    """Uses the queries to produce probabilities which can then be used with the 
       target labels to calculate the ROC metrics.
    """
    metrics = defaultdict(dict)
    metrics["global"] = defaultdict(float)
    all_roc_metrics = []
    
    logging.info("tokenizing background and machines (this takes a minute)")
    tokenized_background = tokenize_dataset(background, 0, model, tokenizer, concat=True)
    tokenized_machine_queries = tokenize_dataset(machine_queries, 1, model, tokenizer, concat=True)
    tokenized_machine_support = tokenize_dataset(machine_support.syms, 1, model, tokenizer)

    generator = index_generator(
        INDEX_TO_AUTHOR, 
        machine_support_labels, 
        machine_queries_labels, 
        support_to_queries, 
        max_trials,
        mode,
    )
    
    for index in generator:
        LM, trial, support_index, query_indices = index

        logging.info(f"***************Few Shot Trials for Model {LM}***************")
        trial_start = datetime.now()
        
        # (1) train on the query for N-epochs
        learner = maml_adapt(model, tokenized_machine_support[int(support_index)], n_inner_loop=num_few_shot_epochs)

        # (2) evaluate on the machine targets and the background set
        tokenized_machine = create_dataset(tokenized_machine_queries[query_indices])
        targets = torch.utils.data.ConcatDataset([tokenized_background, tokenized_machine])
        probs = maml_evaluate(learner, targets, batch_size=128)
        
        labels = tokenized_background['labels'].tolist() + tokenized_machine['labels'].tolist() 
        roc_metrics = calculate_roc_metrics(labels, probs)
        all_roc_metrics.append(roc_metrics)
        metrics[LM]["trial_{:03d}".format(trial)] = (roc_metrics)

        if trial % 5 == 0 :
            logging.info(f"Trial #{trial} took {datetime.now() - trial_start}")
            logging.info(f"AUC_CUTOFF {roc_metrics['roc_auc_cutoff']}")

    return metrics 

def calculate_proto_metrics(
        INDEX_TO_AUTHOR,
        machine_support_proto, machine_support_labels,
        machine_queries_proto, machine_queries_labels,
        background_proto,
        support_to_queries,
        max_trials,
        mode,
):
    metrics = defaultdict(dict)
    metrics["global"] = defaultdict(float)
    all_roc_metrics = []
    
    generator = index_generator(
        INDEX_TO_AUTHOR, 
        machine_support_labels, 
        machine_queries_labels, 
        support_to_queries, 
        max_trials,
        mode,
    )
    
    for index in generator:
        LM, trial, support_index, query_indices = index
        targets = torch.cat(
            (background_proto, machine_queries_proto[query_indices, :]),
            axis=0
        )
        if mode == "multiple_target" or mode == "multiple_target_paraphrase":
            if mode == "multiple_target_paraphrase":
                offset = machine_support_proto.shape[0] // 2
                support_index = [support_index, support_index + offset]

            distances = []
            for sindex in support_index:
                prototypes = machine_support_proto[sindex].unsqueeze(0)
                dist = distance_metric(targets, prototypes, metric="euclidean")
                dist = dist.squeeze().cpu().tolist()
                distances.append(dist)
            dist = np.vstack(distances).min(axis=0)
        else:
            query_indices = torch.LongTensor(query_indices)
            prototypes = machine_support_proto[support_index].unsqueeze(0)
            dist = distance_metric(targets, prototypes, metric="euclidean")
            dist = dist.squeeze().cpu().tolist()

        labels = [0 for _ in range(len(background_proto))] + [1 for _ in range(len(machine_queries_proto[query_indices, :]))]
        roc_metrics = calculate_roc_metrics(labels, dist)
        all_roc_metrics.append(roc_metrics)
        metrics[LM]["trial_{:03d}".format(trial)] = (roc_metrics)

    return metrics 
