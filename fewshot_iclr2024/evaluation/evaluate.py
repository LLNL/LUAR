# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation Script for Few-Shot Machine Text-Detection.
Modes Supported:
1. Single-Target - Given N examples of Machine X, identify more of Machine X
2. Multiple-Target - Given N examples from all Machines, identify whether the query is from a Machine
3. Multiple-Target with Paraphrases - Given N examples of Machine X, and N examples of paraphrases of Machine X, identify more of Machine X or Paraphrased Machine X
"""

import json
import logging
import os
import random
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import spacy
from termcolor import colored
from tqdm import tqdm

sys.path.append("../")
from forward_utils import extract_luar_embeddings, extract_sbert_embeddings, get_prototypes
from evaluation.fewshot_helper import *
from evaluation.zeroshot_baselines.detect_basic import detect_basic
from file_utils import Utils as utils

random.seed(43)
np.random.seed(43)

parser = ArgumentParser()
parser.add_argument("--mode", type=str, default="single_target",
                    choices=["single_target", "multiple_target", "multiple_target_paraphrase"])
parser.add_argument("--model_name", type=str, default="CRUD",
                    choices=["MUD", "CRUD", "Multi-LLM", "Multi-domain", 
                             "MAML", "PROTONET", "SBERT", "WEGMANN",
                             "roberta_finetuned", "openAI", "rank", "logrank", "entropy"])
parser.add_argument("--model_suffix", type=str, default="",
                    help="Optional model suffix to save files with.")
parser.add_argument("--dirname", type=str, default=None,
                    help="Directory containing the data to evaluate on, will ignore all other directories.")
parser.add_argument("--force_recompute", default=False, action="store_true",
                    help="Recomputes the metrics even if they exist already.")
parser.add_argument("--max_trials", type=int, default=20,
                    help="Number of trials to run for each machine query & target.")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_tokens", type=int, default=128,
                    help="Number of tokens to use for each post.")
parser.add_argument("--support_size", type=int, default=10)
parser.add_argument("--query_size", type=int, default=10)
parser.add_argument("--debug", default=False, action="store_true")
parser.add_argument("--seed", type=int, default=43)

# fast-adaptation arguments:
parser.add_argument("--num_few_shot_epochs", type=int, default=5)
parser.add_argument("--adaptation_lr", type=float, default=2e-5)

# paraphrasing arguments
parser.add_argument("--paraphrase_p", type=float, default=0.)
parser.add_argument("--paraphrase_L", type=int, default=20, choices=[20])

args = parser.parse_args()

NLP = spacy.load("en_core_web_sm")

LUAR_METHODS = ["IUR", "MUD", "CRUD", "Multi-LLM", "Multi-domain"]
PURE_EMBEDDING_METHODS = LUAR_METHODS + ["SBERT", "WEGMANN"]
FAST_ADAPTATION_METHODS = ["MAML"]
METRIC_LEARNING_METHODS = ["PROTONET"]
BASIC_METHODS = ["rank", "logrank", "entropy"]

def get_model_str():
    """Returns the model string to use for saving files.
    """
    suffix_str = f"_suffix={args.model_suffix}" if args.model_suffix else ""
    mode_str = f"_mode={args.mode}"
    paraphrase_str = f"_p={args.paraphrase_p}_L={args.paraphrase_L}" if args.paraphrase_p > 0. else ""
    model_str = args.model_name + suffix_str + mode_str + paraphrase_str
    return model_str

def build_support_to_queries_map(
    machine_support_labels, machine_queries_labels,
    support_size, queries_size
):
    """Builds a mapping from the support indices to the queries indices.
    """
    if support_size % queries_size != 0:
        logging.warning("support_size % queries_size != 0, this may cause overlap between the supports and queries sometimes.")
    if support_size < queries_size:
        raise ValueError("support_size < queries_size, this is not supported!")

    counts = np.unique(machine_queries_labels, return_counts=True)[1]
    queries_LM_start = np.insert(counts, 0, 0).cumsum().tolist()
    
    support_to_queries = {}

    last_LM_label, LM_index = 0, 0
    chunksize = support_size // queries_size
    for support_index, support_label in enumerate(machine_support_labels):
        if support_label != last_LM_label:
            last_LM_label = support_label
            LM_index = 0

        offset = LM_index * chunksize
        start = queries_LM_start[support_label] + offset
        end = start + chunksize
        # don't violate LM boundaries, or go over the number of queries
        end = min(end, queries_LM_start[support_label+1])
        
        support_to_queries[support_index] = np.arange(start, end).tolist()
        LM_index += 1

    return support_to_queries
    
def create_index_to_author(machine_episodes):
    """Creates a mapping from the index to the author.
    """
    unique = pd.unique(machine_episodes[["author_id", "author"]].values.ravel())
    keys = unique[::2]
    values = unique[1::2]
    INDEX_TO_AUTHOR = dict(zip(keys, values))
    logging.info(f"INDEX_TO_AUTHOR={INDEX_TO_AUTHOR}")
    return INDEX_TO_AUTHOR

def evaluate(dataset_dirname, model, tokenizer):
    """Evaluate the model on the dataset and return the metrics.
    """
    # example: reddit_num-tokens=512
    dataset_name = os.path.basename(dataset_dirname)
    logging.info(f"dataset_name={dataset_name}")

    nrows = 100 if args.debug else None
    nbackground = 100 if args.debug else 3000 
    logging.info("read_json(machine_episodes)")
    if args.paraphrase_p > 0.:
        me_path = os.path.join(dataset_dirname, f"machine_episodes_num-tokens={args.num_tokens}_p={args.paraphrase_p}_L={args.paraphrase_L}_episode-size={args.support_size}_support.jsonl")
        machine_support = pd.read_json(me_path, lines=True, nrows=nrows)
        machine_support_labels = np.array(machine_support.author_id.tolist())
        
        me_path = os.path.join(dataset_dirname, f"machine_episodes_num-tokens={args.num_tokens}_p={args.paraphrase_p}_L={args.paraphrase_L}_episode-size={args.support_size}_queries.jsonl")
        machine_queries = pd.read_json(me_path, lines=True, nrows=nrows)
        machine_queries_labels = np.array(machine_queries.author_id.tolist())
    else:
        me_path = os.path.join(dataset_dirname, f"machine_episodes_num-tokens={args.num_tokens}_episode-size={args.support_size}.jsonl")

        logging.info("read_json(machine_episodes)")
        machine_episodes = pd.read_json(me_path, lines=True, nrows=nrows)
        machine_support = machine_episodes.copy()
        machine_support_labels = np.array(machine_support.author_id.tolist())
        machine_queries = machine_episodes.copy()
        machine_queries_labels = np.array(machine_queries.author_id.tolist())

    logging.info("read_json(background)")
    background = pd.read_json(os.path.join(dataset_dirname, f"background_num-tokens={args.num_tokens}_episode-size={args.support_size}_num-background=3000.jsonl"), lines=True, nrows=nbackground)
    logging.info(f"background size = {len(background)}")
    
    INDEX_TO_AUTHOR = create_index_to_author(machine_queries)
    
    support_to_queries = build_support_to_queries_map(
        machine_support_labels, machine_queries_labels,
        args.support_size, args.query_size
    )

    if args.model_name in PURE_EMBEDDING_METHODS:
        # Pure Embedding Methods
        extract_fn = extract_luar_embeddings if args.model_name in LUAR_METHODS else extract_sbert_embeddings
        machine_support_embeddings = extract_fn(machine_support.syms.tolist(), model, tokenizer, 512, batch_size=args.batch_size)
        if args.mode == "multiple_target_paraphrase":
            machine_support_paraphrase_embeddings = extract_fn(machine_support.syms_paraphrase.tolist(), model, tokenizer, 512, batch_size=args.batch_size)
            machine_support_embeddings = np.concatenate([machine_support_embeddings, machine_support_paraphrase_embeddings], axis=0)
        machine_queries_embeddings = extract_fn(machine_queries.syms.tolist(), model, tokenizer, 512, batch_size=args.batch_size)
        background_embeddings = extract_fn(background.syms.tolist(), model, tokenizer, 512, batch_size=args.batch_size)

        logging.info("calculate_nn_metrics")
        metrics = calculate_nn_metrics(
            INDEX_TO_AUTHOR,
            machine_support_embeddings, machine_support_labels,
            machine_queries_embeddings, machine_queries_labels,
            background_embeddings,
            support_to_queries,
            args.max_trials,
            args.mode,
        )
        
    elif args.model_name in FAST_ADAPTATION_METHODS:

        logging.info(args.model_name)
        concatenated_background = background.syms.apply(lambda x: ' '.join(x)).tolist()
        concatenated_machine_queries = machine_queries.syms.apply(lambda x: ' '.join(x)).tolist()
        metrics = calculate_adaptation_metrics(
            INDEX_TO_AUTHOR,
            model, tokenizer,
            machine_support, machine_support_labels,
            concatenated_machine_queries, machine_queries_labels,
            concatenated_background, 
            support_to_queries,
            max_trials=args.max_trials,
            num_few_shot_epochs=args.num_few_shot_epochs,
            mode=args.mode,
        )
        
    elif args.model_name in METRIC_LEARNING_METHODS:
        
        logging.info(args.model_name)

        tokenized_background = tokenize_dataset(background.syms, 0, model, tokenizer, concat=False)
        tokenized_machine_support = tokenize_dataset(machine_support.syms, 1, model, tokenizer, concat=False)
        tokenized_machine_queries = tokenize_dataset(machine_queries.syms, 1, model, tokenizer, concat=False)
        if args.mode == "multiple_target_paraphrase":
            tokenized_machine_support_paraphrase = tokenize_dataset(machine_support.syms_paraphrase, 1, model, tokenizer, concat=False)

        background_proto = get_prototypes(model, tokenized_background, args.batch_size)
        machine_support_proto = get_prototypes(model, tokenized_machine_support, args.batch_size)
        machine_queries_proto = get_prototypes(model, tokenized_machine_queries, args.batch_size)

        if args.mode == "multiple_target_paraphrase":
            machine_support_paraphrase_proto = get_prototypes(model, tokenized_machine_support_paraphrase, args.batch_size)
            machine_support_proto = torch.cat([machine_support_proto, machine_support_paraphrase_proto], dim=0)
        
        metrics = calculate_proto_metrics(
            INDEX_TO_AUTHOR,
            machine_support_proto, machine_support_labels,
            machine_queries_proto, machine_queries_labels,
            background_proto,
            support_to_queries,
            args.max_trials,
            args.mode,
        )
        
    elif args.model_name in BASIC_METHODS:
        logging.info(args.model_name)
        
        concatenated_machine_episodes = machine_queries.syms.apply(lambda x: ' '.join(x)).tolist()
        machine_episode_scores = [
            detect_basic(episode, model, tokenizer, method_name=args.model_name) for episode in tqdm(concatenated_machine_episodes)
        ]
        machine_episode_scores = np.array(machine_episode_scores)
        
        concatenated_background = background.syms.apply(lambda x: ' '.join(x)).tolist()
        background_scores = [
            detect_basic(episode, model, tokenizer, method_name=args.model_name) for episode in tqdm(concatenated_background)
        ]
        background_scores = np.array(background_scores)

        machine_mask = ~np.isnan(machine_episode_scores)
        machine_episode_scores = machine_episode_scores[machine_mask]
        machine_queries_labels = machine_queries_labels[machine_mask]
        background_scores = background_scores[~np.isnan(background_scores)]
        
        logging.info("calculate_detector_metrics")
        metrics = calculate_detector_metrics(
            INDEX_TO_AUTHOR,
            machine_queries_labels, 
            machine_episode_scores, background_scores
        )
        
    else:
        # RoBERTa and OpenAI

        logging.info(f"{args.model_name}_inference(machine_episodes)")
        concatenated_machine_episodes = machine_queries.syms.apply(lambda x: ' '.join(x)).tolist()
        concatenated_machine_episodes = model.create_dataset({"text": concatenated_machine_episodes}, tokenizer)
        _, machine_episodes_probs = model.evaluate(concatenated_machine_episodes, args.batch_size)
        machine_episodes_probs = np.array(machine_episodes_probs)
        
        logging.info(f"{args.model_name}_inference(background)")
        concatenated_background = background.syms.apply(lambda x: ' '.join(x)).tolist()
        concatenated_background = model.create_dataset({"text": concatenated_background}, tokenizer)
        _, background_probs = model.evaluate(concatenated_background, args.batch_size)
        background_probs = np.array(background_probs)
        
        logging.info("calculate_detector_metrics")
        metrics = calculate_detector_metrics(
            INDEX_TO_AUTHOR,
            machine_queries_labels, 
            machine_episodes_probs, background_probs
        )

    return metrics

def main():
    if args.model_name == "MAML":
        logging.info("REMEMBER: when using MAML make sure that the adaptation_lr and num_fewshot_epochs is correct!")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.adaptation_lr)

    directories = os.listdir(utils.fewshot_data_path) if args.dirname is None else [args.dirname]
    for dirname in directories:

        logging.info(f"dirname={dirname}")
        dataset_dirname = os.path.join(utils.fewshot_data_path, dirname)
        if not os.path.isdir(dataset_dirname) or "." in dirname:
            continue

        model_str = get_model_str()
        print(colored(f"model_str={model_str}", "yellow"))
        metrics_fname = os.path.join(
            dataset_dirname, 
            f"metrics_{model_str}_{args.support_size}_{args.query_size}_{args.num_tokens}.json"
        )
        metrics_fname += ".debug" if args.debug else ""

        logging.info(f"metrics_fname: {metrics_fname}")
        if os.path.exists(metrics_fname) and not args.force_recompute:
            continue

        metrics = evaluate(dataset_dirname, model, tokenizer)
        with open(metrics_fname, "w") as fout:
            json.dump(metrics, fout, indent=4)

        if args.debug:
            break

    return 0

if __name__ == "__main__":
    assert args.paraphrase_p >= 0. and args.paraphrase_p <= 1.
    assert args.paraphrase_L in [20]

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler()]
    )

    for k, v in vars(args).items():
        logging.info(f"{k}={v}")

    sys.exit(main())