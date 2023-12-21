# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

"""

Run Inference Using Pre-Trained LUAR KNN Classifier and Save Results.

"""
import logging; logging.basicConfig(); logging.root.setLevel(logging.INFO)
import sys
from argparse import ArgumentParser

import torch
from transformers import set_seed

from machine_utils import (create_batches, extract_luar_embeddings, 
                           get_data, load_knn_model, 
                           load_luar_model_tokenizer, write_output)
parser = ArgumentParser()
parser.add_argument("--knn_model_path", type=str, default="./knn_model")
parser.add_argument("--data_path", type=str, default="./data")
parser.add_argument("--output_path", type=str, default="./output")
parser.add_argument("--file_name", type=str, default="test.jsonl",
                    help="file name for test data (default=test.jsonl")
parser.add_argument("--max_token_length", type=int, default=512, 
                    help="Max tokens for tokenizer (default=512).")
parser.add_argument("--seed", type=int, default=42, 
                    help="Random seed for reproducibility.")
parser.add_argument("--debug", action="store_true",
                    help="if true, run on first 100 rows.")
parser.add_argument("--threshold", type=float, default=0.7,
                    help="threshold for determining machine generated (default=0.7).")
parser.add_argument("--knn_name", type=str, default="knn.pkl",
                    help="name of KNN model file (default=knn.pkl).")
parser.add_argument("--max_batch_size", type=int, default=128, 
                    help="maximum batch size (default 128).")
parser.add_argument("--max_n_episodes", type=int, default=20,
                    help="max number of subsections of 512 tokens for a single document (default=20).")

def main(args):
    """
    Run LUAR KNN classifier on supplied test dataset.
    """
    set_seed(args.seed)
    predictions = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Running on {}".format(device))
    
    logging.info("Loading models")
    luar_model, luar_tokenizer = load_luar_model_tokenizer(device)
    knn = load_knn_model(args.knn_model_path, args.knn_name)

    input_data = get_data(args.data_path, 
                        args.file_name, 
                        nrows=100 if args.debug else None)
    tokenized_text = luar_tokenizer(input_data.text.tolist())

    logging.info("Batching documents")
    batches = create_batches(tokenized_text,
                             args.max_n_episodes,
                             args.max_batch_size,
                             device,
                             embedding_size = luar_tokenizer.model_max_length)

    logging.info("Getting LUAR features")
    luar_embeddings = extract_luar_embeddings(batches, luar_model)

    logging.info("Running Inference")
    conf = knn.predict_proba(luar_embeddings)[:,1]
    predictions += [{"confidence": c, "decision": "machine" if c > args.threshold else "human"} for c in conf]

    logging.info("Saving results to {}".format(args.output_path))
    write_output(predictions, args.output_path)

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    sys.exit(main(args))
