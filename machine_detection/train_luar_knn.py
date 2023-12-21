# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

"""

Train LUAR KNN
Optionally, perform grid search for k

"""
import logging; logging.basicConfig(); logging.root.setLevel(logging.INFO)
import os
import pickle
import sys
from argparse import ArgumentParser

import torch
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from transformers import set_seed

from machine_utils import create_batches, extract_luar_embeddings, get_data, load_luar_model_tokenizer

parser = ArgumentParser()
parser.add_argument("--knn_model_path", type=str, default="./knn_model")
parser.add_argument("--data_path", type=str, default="./data")
parser.add_argument("--file_name", type=str, default="machine_gen_dataset_train.jsonl",
                    help="file name for training data (default=machine_gen_dataset_train.jsonl) ")
parser.add_argument("--max_token_length", type=int, default=512, 
                    help="Max tokens for tokenizer (default=512).")
parser.add_argument("--seed", type=int, default=42, 
                    help="Random seed for reproducibility.")
parser.add_argument("--debug", action="store_true",
                    help="if true, run on first 100 rows.")
parser.add_argument("--grid_search", action="store_true",
                    help="Whether to perform grid search for k (default=False).")
parser.add_argument("--n_neighbors", type=int, default=5,
                help="Number of neighbors for KNN classifier (default=5).")
parser.add_argument("--max_batch_size", type=int, default=128, 
                    help="maximum batch size (default 128).")
parser.add_argument("--max_n_episodes", type=int, default=20,
                    help="max number of subsections of 512 tokens for a single document (default=20).")

def main(args):
    """
    Train KNN Model.
    """
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Running on {}".format(device))

    logging.info("Loading models")
    luar_model, luar_tokenizer = load_luar_model_tokenizer(device)
    knn = KNeighborsClassifier(metric="cosine", n_neighbors=args.n_neighbors)

    logging.info(f"training file = {args.file_name}")
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
    labels = input_data.labels.tolist()
   
    if args.grid_search:
        logging.info("performing grid search for values of k")
        parameters = {"n_neighbors": [5, 10, 25, 50, 100]}
        knn = GridSearchCV(knn, parameters)

    logging.info("fitting KNN")
    knn.fit(luar_embeddings, labels)
    
    if args.grid_search:
        logging.info(f"best k: {str(knn.best_params_['n_neighbors'])}")

    model_name = os.path.join(args.knn_model_path, f"knn_classifier_n={args.n_neighbors}.pkl")

    logging.info(f"saving KNN classifier {model_name} here: {args.knn_model_path}")
    with open(model_name, "wb") as f:
        pickle.dump(knn, f)

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    sys.exit(main(args))
