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
from tqdm import tqdm
from transformers import set_seed

from machine_utils import get_data, load_luar_model_tokenizer, vectorize

parser = ArgumentParser()
parser.add_argument("--knn_model_path", type=str, default="./knn_model")
parser.add_argument("--data_path", type=str, default="./data")
parser.add_argument("--n_neighbors", type=int, default=5, 
                    help="'Number of neighbors for KNN classifier (default=5).")
parser.add_argument("--max_token_length", type=int, default=512, 
                    help="'Max tokens for tokenizer (default=512).")
parser.add_argument("--seed", type=int, default=42, 
                    help="Random seed for reproducibility.")
parser.add_argument("--batch_size", type=int, default=128, 
                    help="Number of documents to evaluate in each batch (default=128).")
parser.add_argument("--gpus", type=int, default=1, 
                    help="Number of GPUs (default=1).")
parser.add_argument("--grid_search", action="store_true",
                    help="Whether to perform grid search for k (default=False).")
parser.add_argument("--debug", action="store_true",
                    help="if true, run on first 100 rows (default False).")
parser.add_argument("--file_name", type=str, default="machine_gen_dataset_train.jsonl",
                    help="file name for training data (default=machine_gen_dataset_train.jsonl")
parser.add_argument("--luar_name", type=str, default="LUAR.pth",
                    help="Name of LUAR model file (default=LUAR.pth).")

def main(args):
    """
    Train KNN Model.
    """
    set_seed(args.seed)
    device = 'cuda' if args.gpus >= 1 and torch.cuda.is_available() else 'cpu'
    logging.info(f'Using {device}')
    logging.info('loading LUAR model and tokenizer')
    luar_model, luar_tokenizer = load_luar_model_tokenizer(device, args.gpus)
    knn_classifier = KNeighborsClassifier(metric='cosine', n_neighbors=args.n_neighbors)

    if args.grid_search:
        logging.info('performing grid search for values of k')
        parameters = {'n_neighbors': [5, 10, 25, 50, 100]}
        knn_classifier = GridSearchCV(knn_classifier, parameters)

    logging.info('loading and vectorizing data')
    train_features = []
    train_labels = []
    data = get_data(args.data_path, 
                    args.file_name, 
                    batch_size=args.batch_size, nrows=100 if args.debug else None)

    for chunk in data:
        train_features += vectorize(chunk['text'].tolist(), 
                                    luar_tokenizer, 
                                    luar_model, 
                                    args.max_token_length, 
                                    device)

        train_labels.extend(chunk['labels'].tolist())

    logging.info('fitting KNN')
    knn_classifier.fit(train_features, train_labels)
    
    if args.grid_search:
        logging.info(f'best k: {knn_classifier.best_params_}')

    model_name = os.path.join(args.knn_model_path, f'luar_classifier_n={args.n_neighbors}.pkl')
    with open(model_name, 'wb') as f:
        pickle.dump(knn_classifier, f)


if __name__ == "__main__":
    args = parser.parse_args()
    sys.exit(main(args))
