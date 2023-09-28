# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

"""

Run Inference Using Pre-Trained LUAR KNN Classifier and Save Results

"""
import logging; logging.basicConfig(); logging.root.setLevel(logging.INFO)
import sys
from argparse import ArgumentParser

import torch
from transformers import set_seed

from machine_utils import (get_data, load_knn_model, load_luar_model_tokenizer,
                           vectorize, write_output)

parser = ArgumentParser()
parser.add_argument("--knn_model_path", type=str, default="./knn_model")
parser.add_argument("--data_path", type=str, default="./data")
parser.add_argument("--output_path", type=str, default="./output")
parser.add_argument("--file_name", type=str, default="test.jsonl",
                    help="file name for training data (default=test.jsonl")
parser.add_argument("--max_token_length", type=int, default=512, 
                    help="'Max tokens for tokenizer (default=512).")
parser.add_argument("--seed", type=int, default=42, 
                    help="Random seed for reproducibility.")
parser.add_argument("--batch_size", type=int, default=128, 
                    help="Number of documents to evaluate in each batch (default=128).")
parser.add_argument("--gpus", type=int, default=1, 
                    help="Number of GPUs.")
parser.add_argument("--debug", action="store_true",
                    help="if true, run on first 100 rows.")
parser.add_argument("--threshold", type=float, default=0.7,
                    help="threshold for determining machine generated (default=0.7).")
parser.add_argument("--knn_name", type=str, default="knn.pkl",
                    help="name of KNN model file (default=knn.pkl).")
                    
def main(args):
    """
    Run LUAR KNN classifier on supplied dataset.
    """
    set_seed(args.seed)
    predictions = []
    device = 'cuda' if args.gpus >= 1 and torch.cuda.is_available() else 'cpu'
    logging.info("Running on {}".format(device))
    logging.info("Loading models")
    luar_model, luar_tokenizer = load_luar_model_tokenizer(device, args.gpus)
    knn = load_knn_model(args.knn_model_path, args.knn_name)

    logging.info("Running inference")
    input_data = get_data(args.data_path, 
                        args.file_name, 
                        args.batch_size, 
                        nrows=100 if args.debug else None)
    
    for chunk in input_data:
        features = vectorize(chunk.text.tolist(), 
                            luar_tokenizer, 
                            luar_model, 
                            args.max_token_length, 
                            device)

        # Predict with KNN
        conf = knn.predict_proba(features)[:,1]

        predictions += [{'confidence': c, 'decision': 'machine' if c > args.threshold else 'human'} for c in conf]
    logging.info("Saving results to {}".format(args.output_path))
    write_output(predictions, args.output_path)
    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    sys.exit(main(args))
