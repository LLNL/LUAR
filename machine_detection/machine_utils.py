# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import pickle

import pandas as pd
import torch
from collections import defaultdict
from termcolor import colored
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def create_batches(tokenized_data, max_n_episodes, max_batch_size, device, embedding_size=512):
    """ 
    Create batches based on document length. 
    Restrict batches to max_batch_size and documents to max_n_episodes.
    """
    print(colored("Restricting documents to a maximum of {} episode(s) of {} tokens (total = {} tokens)".format(max_n_episodes, embedding_size, (max_n_episodes*embedding_size)), "yellow"))
    batches = defaultdict(list)
    for i in range(len(tokenized_data["input_ids"])):
        input_ids = tokenized_data["input_ids"][i]
        attention_mask = tokenized_data["attention_mask"][i]
        remainder = len(input_ids) % embedding_size
        if remainder != 0:
            input_ids += [1] * (embedding_size - remainder)
            attention_mask += [0] * (embedding_size - remainder)

        n_episodes = len(input_ids) // embedding_size
        if n_episodes > max_n_episodes:
            n_episodes = max_n_episodes
            input_ids = input_ids[:n_episodes * embedding_size]
            attention_mask = attention_mask[:n_episodes * embedding_size]
        batches[n_episodes].append((input_ids, attention_mask))

    for k, v in batches.items():
        B = len(v)
        input_ids = torch.LongTensor([x[0] for x in v]).reshape(B, k, embedding_size).to(device).split(max_batch_size)
        attention_mask = torch.LongTensor([x[1] for x in v]).reshape(B, k, embedding_size).to(device).split(max_batch_size)
        batches[k] = [{"input_ids": i, "attention_mask": a} for i,a in zip(input_ids, attention_mask)]
    return batches

def extract_luar_embeddings(batches, luar_model):
    """
    Extract LUAR embeddings to use as features in KNN.
    """
    embeddings = []
    batches = [b for batch in batches.values() for b in batch]
    for batch in tqdm(batches):
        embedding = luar_model(**batch).detach().cpu().numpy()
        embeddings.extend(embedding)
    return embeddings

def get_data(data_path, file_name, nrows):
    """ 
    Load JSONL file with text key. 
    """
    path = os.path.join(data_path, file_name )
    df = pd.read_json(path, lines=True, nrows=nrows)
    return df

def load_luar_model_tokenizer(device):
    """ 
    Load LUAR checkpoint and tokenizer.
    """
    luar_model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
    luar_tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
    if torch.cuda.is_available():
        luar_model = torch.nn.DataParallel(luar_model, device_ids= [i for i in range(torch.cuda.device_count())])
    luar_model.to(torch.device(device))
    luar_model.eval()
    return luar_model, luar_tokenizer

def load_knn_model(model_path, knn_filename):
    """ 
    Load pre-trained KNN model. 
    """
    with open(os.path.join(model_path, knn_filename), 'rb') as k:
        knn = pickle.load(k)
    return knn

def write_output(predictions, output_path):
    """
    Save prediction output.
    """
    predictions = pd.DataFrame(predictions)
    predictions.to_json(os.path.join(output_path, "predictions.jsonl"), orient='records', lines=True)
