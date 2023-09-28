# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

def get_data(data_path, file_name, batch_size, nrows):
    """ 
    Load JSONL file with text key. 
    """
    path = os.path.join(data_path, file_name )
    df = pd.read_json(path, lines=True, chunksize=batch_size, nrows=nrows)
    return df
    
def vectorize(texts, tokenizer, model, max_token_length, device):
    """
    Get LUAR text embeddings.
    """
    # Tokenize
    data = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_token_length, 
        return_tensors='pt')

    input_ids, attention_mask = data.values()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    B, N = input_ids.shape
    
    # Change shapes to [B, E, max_token_length] for LUAR
    input_ids = input_ids.unsqueeze(1)
    attention_mask = attention_mask.unsqueeze(1)
    embeddings = model(input_ids, attention_mask)
    return [e.tolist() for e in embeddings]

def load_luar_model_tokenizer(device, gpus=None):
    """ 
    Load LUAR checkpoint and tokenizer.
    """
    luar_model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
    luar_tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
    if gpus > 1:
        gpus = list(range(gpus))
        luar_model = torch.nn.DataParallel(luar_model, device_ids=gpus)
    luar_model.to(torch.device(device))
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
