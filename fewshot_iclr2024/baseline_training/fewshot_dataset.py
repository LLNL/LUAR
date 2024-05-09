# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import random
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import AutoTokenizer

sys.path.append("../")
from file_utils import Utils as utils

def toklen(text_batch, tokenizer):
    tokenized = tokenizer(text_batch, add_special_tokens=True)["input_ids"]
    return [len(x) for x in tokenized]

def toklen_all(all_text, num_workers=40, batch_size=1000):
    # gpt2-large so that we can match the distribution of the prompts
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    batches = [all_text[i:i+batch_size].tolist() for i in range(0, len(all_text), batch_size)]
    with Pool(num_workers) as p:
        fn = partial(toklen, tokenizer=tokenizer)
        lengths = p.map(fn, batches)
    lengths = np.array([l for sublist in lengths for l in sublist])
    return lengths

def filter_by_length(text, labels):
    human_lengths = toklen_all(text[labels == 0], num_workers=40)
    mask = human_lengths >= 60
    text = np.concatenate([text[labels == 0][mask], text[labels != 0]])
    labels = np.concatenate([labels[labels == 0][mask], labels[labels != 0]])
    return text, labels

def read_reddit_data():
    path = "/data1/yubnub/data/reddit_machine_prompting_may23/meta_learning/reddit_smalltopics.jsonl"
    df = pd.read_json(path, lines=True)
    text = np.array(df.syms.tolist())
    labels = np.zeros(len(text))
    return text, labels

def tokenize(text, tokenizer, max_token_length=512):
    """Tokenize text using the tokenizer. 
    """
    text = text if isinstance(text, list) else [text]
    max_length = max_token_length
    tokenized_data = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length, 
        return_tensors="pt"
    )
    return tokenized_data

class MetaDataset(Dataset):
    """Meta dataset for episodic training.
    
       Every __getitem__ call will return a task, which in this case 
       is a single LM classification problem.
    """
    def __init__(
        self, 
        tokenizer,
        split="train", 
        model_to_label=None,
        k_support=16,
        k_query=16,
        num_tokens=64,
        filter_lengths=True,
        use_many_topics=False,
    ):
        assert split in ["train", "valid"]

        dataset = load_from_disk(utils.aac_data_path)

        if split == "train":
            text = np.array(dataset["train"]["text"])
            unique_models = np.unique(dataset["train"]["model"])
            model_to_label = {model: i for i, model in enumerate(unique_models)}
            labels = np.array([model_to_label[model] for model in dataset["train"]["model"]])
            
        elif split == "valid":
            if model_to_label is None:
                print("model_to_label=None when split=\"valid\"")
                sys.exit(1)
            
            text = np.array(dataset["valid"]["text"])
            labels = np.array([model_to_label[model] for model in dataset["valid"]["model"]])
            
        if filter_lengths:
            text, labels = filter_by_length(text, labels)
            
        self.batch_composition = ["H", "M"]
        self.labels = labels
        self.model_to_label = model_to_label
        self.num_tokens = num_tokens
        self.k_support = k_support
        self.k_query = k_query
        self.text = text
        self.tokenizer = tokenizer
        
        if use_many_topics:
            varied_text, varied_labels = read_reddit_data()
            mask = self.labels > 0
            self.text = np.concatenate([self.text[mask], varied_text])
            self.labels = np.concatenate([self.labels[mask], varied_labels])
            
    def __getitem__(self, task_id):
        human_id = 0
        task_id += 1 # 0 is reserved for human-written text

        samples, labels = [], []
        for cls in self.batch_composition:
            # k_support samples for support set
            # k_query samples for query set
            if cls == "H":
                # sample humans
                indices = np.where(self.labels == human_id)[0]
                samples.extend(random.sample(self.text[indices].tolist(), self.k_support + self.k_query))
                labels.extend([0 for _ in range(len(samples))])
            else:
                # sample machines
                indices = np.where(self.labels == task_id)[0]
                samples.extend(random.sample(self.text[indices].tolist(), self.k_support + self.k_query))
                labels.extend([1 for _ in range(len(samples))])
                
        if len(self.batch_composition) == 2:
            # https://stackoverflow.com/questions/7946798/interleave-multiple-lists-of-the-same-length-in-python
            # interleave the lists so that batches are composed of both humans and machines:
            a = list(range(len(samples) // 2))
            b = list(range(len(samples) // 2, len(samples)))
            c = list(range(len(samples)))
            c[::2] = a
            c[1::2] = b
            samples = [samples[i] for i in c]
            labels = [labels[i] for i in c]

        # split into the support and query sets
        N = self.k_support * len(self.batch_composition)
        X_support, X_query = samples[:N], samples[N:]
        y_support, y_query = np.array(labels[:N]), np.array(labels[N:])
        
        X_support = tokenize(X_support, self.tokenizer, self.num_tokens)
        X_query = tokenize(X_query, self.tokenizer, self.num_tokens)

        return X_support, y_support, X_query, y_query
    
    def __len__(self):
        # -1 because we don't want to count the human-written text
        return len(np.unique(self.labels)) - 1