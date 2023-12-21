# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import torch

from datasets.retrieval_dataset import RetrievalDataset
from utilities.file_utils import Utils as utils

class Short_Stories_Dataset(RetrievalDataset):
    """Torch Dataset object for the PAN Short Stories dataset
    """
    def __init__(
        self, 
        params: argparse.Namespace, 
        split: str, 
        num_sample_per_author: int, 
        is_queries=True
    ):            
        super().__init__(params, split, num_sample_per_author, is_queries)

        self.dataset_path = utils.path_exists(os.path.join(utils.data_path, self.dataset_name))
         
        if split == 'train' or params.sanity:
            filename = 'train_raw.jsonl'
        else:
            filename = 'queries_raw.jsonl' if self.is_queries else 'targets_raw.jsonl'

        filename = os.path.join(self.dataset_path, filename) + self.params.suffix
        self.load_data(filename)
        self.is_test = split != "train"

    def __getitem__(
        self, 
        index: int
    ):        
        if self.split == "test":
            # During test time, we read in every paragraph from the short story, 
            # tokenize it, and take the first 32 subwords from each:
            # author_data = self.read_line(self.fhandle, index)
            author_data = self.data.iloc[index].to_dict()
            
            tokenized_episode = self.tokenizer(
                author_data[self.text_key], 
                padding="max_length", 
                truncation=True, 
                max_length=self.params.token_max_length, 
                return_tensors='pt'
            )
            data = self.reformat_tokenized_inputs(tokenized_episode)
            
            data = [d.reshape(1, -1, self.params.token_max_length) for d in data]
            author = torch.tensor([author_data["author_id"] for _ in range(self.num_sample_per_author)])
        else:
            text = []
            
            for _ in range(self.num_sample_per_author):
                episode = self.sample_random_episode(index, is_test=self.is_test)
                text.extend(episode[self.text_key])
                    
            author = torch.tensor([episode["author_id"] for _ in range(self.num_sample_per_author)])
            
            data = self.tokenize_text(text)
            if self.params.use_random_windows:
                data = self.sample_random_window(data)
            
            data = [d.reshape(self.num_sample_per_author, -1, self.params.token_max_length) for d in data]

        self.mask_data_bpe(data)
        return data, author
