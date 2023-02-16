# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import torch

from datasets.retrieval_dataset import RetrievalDataset
from utilities.file_utils import Utils as utils

class AmazonDataset(RetrievalDataset):
    """Torch Dataset object for Amazon Reviews
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

        if self.split == 'train' or self.params.sanity:
            filename = "train.jsonl"
        else:
            filename = 'validation_queries.jsonl' if self.is_queries else 'validation_targets.jsonl'

        filename = os.path.join(self.dataset_path, filename) + self.params.suffix
        self.load_data(filename)
    
    def __getitem__(
        self, 
        index: int
    ):        
        text = []
        for _ in range(self.num_sample_per_author):
            episode = self.sample_random_episode(index)
            text.extend(episode[self.text_key])

        author = torch.tensor([episode["author_id"] for _ in range(self.num_sample_per_author)])

        data = self.tokenize_text(text)
        if self.params.use_random_windows:
            data = self.sample_random_window(data)
        data = [d.reshape(self.num_sample_per_author, -1, self.params.token_max_length) for d in data]

        self.mask_data_bpe(data)

        return data, author





