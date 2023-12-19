# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data.dataset import Dataset

from utilities.file_utils import Utils as utils

class RetrievalDataset(Dataset):
    """This is the Super Class of all the other Dataset classes. It provides general 
       utilities for loading in the data and sampling random episodes.
    """
  
    def __init__(
        self, 
        params: argparse.Namespace, 
        split: str, 
        num_sample_per_author: int, 
        is_queries=False
    ):
        """Initializes the Dataset class.

        Args:
            params (argparse.Namespace): Command-line parameters.
            split (str): Name of the split: train, validation, or test.
            num_sample_per_author (int): Number of data points to sample per author.
            is_queries (bool, optional): Whether or not we're reading in the queries.
        """
        self.params = params
        
        self.split = split
        assert self.split in ["train", "validation", "test"]
        
        self.dataset_name = params.dataset_name
        self.num_sample_per_author = num_sample_per_author
        self.is_queries = is_queries
        self.sanity = self.params.sanity
        self.episode_length = self.params.episode_length
      
        self.dataset_path = os.path.join(utils.data_path, self.dataset_name)
        self.model_path = self.get_model_path()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path) 
        
        self.text_key = self.params.text_key
        self.time_key = self.params.time_key

    def get_model_path(self):
        transformer_modelnames = {
            "roberta": "paraphrase-distilroberta-base-v1",
            "roberta_base": "roberta-base",
        }

        return os.path.join(utils.transformer_path, transformer_modelnames[self.params.model_type])

    def tokenize_text(self, all_text):
        tokenized_episode = self.tokenizer(
            all_text, 
            padding=True if self.params.use_random_windows else "max_length", 
            truncation=False if self.params.use_random_windows else True, 
            max_length=None if self.params.use_random_windows else self.params.token_max_length, 
            return_tensors='pt'
        )
        tokenized_episode =  self.reformat_tokenized_inputs(tokenized_episode)
        
        return tokenized_episode

    def reformat_tokenized_inputs(self, tokenized_episode):
        """Reformats the output from HugginFace Tokenizers.
        """
        if len(tokenized_episode.keys()) == 3:
            input_ids, _, attention_mask = tokenized_episode.values()
            data = [input_ids, attention_mask]
        else:
            input_ids, attention_mask = tokenized_episode.values()
            data = [input_ids, attention_mask]

        return data
   
    def sample_random_episode(self, index: int, is_test=False):
        """Samples a random episode of size `episode_length`.

        Args:
            index (int): Index of the author we're sampling.
        """
        if hasattr(self, "fhandle"):
            author_data = self.read_line(self.fhandle, index)
        else:
            author_data = self.data.iloc[index].to_dict()

        num_docs = len(author_data[self.text_key])
        episode_length = num_docs if num_docs < self.episode_length else self.episode_length

        maxval = num_docs - episode_length

        if is_test:
            start_index = 0
        else:
            start_index = random.randint(0, maxval)
    
        episode = {k: v[start_index: start_index + episode_length] 
                   for k, v in author_data.items() if isinstance(v, list) and len(v) > 0}

        if self.sanity or "author_id" not in author_data:
            episode["author_id"] = index
        else:            
            episode["author_id"] = author_data["author_id"] 
            
        return episode
    
    def sample_random_window(self, data, window_length=32):
        """Samples a random window from the text.
        """
        input_ids, attention_mask = data

        cls = self.tokenizer.cls_token_id
        pad = self.tokenizer.pad_token_id
        eos = self.tokenizer.eos_token_id
        if type(eos) != int:
            eos = self.tokenizer.sep_token_id

        # Inputs are smaller than window size -> add padding
        padding = window_length - input_ids.shape[1]
        if padding > 0:
            input_ids = F.pad(input_ids, (0, padding), 'constant', pad) 
            attention_mask = F.pad(attention_mask, (0, padding), 'constant', 0) 
            return [input_ids, attention_mask]

        # Inputs are larger than window size -> sample random windows
        true_lengths = torch.sum(torch.where(input_ids != 1, 1, 0), 1)
        start_indices = torch.tensor([random.randint(1, l - window_length + 2) if l >= window_length else 1 for l in true_lengths])
        indices = torch.tensor([list(range(start, start + window_length - 2)) for start, l in zip(start_indices, true_lengths)])
        input_ids = input_ids.gather(1, indices)
        attention_mask = attention_mask.gather(1, indices)
        
        # Add cls token
        input_ids = F.pad(input_ids, (1, 0), 'constant', cls)
        attention_mask = F.pad(attention_mask, (1, 0), 'constant', 1)
        
        # Add eos token
        input_ids = torch.cat((input_ids, torch.where(true_lengths >= window_length, eos, pad).unsqueeze(1)), 1)
        attention_mask = torch.cat((attention_mask, torch.where(true_lengths >= window_length, 1, 0).unsqueeze(1)), 1)

        return [input_ids, attention_mask]

    def mask_data_bpe(self, data):
        """Masks x% of Byte-Pair Encodings from the input. 
        """
        if self.params.mask_bpe_percentage > 0.0:
            mask = torch.rand(data[0].size()) >= (1. - self.params.mask_bpe_percentage)

            # This is why we won't quite get to the mask percentage asked for by the user.
            pad_mask = ~(data[0] == self.tokenizer.pad_token_id)
            mask *= pad_mask

            data[0].masked_fill_(mask, self.tokenizer.mask_token_id)

        return data

    def load_data(self, filename: str, build_byte_count_list=False):
        """Loads in the data specified in `filename` and populates the necessary 
           variables for sampling the dataset.

        Args:
            filename (str): JSON file to load.
        """
        query_file = 'query' if self.is_queries else 'targets'
        print("Loading {} dataset {} {} file: {}".format(
              self.dataset_name, self.split, query_file, filename))

        if self.params.dataset_name in ["iur_dataset", "raw_all"] or build_byte_count_list:
            self.build_byte_count_list(filename, load_first_N=self.params.sanity)
            self.num_authors = len(self.byte_count_list)
        else: 
            self.data = pd.read_json(filename, lines=True, nrows=self.params.sanity)
            self.num_authors = len(self.data)
        
    def build_byte_count_list(self, filename: str, load_first_N: int):
        """Builds a list where each element contains the number of bytes for 
           that particular line.
        """
        byte_count_list = []

        with open(filename, 'r') as fhandle:
            i = 0
            line = fhandle.readline()

            while line != "":
                if load_first_N is not None and i > load_first_N - 1:
                    break

                byte_count_list.append(len(line))
                line = fhandle.readline()
                i += 1

        self.byte_count_list = byte_count_list

    def read_line(self, fhandle, index):
        """Reads one line from the filehandle provided. 
           Assumes that build_byte_count_list() has already been called.
        """
        num_bytes_to_seek = sum(self.byte_count_list[:index])
        num_bytes_to_read = self.byte_count_list[index]

        fhandle.seek(0)
        fhandle.seek(num_bytes_to_seek)
        line = str(fhandle.read(num_bytes_to_read))
        json_line = json.loads(line)

        return json_line

    def __len__(self):
      return self.num_authors

    def __getitem__(self, index: int):
        """Returns the item located at `index` from the dataset.
           Must be implemented by the sub-classes.
        """
        raise NotImplementedError
