# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import random

from torch.utils.data.dataset import Dataset

from datasets.utils import get_dataset

class Multidomain_Dataset(Dataset):
    """Dataset Object for the Multi-Domain training.

       Expects dataset name to be: "dataset_1+dataset_2" and for there 
       only to be two datasets.
    """

    # TODO: remove num_sample_per_author from everything
    def __init__(self, params, dataset_split_name, is_queries=True):
        """`dataset` parameter is not used.
        """

        self.params = params
        self.dataset_split_name = dataset_split_name
        self.num_sample_per_author = self.params.num_sample_per_author

        # ensure correct input to dataset_name: "dataset_1+dataset_2"
        dataset_names = self.params.dataset_name.split("+")
        print(self.params.dataset_name)
        assert len(dataset_names) == 2
        
        only_queries = (is_queries == True) and (dataset_split_name in ["validation", "test"])
        only_targets = (is_queries == False) and (dataset_split_name in ["validation", "test"])

        # HACK: temporarily modifying cmd-line arguments to load datasets
        params.dataset_name = dataset_names[0]
        self.dataset_1 = get_dataset(params, dataset_split_name, only_queries, only_targets)
        params.dataset_name = dataset_names[1]
        self.dataset_2 = get_dataset(params, dataset_split_name, only_queries, only_targets)
        self.params.dataset_name = "+".join(dataset_names)
        
        self.multidomain_prob = 0.5
        if self.params.multidomain_prob is not None:
            assert self.params.multidomain_prob > 0.0 and self.params.multidomain_prob < 1.0
            self.multidomain_prob = self.params.multidomain_prob

    def __len__(self):
        return len(self.dataset_1) + len(self.dataset_2)

    def __getitem__(self, index):
        """`index` will be ignored here in order to ensure 50/50 sampling rate.
        """
        if self.dataset_split_name in ["validation", "test"]:
            data = self.sample_val_or_test(index)
        else:
            data = self.sample_training()

        return data

    def sample_val_or_test(self, index):
        if index >= len(self.dataset_1):
            text, author = self.dataset_2[index - len(self.dataset_1)]
            # make sure to offset the author_label by len(self.dataset_1)
            author += len(self.dataset_1)
            to_return = [text, author]
        else:
            text, author = self.dataset_1[index]
            to_return = [text, author]
        
        return to_return

    def sample_training(self):
        to_return = []
        
        if random.random() > self.multidomain_prob:
            new_index = random.randint(0, len(self.dataset_1) - 1)
            text, author = self.dataset_1[new_index]
            data = [text, author]
            to_return.extend(data)
        else:
            new_index = random.randint(0, len(self.dataset_2) - 1)
            # make sure to offset the author_label by len(self.dataset_1)
            text, author = self.dataset_2[new_index]
            author += len(self.dataset_1)
            data = [text, author]
            to_return.extend(data)

        return to_return