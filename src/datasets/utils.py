# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from datasets.amazon_dataset import AmazonDataset
from datasets.reddit_dataset import RedditDataset
from datasets.short_stories_dataset import Short_Stories_Dataset

DNAME_TO_CLASS = {
    'iur_dataset': RedditDataset,
    'raw_all': RedditDataset,
    'raw_amazon': AmazonDataset,
    'pan_paragraph': Short_Stories_Dataset,
}

def get_dataset(
    params: argparse.Namespace, 
    split: str, 
    only_queries=False, 
    only_targets=False
):
    """Returns the appropriate Torch Dataset object for the dataset
       specified through the command-line.

    Args:
        params (argparse.Namespace): Command-line arguments.
        split (str): Name of the split: train, validation, or test.
        only_queries (bool, optional): Only read the queries. Defaults to False.
        only_targets (bool, optional): Only read the targets. Defaults to False.
    """
    assert split in ["train", "validation", "test"]
    
    if split == "train":
        return get_train_dataset(params)
    else:
        return get_val_or_test_dataset(params, split, only_queries, only_targets)

def get_train_dataset(
    params: argparse.Namespace
):
    """Returns the training dataset as a Torch object.

    Args:
        params (argparse.Namespace): Command-line arguments.
    """
    num_sample_per_author = params.num_sample_per_author
    dataset_class = DNAME_TO_CLASS[params.dataset_name]

    train_dataset = dataset_class(params, 'train', num_sample_per_author)
    return train_dataset

def get_val_or_test_dataset(
    params, 
    split, 
    only_queries=False, 
    only_targets=False
):
    """Returns the validation or test dataset as a Torch object.

    Args:
        params (argparse.Namespace): Command-line arguments.
        split (str): Name of the split: train, validation, or test.
        only_queries (bool, optional): Only read the queries. Defaults to False.
        only_targets (bool, optional): Only read the targets. Defaults to False.
    """
    dataset_class = DNAME_TO_CLASS[params.dataset_name]
    assert (only_queries == False and only_targets == False) or (only_queries ^ only_targets), "specified both only_queries=True and only_targets=True"

    if only_queries:
        queries = dataset_class(params, split, num_sample_per_author=1, is_queries=True)
        return queries
    if only_targets:
        targets = dataset_class(params, split, num_sample_per_author=1, is_queries=False)
        return targets
    
    queries = dataset_class(params, split, num_sample_per_author=1, is_queries=True)
    targets = dataset_class(params, split, num_sample_per_author=1, is_queries=False)
    return queries, targets
