# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import time


def create_argument_parser():
    """Defines a parameter parser for all of the arguments of the application.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ###### MISC parameters ##########
    parser.add_argument("--dataset_name", type=str, default="raw_all",
                        help="Specifies which dataset to use, see README for options")
    parser.add_argument("--experiment_id", type=str, default="{}".format(int(time.time()),
                        help="Experiment identifier for an experiment group"))
    parser.add_argument("--version", type=str, default=None,
                        help="PyTorch Lightning's folder version name.")
    parser.add_argument("--log_dirname", type=str, default='lightning_logs',
                        help="Name to assign to the log directory")
    parser.add_argument("--model_type", type=str, default="roberta",
                        choices=["roberta", "roberta_base"],
                        help="Specifies which Transformer backbone to use")
    parser.add_argument("--text_key", type=str, default="syms",
                       help="Dictionary key name where the text is located in the data")
    parser.add_argument("--time_key", type=str, default="hour",
                       help="Dictionary key name where the text is located in the data")
    parser.add_argument("--do_learn", action='store_true', default=False,
                        help="Whether or not to train on the training set")
    parser.add_argument("--validate", action='store_true', default=False,
                        help="Whether or not to validate on the dev set")
    parser.add_argument("--evaluate", action='store_true', default=False,
                        help="Whether or not to evaluate on the test set")
    parser.add_argument("--validate_every", type=int, default=5,
                        help="Validate every N epochs")
    parser.add_argument("--sanity", type=int, default=None,
                        help="Subsamples N authors from the dataset, used for debugging")
    parser.add_argument("--random_seed", type=int, default=777,
                        help="Seed for PyTorch and NumPy random operations")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use for training")
    parser.add_argument("--period", type=int, default=5,
                        help="Periodicity to save checkpoints when not validating")
    parser.add_argument("--suffix", default="", type=str,
                        help="Suffix to indicate which data files to load")

    ##### Training parameters ##########
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Specifies learning rate")
    parser.add_argument("--learning_rate_scaling", action="store_true", default=False,
                        help="Toggles variance-based learning rate scaling")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Number of authors to include in each batch")
    parser.add_argument("--load_checkpoint", default=False, action = 'store_true', 
                        help="If True, will load the latest checkpoint")
    parser.add_argument("--precision", default=16, type=int,
                        help="Precision of model weights")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of workers to prefetch data")
    parser.add_argument("--num_epoch", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--pin_memory", action='store_true', default=False,
                        help="Used pin memory for prefetching data")
    parser.add_argument("--gradient_checkpointing", default=False, action="store_true",
                        help="If True, activates Gradient Checkpointing")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Temperature to use for SupCon")
    parser.add_argument("--multidomain_prob", default=None, type=float,
                        help="Sampling probability for the Multi-Domain dataset")
    parser.add_argument("--mask_bpe_percentage", default=0.0, type=float,
                        help="Approximate percentage of BPE to mask during training")
    
    ##### Model Hyperparameters #####
    parser.add_argument("--episode_length", type=int, default=16,
                        help="Number of actions to include in an episode")
    parser.add_argument("--token_max_length", type=int, default=32,
                        help="Number of tokens to take per example")
    parser.add_argument("--num_sample_per_author", type=int, default=2,
                        help="Number of episodes to sample per author during training")
    parser.add_argument("--embedding_dim", type=int, default=512,
                        help="Final output embedding dimension")
    parser.add_argument("--attention_fn_name", type=str, default="memory_efficient",
                        choices=["default", "memory_efficient"],
                        help="Which Attention mechanism to use, uses basic Self-Attention by default")
    parser.add_argument("--use_random_windows", default=False, action="store_true",
                        help="Use random windows when training")
    
    return parser.parse_args()
