# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import os
from abc import ABC, abstractmethod
from itertools import chain
from math import ceil

import numpy as np
import pytorch_lightning as pt
import torch
import torch.optim as optim
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from torch.utils.data import DataLoader

from datasets.multidomain_dataset import Multidomain_Dataset
from datasets.utils import get_dataset
from utilities import metric as M
from utilities.file_utils import Utils as utils


class LightningTrainer(pt.LightningModule, ABC):
    """Defines all the PyTorch Lightning training functions. 
       Our model (SBERT), inherits from this class.
    """
    @abstractmethod
    def _model_forward(self, batch):
        """Passes a batch of data through the model. 
           This method must be implemented within the model.
        """
        pass
    
    def __init__(self, params):
        super().__init__()
        self.params = params
    
        self.experiment_log_filename = os.path.join(
            utils.output_path, 
            'experiments.log'
        )
        
        self.loss = losses.SupConLoss(
            temperature=self.params.temperature, 
            distance=CosineSimilarity()
        )

                 
    def configure_optimizers(self):
        """Configures the LR Optimizer & Scheduler.
        """
        # Scale learning rate to preserve variance based on success of 2e-5@64 per GPU
        if self.params.learning_rate_scaling:
            lr_factor = np.power(self.params.batch_size / 32, 0.5)
        else:
            lr_factor = 1
            
        learning_rate = self.learning_rate * lr_factor
        print("Using LR: {}".format(learning_rate))
        optimizer = optim.AdamW(
            chain(self.parameters()), lr=learning_rate)
            
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=300, eta_min=0.0001)

        return [optimizer], [scheduler]

    def train_collate_fn(self, batch):
        """This function will sample a random number of episodes as per Section 2.3 of:
                https://arxiv.org/pdf/2105.07263.pdf
        """
        data, author = zip(*batch)

        author = torch.stack(author)

        # Minimum number of posts for an author history in batch
        min_posts = min([d[0].shape[1] for d in data])

        # Size of episode = R + ⌈x(S-R)⌉, x ~ Beta(3,1)
        sample_size = min(1 + ceil(np.random.beta(3, 1) *
                          15), min_posts)

        # If minimum posts < episode length, make start 0 to ensure you get all posts
        if min_posts < 16:
            start = 0
        else:
            # Pick a random start index
            start = np.random.randint(0, 16 - sample_size + 1)
    
        data = [torch.stack([f[:, start:start + sample_size, :]
                            for f in feature]) for feature in zip(*data)]

        return data, author

    def validation_collate_fn(self, batch):
        """Some validation datasets have authors with less than < 16 episodes. 
           When batching, make sure that we don't run into stacking problems. 
        """

        data, author = zip(*batch)

        author = torch.stack(author)

        # Minimum number of posts for an author history in batch
        min_posts = min([d[0].shape[1] for d in data])
        # If min_posts < episode length, need to subsample
        if min_posts < 16:
            data = [torch.stack([f[:, :min_posts, :] for f in feature])
                    for feature in zip(*data)]
        # Otherwise, stack data as is
        else:
            data = [torch.stack([f for f in feature])
                    for feature in zip(*data)]

        return data, author

    def train_dataloader(self):
        """Returns the training DataLoader.
        """        
        if "+" in self.params.dataset_name:
            train_dataset = Multidomain_Dataset(self.params, "train")
        else:
            train_dataset = get_dataset(self.params, split="train")

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            collate_fn=self.train_collate_fn
        )

        return data_loader

    def val_dataloader(self):
        """Returns the validation DataLoader.
        """
        # to counteract different episode sizes during validation / testing
        batch_size = 1 if self.params.dataset_name in ["raw_amazon", "pan_paragraph"] else self.params.batch_size
        
        if "+" in self.params.dataset_name:
            queries = Multidomain_Dataset(self.params, "validation", is_queries=True)
            targets = Multidomain_Dataset(self.params, "validation", is_queries=False)
        else:
            queries, targets = get_dataset(self.params, split="validation")

        data_loaders = [
            DataLoader(
                queries,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=self.params.pin_memory,
                num_workers=self.params.num_workers,
                collate_fn=self.validation_collate_fn
                ),
            DataLoader(
                targets,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=self.params.pin_memory,
                collate_fn=self.validation_collate_fn
                )
            ]

        return data_loaders

    def test_dataloader(self):
        # to counteract different episode sizes during validation / testing
        batch_size = 1 if self.params.dataset_name in ["raw_amazon", "pan_paragraph"] else self.params.batch_size
        
        if "+" in self.params.dataset_name:
            queries = Multidomain_Dataset(self.params, "test", is_queries=True)
            targets = Multidomain_Dataset(self.params, "test", is_queries=False)
        else:
            queries, targets = get_dataset(self.params, split="test")


        data_loader = [
            DataLoader(
                queries,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=self.params.pin_memory
                ),
            DataLoader(
                targets,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=self.params.pin_memory
                )
            ]

        return data_loader

    ##################################################
    # Training Methods
    ##################################################
    
    def _internal_step(self, batch, split_name):
        """Internal step function used by {training/validation/test}_step 
           functions.
        """
        labels = batch[-1].flatten()
        episode_embeddings, comment_embeddings = self._model_forward(batch)

        return_dict = {
            f"{split_name}embedding": episode_embeddings,
            "ground_truth": labels,
        }

        return return_dict
    
    def training_step(self, batch, batch_idx):
        """Executes one training step.
        """
        return_dict = self._internal_step(batch, split_name="")
        return return_dict

    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Executes one validation step.
        """
        return_dict = self._internal_step(batch, split_name="val_")
        return return_dict

    def test_step(self, batch, batch_idx, dataloader_idx):
        """Executes one test step.
        """
        return_dict = self._internal_step(batch, split_name="test_")
        return return_dict

    def training_step_end(self, step_outputs):
        """Calculate the loss function according to the embeddings from every step.
        """
        episode_embeddings = step_outputs["embedding"]
        labels = step_outputs["ground_truth"]

        loss = self.loss(episode_embeddings, labels)

        return_dict = {"loss": loss}

        self.log("loss", return_dict["loss"], 
                on_step=True, on_epoch=True, prog_bar=True,
                batch_size=episode_embeddings.size(0))
        
        return return_dict

    def validation_epoch_end(self, outputs):
        
        """Calculates metrics using the outputs from each validation step.

        Args:
            outputs (list): A list of lists of dicts with shape [2, num_batches] where 
                the dicts have the keys: 'val_loss', 'ground_truth' and 'validation_embedding'.
        """
        # Skip computing metrics for the validation sanity check
        if len(outputs[0]) < 3:
            return
        
        metrics = M.compute_metrics(outputs[0], outputs[1], 'val')

        for k,v in metrics.items():
            self.log('val_{}'.format(k), v, prog_bar=True, 
                    batch_size = self.params.batch_size)

    def test_epoch_end(self, outputs):
        """ Calculates metrics using the outputs from each test step.

        Args:
            outputs: A list of lists of dicts with shape [2, num_batches] where 
                the dicts have the keys: 'test_loss', 'ground_truth' and 'test_embedding'.
        """
        logs = {}

        metrics = M.compute_metrics(outputs[0], outputs[1], 'test')

        for k, v in metrics.items():
            logs['test_{}'.format(k)] = v
            self.log('test_{}'.format(k), logs['test_{}'.format(k)], 
                    batch_size = self.params.batch_size)

        scores = utils.dict2string(logs, '{} version {}'.format(self.params.experiment_id, self.logger.version))
        mode = 'a' if os.path.exists(self.experiment_log_filename) else 'w'

        with open(self.experiment_log_filename, mode) as f:
            f.write(scores)

        print(scores)
