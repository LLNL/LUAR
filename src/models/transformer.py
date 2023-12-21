# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import os
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from transformers import AutoModel

from models.layers import MemoryEfficientAttention, SelfAttention
from models.lightning_trainer import LightningTrainer
from utilities.file_utils import Utils as utils


class Transformer(LightningTrainer):
    """Defines the SBERT model.
    """
    def __init__(self, params):
        super(Transformer, self).__init__(params)
        self.save_hyperparameters()
        self.create_transformer()
        
        self.learning_rate = params.learning_rate
        self.attn_fn = SelfAttention()
        self.linear = nn.Linear(self.hidden_size, self.params.embedding_dim)
        
    def create_transformer(self):
        """Creates the Transformer model.
        """
        transformer_modelnames = {
            "roberta": "paraphrase-distilroberta-base-v1",
            "roberta_base": "roberta-base",
        }
        modelname = transformer_modelnames[self.params.model_type]

        model_path = os.path.join(utils.transformer_path, modelname)
        self.transformer = AutoModel.from_pretrained(model_path)

        self.hidden_size = self.transformer.config.hidden_size
        self.num_attention_heads = self.transformer.config.num_attention_heads
        self.dim_head = self.hidden_size // self.num_attention_heads
        
        if self.params.attention_fn_name != "default":
            self.replace_attention()
        
        if self.params.gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()
    
    def replace_attention(self):
        """Replaces the Transformer's Attention mechanism.
        
           NOTE: This feature has only been tested with the regular 
                 original LUAR SBERT pretrained-model.
        """ 
        attn_fn = {
            "memory_efficient": partial(
                MemoryEfficientAttention, 
                q_bucket_size=16, k_bucket_size=32, 
                heads=self.num_attention_heads, dim_head=self.dim_head,
            ),
        }
        
        for i, layer in enumerate(self.transformer.encoder.layer):
            attention = attn_fn[self.params.attention_fn_name]()
            
            state_dict = layer.attention.self.state_dict()
            attention.load_state_dict(state_dict, strict=True)

            self.transformer.encoder.layer[i].attention.self = attention
        
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean Pooling as described in the SBERT paper.
        """
        input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=self.hidden_size).float()
        sum_embeddings = reduce(token_embeddings * input_mask_expanded, 'b l d -> b d', 'sum')
        sum_mask = torch.clamp(reduce(input_mask_expanded, 'b l d -> b d', 'sum'), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_episode_embeddings(self, data):
        """Computes the Author Embedding. 
        """
        # batch_size, num_sample_per_author, episode_length
        input_ids, attention_mask = data[0], data[1]
            
        B, N, E, _ = input_ids.shape
        
        input_ids = rearrange(input_ids, 'b n e l -> (b n e) l')
        attention_mask = rearrange(attention_mask, 'b n e l -> (b n e) l')
        
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        # at this point, we're embedding individual "comments"
        comment_embeddings = self.mean_pooling(outputs['last_hidden_state'], attention_mask)
        comment_embeddings = rearrange(comment_embeddings, '(b n e) l -> (b n) e l', b=B, n=N, e=E)

        # aggregate individual comments embeddings into episode embeddings
        episode_embeddings = self.attn_fn(comment_embeddings, comment_embeddings, comment_embeddings)
        episode_embeddings = reduce(episode_embeddings, 'b e l -> b l', 'max')
        
        episode_embeddings = self.linear(episode_embeddings)
        
        return episode_embeddings, comment_embeddings
    
    def forward(self, data):
        """Calculates a fixed-length feature vector for a batch of episode samples.
        """
        output = self.get_episode_embeddings(data)

        return output

    def _model_forward(self, batch):
        """Passes a batch of data through the model. 
           This is used in the lightning_trainer.py file.
        """
        data, labels = batch

        episode_embeddings, comment_embeddings = self.forward(data)
        labels = torch.flatten(labels)
                
        return episode_embeddings, comment_embeddings
