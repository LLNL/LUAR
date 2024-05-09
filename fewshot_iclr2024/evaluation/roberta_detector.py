# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed
)

from baseline_training.prototypical_helper import mean_pooling
from file_utils import Utils as utils

class Detector(torch.nn.Module):
    """Detector using RoBERTa"""

    def __init__(
        self,
        model_name,
    ):
        super(Detector, self).__init__()
        self.model_name = model_name
        hfid_or_checkpoint = self.get_hfid_or_checkpoint(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(hfid_or_checkpoint)
        self.machine_value, self.human_value = self.model_values()

    def get_hfid_or_checkpoint(self, model_name):
        """Get the Hugging Face ID or checkpoint path for the model.
        """
        if model_name == "roberta_finetuned":
            hfid_or_checkpoint = os.path.join(utils.weights_path, "roberta_finetuned")
        elif model_name == "openAI":
            hfid_or_checkpoint = "openai-community/roberta-base-openai-detector"
        else:
            raise ValueError(f"Model name {model_name} not recognized.")
        return hfid_or_checkpoint

    def to_gpu(self):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device_ids = list(range(torch.cuda.device_count()))
                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            else:
                self.model = self.model.cuda()

    def model_values(self):
        values =  {
            "openAI": {"human" : 1, "machine" : 0},
            "roberta_finetuned": {"human": 0, "machine": 1}
        }
        return values[self.model_name]["machine"], values[self.model_name]["human"]
    
    def get_tokenizer(self):
        tokenizer_name = "roberta-base" if self.model_name == "roberta_finetuned" else "openai-community/roberta-base-openai-detector"
        return AutoTokenizer.from_pretrained(tokenizer_name)    

    def tokenize_function(self,examples, tokenizer, max_length=512):
        tokenized_examples = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length, 
            return_tensors="pt"
        )
        return tokenized_examples

    def create_dataset(self, dictionary, tokenizer=False, max_token_length=512):
        """Create dataset, and optionally tokenizes it.
        """
        dataset = Dataset.from_dict(dictionary)
        if tokenizer:
            dataset = dataset.map(lambda x : self.tokenize_function(x, tokenizer, max_token_length))
            dataset = dataset.remove_columns(["text"])
        dataset.set_format("torch")
        return dataset

    def freeze_backbone(self):
        """Freeze all layers except classification head.
        """
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def evaluate(self, dataset, batch_size=64, return_embeddings=False):
        self.model.eval()
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
        
        if return_embeddings:
            embeddings = []
        else:
            decisions, probs = [], []

        with torch.inference_mode():
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].squeeze(1).to(self.model.device)
                attention_mask = batch["attention_mask"].squeeze(1).to(self.model.device)
                out = self.model(
                    input_ids, attention_mask, 
                    output_hidden_states=True if return_embeddings else False
                )
                
                if return_embeddings:
                    embedding = mean_pooling(
                        self.model.config.hidden_size, 
                        out.hidden_states[-1], 
                        attention_mask
                    )
                    embeddings.append(embedding)
                else:
                    prob = out.logits.softmax(dim=-1)
                    dec = (prob[:, self.machine_value] > 0.5).long()
                    prob = prob[:, self.machine_value]
                    decisions.append(dec)
                    probs.append(prob)

        if return_embeddings:
            embeddings = torch.cat(embeddings, axis=0).cpu().numpy()
            return embeddings
        else:
            decisions = torch.cat(decisions, axis=0).cpu().tolist()
            probs = torch.cat(probs, axis=0).cpu().tolist()
            return decisions, probs

    def few_shot(self, dataset, seed, num_epochs=20, lr=5e-5, shuffle=True, batch_size=4):
        """Performs fine-tuning on the set of Few-Shot examples.
        """
        set_seed(seed)
        self.freeze_backbone()
        optimizer = AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
        num_training_steps = num_epochs * len(dataloader)

        progress_bar = tqdm(range(num_training_steps))
        
        for _ in range(num_epochs):
            for batch  in dataloader:
                batch['labels'] = batch['labels'].to(self.model.device)
                batch["input_ids"] = batch["input_ids"].squeeze(1).to(self.model.device)
                batch["attention_mask"] = batch["attention_mask"].squeeze(1).to(self.model.device)

                outputs = self.model(**batch)
                loss = outputs.loss.mean()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)