# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from argparse import ArgumentParser

from datasets import load_from_disk
import evaluate
import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          set_seed)
sys.path.append(os.path.abspath("../"))
from file_utils import Utils as utils

parser = ArgumentParser()
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--model_name", type=str, default="roberta-base", help="base model to finetune.")
parser.add_argument("--max_token_length", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--early_stopping_patience", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--debug", action="store_true", help="if true, n_train, n_val = 100")
parser.add_argument("--hyperparameter_search", action="store_true")
parser.add_argument("--n_trials", type=int, default=10, help="number of trials for hyperparameter search.")

"""Hyper parameter search """
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate",[0.001, 0.01, 0.1]),
        "weight_decay" : trial.suggest_categorical("weight_decay", [0.001, 0.001, 0.01])
    }

def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(os.path.join(utils.weights_path, args.model_name))

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def tokenize_data(dataset_split, tokenizer):
    if args.debug:
        dataset_split = dataset_split.shuffle(seed=args.seed).select(range(100))
    else:
        dataset_split
    tokenized_split = dataset_split.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_split = tokenized_split.remove_columns(["text", "model", "decoding", "temp", "token"])
    return tokenized_split

""" Training """
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def set_training_args(output_directory):
    training_args = TrainingArguments(
        output_dir=output_directory, 
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size, 
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="accuracy",
        num_train_epochs=args.max_epochs,
        seed=args.seed,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optim="adamw_torch"
    )
    return training_args

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"]="true"
    set_seed(args.seed)
    directory_name = os.path.join(utils.output_path, "model_checkpoints", "roberta_finetuned")
    os.makedirs(directory_name, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(utils.weights_path, args.model_name), num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(utils.weights_path, args.model_name), model_max_length=args.max_token_length)
    dataset = load_from_disk(utils.aac_data_path)
    train = tokenize_data(dataset["train"], tokenizer)
    valid = tokenize_data(dataset["valid"], tokenizer)

    trainer = Trainer(
        model=None if args.hyperparameter_search else model,
        args=set_training_args(directory_name),
        train_dataset=train,
        eval_dataset=valid,
        compute_metrics=compute_metrics,
        model_init=model_init if args.hyperparameter_search else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)])

    if args.hyperparameter_search:
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            study_name="roberta",
            hp_space=optuna_hp_space,
            n_trials=args.n_trials,
        )
    else:
        trainer.train()
        trainer.save_model(directory_name)

if __name__ == "__main__":
    args = parser.parse_args()
    sys.exit(main(args))
