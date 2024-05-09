#!/bin/sh

# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

export CUDA_VISIBLE_DEVICES=0;

python meta_train.py --experiment_id "from_scratch" --fewshot_method PROTONET
python models/roberta_baseline/meta_train.py --experiment_id "from_scratch-ablation=length" --fewshot_method PROTONET --dont_filter_lengths
python models/roberta_baseline/meta_train.py --experiment_id "from_scratch-ablation=length,smalltopic" --fewshot_method PROTONET --dont_filter_lengths --use_many_topics
