#!/bin/sh

# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

export CUDA_VISIBLE_DEVICES=0;

MODELS=(
    "CRUD"
)

DATASETS=(
    "M4_peerread"
    "M4_arxiv"
    "M4_wikihow"
    "M4_wikipedia"
    "amazon"
)

for size in 5 10; do
    for dataset in ${DATASETS[@]}; do
        for model in ${MODELS[@]}; do
            let max_trials=99999999

            python evaluate.py \
                --mode "single_target" \
                --model_name ${model} \
                --force_recompute \
                --query_size ${size} \
                --support_size ${size} \
                --num_tokens 128 \
                --max_trials ${max_trials} \
                --dirname ${dataset}
        done
    done
done
