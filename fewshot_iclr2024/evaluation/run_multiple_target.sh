#!/bin/sh

# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

export CUDA_VISIBLE_DEVICES=0;

MODELS=(
    "CRUD"
    "Multi-LLM"
    "Multi-domain"
    "SBERT"
    "WEGMANN"
    "PROTONET"
)

DATASETS=(
    "M4_peerread"
    "M4_arxiv"
    "M4_wikihow"
    "M4_wikipedia"
    "amazon"
)

for size in 1 2 3 4 5 6 7 8 9 10; do
    for dataset in ${DATASETS[@]}; do
        for model in ${MODELS[@]}; do
            if [[ ${model} == "MAML" ]]; then
                let max_trials=20
            else
                let max_trials=1000
            fi

            python evaluate.py \
                --mode "multiple_target" \
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
