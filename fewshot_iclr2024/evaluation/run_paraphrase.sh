#!/bin/sh

# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

export CUDA_VISIBLE_DEVICES=0;

MODELS=(
    "CRUD"
    "PROTONET"
)

DATASETS=(
    "M4_peerread"
    "M4_arxiv"
    "M4_wikihow"
    "M4_wikipedia"
    "amazon"
)

for dataset in ${DATASETS[@]}; do
    for model in ${MODELS[@]}; do
        if [[ ${model} == "CRUD" ]]; then
            MODES=("single_target" "multiple_target_paraphrase")
        else
            MODES=("single_target")
        fi

        echo ${dataset} ${model}

        # with varying proportion of queries paraphrased:
        for mode in ${MODES[@]}; do
            for p in 0.25 0.50 0.75 1.0; do
                if [[ ${mode} == "single_target" ]]; then
                    let max_trials=99999999
                else
                    let max_trials=1000
                fi

                echo "\t" ${mode} ${p} ${max_trials}

                python evaluate.py \
                    --mode ${mode} \
                    --model_name ${model} \
                    --force_recompute \
                    --query_size 5 \
                    --support_size 5 \
                    --num_tokens 128 \
                    --max_trials ${max_trials} \
                    --paraphrase_p ${p} \
                    --dirname ${dataset}
            done
        done

        # # # without any of the queries paraphrased:
        python evaluate.py \
            --mode "single_target" \
            --model_name ${model} \
            --force_recompute \
            --query_size 5 \
            --support_size 5 \
            --num_tokens 128 \
            --max_trials 99999999 \
            --dirname ${dataset}
    done
done
