# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#! /usr/bin/env bash

if [ ! -d "./data" ]; then
    mkdir ./data
fi

if [ ! -d "./data/raw_amazon" ]; then
    mkdir ./data/raw_amazon
fi

data_path="./data/raw_amazon/All_Amazon_Review.json.gz"
stats_path="./data/raw_amazon/stats.npy"

python scripts/process_amazon_rawtxt.py ${data_path} ${stats_path}

head -n 100000 ./data/raw_amazon/out_raw_100.jsonl > ./data/raw_amazon/train.jsonl
tail -n 35059 ./data/raw_amazon/out_raw_100.jsonl > ./data/raw_amazon/validation.jsonl

python scripts/split_amazon.py
