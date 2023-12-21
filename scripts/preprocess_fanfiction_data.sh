# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#! /usr/bin/env bash

if [ ! -d "./data" ]; then
    mkdir ./data
fi

if [ ! -d "./data/pan_paragraph" ]; then
    mkdir ./data/pan_paragraph
fi

data_path="./data/pan_paragraph/data.jsonl"
truth_path="./data/pan_paragraph/truth.jsonl"

python scripts/preprocess_pan_into_paragraphs.py ${data_path} ${truth_path}
