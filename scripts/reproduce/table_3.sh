# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#!/bin/sh
# Contains all the commands for reproducing Table 3.

export CUDA_VISIBLE_DEVICES=0,1,2,3;

# Train Single-Domain models with BPE masking on Amazon
python main.py --dataset_name raw_amazon --do_learn --validate --gpus 4 --experiment_id amazon_model_0.05 --mask_bpe_percentage 0.05
python main.py --dataset_name raw_amazon --do_learn --validate --gpus 4 --experiment_id amazon_model_0.15 --mask_bpe_percentage 0.15
python main.py --dataset_name raw_amazon --do_learn --validate --gpus 4 --experiment_id amazon_model_0.30 --mask_bpe_percentage 0.30
python main.py --dataset_name raw_amazon --do_learn --validate --gpus 4 --experiment_id amazon_model_0.45 --mask_bpe_percentage 0.45
python main.py --dataset_name raw_amazon --do_learn --validate --gpus 4 --experiment_id amazon_model_0.60 --mask_bpe_percentage 0.60

# Evaluate each model on Reddit / Amazon / Fanfic
python main.py --dataset_name raw_all --evaluate --experiment_id amazon_model_0.05 --load_checkpoint
python main.py --dataset_name raw_amazon --evaluate --experiment_id amazon_model_0.05 --load_checkpoint
python main.py --dataset_name pan_paragraph --evaluate --experiment_id amazon_model_0.05 --load_checkpoint

python main.py --dataset_name raw_all --evaluate --experiment_id amazon_model_0.15 --load_checkpoint
python main.py --dataset_name raw_amazon --evaluate --experiment_id amazon_model_0.15 --load_checkpoint
python main.py --dataset_name pan_paragraph --evaluate --experiment_id amazon_model_0.15 --load_checkpoint

python main.py --dataset_name raw_all --evaluate --experiment_id amazon_model_0.30 --load_checkpoint
python main.py --dataset_name raw_amazon --evaluate --experiment_id amazon_model_0.30 --load_checkpoint
python main.py --dataset_name pan_paragraph --evaluate --experiment_id amazon_model_0.30 --load_checkpoint

python main.py --dataset_name raw_all --evaluate --experiment_id amazon_model_0.45 --load_checkpoint
python main.py --dataset_name raw_amazon --evaluate --experiment_id amazon_model_0.45 --load_checkpoint
python main.py --dataset_name pan_paragraph --evaluate --experiment_id amazon_model_0.45 --load_checkpoint

python main.py --dataset_name raw_all --evaluate --experiment_id amazon_model_0.60 --load_checkpoint
python main.py --dataset_name raw_amazon --evaluate --experiment_id amazon_model_0.60 --load_checkpoint
python main.py --dataset_name pan_paragraph --evaluate --experiment_id amazon_model_0.60 --load_checkpoint
