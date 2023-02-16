# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#!/bin/sh
# Contains all the commands for reproducing Table 1.

export CUDA_VISIBLE_DEVICES=0,1,2,3;

# Train Single-Domain models
python main.py --dataset_name raw_all --do_learn --validate --gpus 4 --experiment_id reddit_model
python main.py --dataset_name raw_amazon --do_learn --validate --gpus 4 --experiment_id amazon_model
python main.py --dataset_name pan_paragraph --do_learn --validate --gpus 4 --experiment_id fanfic_model

# Evaluate Reddit -> (Reddit / Amazon / Fanfic)
python main.py --dataset_name raw_all --evaluate --experiment_id reddit_model --load_checkpoint
python main.py --dataset_name raw_amazon --evaluate --experiment_id reddit_model --load_checkpoint
python main.py --dataset_name pan_paragraph --evaluate --experiment_id reddit_model --load_checkpoint

# Evaluate Amazon -> (Reddit / Amazon / Fanfic)
python main.py --dataset_name raw_all --evaluate --experiment_id amazon_model --load_checkpoint
python main.py --dataset_name raw_amazon --evaluate --experiment_id amazon_model --load_checkpoint
python main.py --dataset_name pan_paragraph --evaluate --experiment_id amazon_model --load_checkpoint

# Evaluate Fanfic -> (Reddit / Amazon / Fanfic)
python main.py --dataset_name raw_all --evaluate --experiment_id fanfic_model --load_checkpoint
python main.py --dataset_name raw_amazon --evaluate --experiment_id fanfic_model --load_checkpoint
python main.py --dataset_name pan_paragraph --evaluate --experiment_id fanfic_model --load_checkpoint