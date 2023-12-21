# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#!/bin/sh
# Contains all the commands for reproducing Table 2.

export CUDA_VISIBLE_DEVICES=0,1,2,3;

# Train Multi-Domain models
python main.py --dataset_name raw_all+raw_amazon --do_learn --validate --gpus 4 --experiment_id reddit_amazon_model
python main.py --dataset_name raw_amazon+pan_paragraph --do_learn --validate --gpus 4 --experiment_id amazon_stories_model
python main.py --dataset_name raw_all+pan_paragraph --do_learn --validate --gpus 4 --experiment_id reddit_stories_model

# Evaluate Reddit+Amazon -> (Reddit / Amazon / Fanfic)
python main.py --dataset_name raw_all --evaluate --experiment_id reddit_amazon_model --load_checkpoint
python main.py --dataset_name raw_amazon --evaluate --experiment_id reddit_amazon_model --load_checkpoint
python main.py --dataset_name pan_paragraph --evaluate --experiment_id reddit_amazon_model --load_checkpoint

# Evaluate Amazon+Fanfic -> (Reddit / Amazon / Fanfic)
python main.py --dataset_name raw_all --evaluate --experiment_id amazon_stories_model --load_checkpoint
python main.py --dataset_name raw_amazon --evaluate --experiment_id amazon_stories_model --load_checkpoint
python main.py --dataset_name pan_paragraph --evaluate --experiment_id amazon_stories_model --load_checkpoint

# Evaluate Reddit+Fanfic -> (Reddit / Amazon / Fanfic)
python main.py --dataset_name raw_all --evaluate --experiment_id reddit_stories_model --load_checkpoint
python main.py --dataset_name raw_amazon --evaluate --experiment_id reddit_stories_model --load_checkpoint
python main.py --dataset_name pan_paragraph --evaluate --experiment_id reddit_stories_model --load_checkpoint