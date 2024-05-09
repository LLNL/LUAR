#!/bin/sh

# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

export CUDA_VISIBLE_DEVICES=0,1,2,3;
python meta_train.py --experiment_id "from_scratch" --num_train_iterations 5000
