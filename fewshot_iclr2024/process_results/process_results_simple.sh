#!/bin/sh

# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

python dump_auc.py --mode "single_target_simple"
python estimate_statistics.py -i single_target_simple.json
