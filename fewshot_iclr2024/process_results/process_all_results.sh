#!/bin/sh

# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

python dump_auc.py
python dump_auc.py --mode "multiple_target"
python dump_auc.py --mode "paraphrase"
python estimate_statistics.py -i single_target.json
python estimate_statistics.py -i multiple_target.json
python estimate_statistics.py -i paraphrase.json
python plot.py -i single_target_results.json
python plot.py -i multiple_target_results.json
python plot.py -i paraphrase_results.json