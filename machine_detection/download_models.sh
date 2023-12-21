# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#!bin/bash

# downloading the pre-trained KNN model
mkdir -p knn_model
wget -O knn.pkl https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/rriver28_jh_edu/Ef_WEm5bG8ZAoqmeoy8oZckBFqjzJ98WRRbBrAnwbIwgJQ?download=1
mv knn.pkl knn_model/

# downloading the LUAR model and tokenizer (optional if online)
# git lfs install
# git clone https://huggingface.co/rrivera1849/LUAR-MUD
# git clone https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1
# mkdir -p sentence-transformers
# mkdir -p rrivera1849
# mv LUAR-MUD rrivera1849/
# mv paraphrase-distilroberta-base-v1 sentence-transformers/