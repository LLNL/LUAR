# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import os
import tarfile

import gdown

os.makedirs("./data", exist_ok=True)
os.makedirs("./data/fewshot_data", exist_ok=True)
os.makedirs("./data/models/", exist_ok=True)
os.makedirs("./data/models/PROTONET", exist_ok=True)

with open("./file_config.ini", "w+") as fin:
    cwd = os.getcwd()
    fin.write("[Experiment_Paths]"); fin.write('\n')
    fin.write(f"project_root_path = {cwd}"); fin.write('\n')
    fin.write(f"aac_data_path = {os.path.join(cwd, 'data/aac')}"); fin.write('\n')
    fin.write(f"lwd_data_path = {os.path.join(cwd, 'data/lwd')}"); fin.write('\n')
    fin.write(f"fewshot_data_path = {os.path.join(cwd, 'data/fewshot_data')}"); fin.write('\n')
    fin.write(f"weights_path = {os.path.join(cwd, 'data/models')}")

gdown.download("https://drive.google.com/uc?id=14loxE_y5hzJB_EKYsnnA54xds-th7hSG", "./data/aac.tar.gz")
gdown.download("https://drive.google.com/uc?id=1s7ndF5zYGvBSjgs7RsiDoQhaAXbmCi8Q", "./data/lwd.tar.gz")
gdown.download("https://drive.google.com/uc?id=10RrYRqyBAKhLBlNEoDTlWmK_fJt-C1Ox", "./data/fewshot_data/fewshot_data.tar.gz")
gdown.download("https://drive.google.com/uc?id=1Tfbzdp5VVqqbva2oyw8O3kMyHyQnpzJp", "./data/models/MAML.tar.gz")
gdown.download("https://drive.google.com/uc?id=1y0v9D7akL6e4ZlKnpvvwHCLQ8l7wQfbD", "./data/models/Multi-domain.tar.gz")
gdown.download("https://drive.google.com/uc?id=1EPwKG2sgOXXsU0d5GJte_TPjB-yB0ff1", "./data/models/Multi-LLM.tar.gz")
gdown.download("https://drive.google.com/uc?id=1FSmWmJ4pMqpWD0fQN2gNYpop_OC5_QhI", "./data/models/roberta_finetuned.tar.gz")
gdown.download("https://drive.google.com/uc?id=1GVO1Fj8lWukm8x3QK1uetchHh2JIDYT5", "./data/models/PROTONET/from_scratch.tar.gz")

with tarfile.open("./data/aac.tar.gz") as tfin:
    tfin.extractall("./data")
with tarfile.open("./data/lwd.tar.gz") as tfin:
    tfin.extractall("./data")
with tarfile.open("./data/fewshot_data/fewshot_data.tar.gz") as tfin:
    tfin.extractall("./data/fewshot_data")
with tarfile.open("./data/models/MAML.tar.gz") as tfin:
    tfin.extractall("./data/models")
with tarfile.open("./data/models/Multi-domain.tar.gz") as tfin:
    tfin.extractall("./data/models")
with tarfile.open("./data/models/Multi-LLM.tar.gz") as tfin:
    tfin.extractall("./data/models")
with tarfile.open("./data/models/roberta_finetuned.tar.gz") as tfin:
    tfin.extractall("./data/models")
with tarfile.open("./data/models/PROTONET/from_scratch.tar.gz") as tfin:
    tfin.extractall("./data/models/PROTONET")

os.remove("./data/aac.tar.gz")
os.remove("./data/lwd.tar.gz")
os.remove("./data/fewshot_data/fewshot_data.tar.gz")
os.remove("./data/models/MAML.tar.gz")
os.remove("./data/models/Multi-domain.tar.gz")
os.remove("./data/models/Multi-LLM.tar.gz")
os.remove("./data/models/roberta_finetuned.tar.gz")
os.remove("./data/models/PROTONET/from_scratch.tar.gz")