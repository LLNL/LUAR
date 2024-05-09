# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
import sys
from argparse import ArgumentParser
from multiprocessing import Pool

sys.path.append(os.path.abspath("../"))
from file_utils import Utils as utils

parser = ArgumentParser()
parser.add_argument("--mode", type=str, default="single_target", 
                    choices=["single_target", "multiple_target", "paraphrase", "single_target_simple"])
args = parser.parse_args()

DIRNAMES = [
    "amazon",
    "M4_arxiv",
    "M4_peerread",
    "M4_wikihow",
    "M4_wikipedia",
]

DATA = {
    "LUAR_CRUD": [
        f"metrics_CRUD_mode=single_target_{i}_{i}_128.json" for i in range(1, 10+1)
    ],
    "LUAR_MACHINE": [
        f"metrics_Multi-LLM_mode=single_target_{i}_{i}_128.json" for i in range(1, 10+1)
    ],
    "LUAR_MULTIDOMAIN": [
        f"metrics_Multi-domain_mode=single_target_{i}_{i}_128.json" for i in range(1, 10+1)
    ],
    "SBERT": [
        f"metrics_SBERT_mode=single_target_{i}_{i}_128.json" for i in range(1, 10+1)
    ],
    "WEGMANN": [
        f"metrics_WEGMANN_mode=single_target_{i}_{i}_128.json" for i in range(1, 10+1)
    ],
    "ROBERTA_MAML": [
        f"metrics_MAML_mode=single_target_{i}_{i}_128.json" for i in range(1, 10+1)
    ],
    "ROBERTA_PROTONET": [
        f"metrics_PROTONET_mode=single_target_{i}_{i}_128.json" for i in range(1, 10+1)
    ],
    "ROBERTA": [
        f"metrics_roberta_finetuned_mode=single_target_{i}_{i}_128.json" for i in range(1, 10+1)
    ],
    "OPENAI": [
        f"metrics_openAI_mode=single_target_{i}_{i}_128.json" for i in range(1, 10+1)
    ]
}

if args.mode == "single_target_simple":
    DATA = {
        "LUAR_CRUD": [DATA["LUAR_CRUD"][4], DATA["LUAR_CRUD"][9]],
    }
elif args.mode == "paraphrase":
    DATA = {
        "LUAR_CRUD_PARAPHRASE": [
            "metrics_CRUD_mode=single_target_5_5_128.json",
            "metrics_CRUD_mode=single_target_p=0.25_L=20_5_5_128.json",
            "metrics_CRUD_mode=single_target_p=0.5_L=20_5_5_128.json",
            "metrics_CRUD_mode=single_target_p=0.75_L=20_5_5_128.json",
            "metrics_CRUD_mode=single_target_p=1.0_L=20_5_5_128.json",
        ],
        "LUAR_CRUD_PARAPHRASE_MULTIPLE_TARGET": [
            "metrics_CRUD_mode=single_target_5_5_128.json",
            "metrics_CRUD_mode=multiple_target_paraphrase_p=0.25_L=20_5_5_128.json",
            "metrics_CRUD_mode=multiple_target_paraphrase_p=0.5_L=20_5_5_128.json",
            "metrics_CRUD_mode=multiple_target_paraphrase_p=0.75_L=20_5_5_128.json",
            "metrics_CRUD_mode=multiple_target_paraphrase_p=1.0_L=20_5_5_128.json",
        ],
        "PROTONET_PARAPHRASE": [
            "metrics_PROTONET_mode=single_target_5_5_128.json",
            "metrics_PROTONET_mode=single_target_p=0.25_L=20_5_5_128.json",
            "metrics_PROTONET_mode=single_target_p=0.5_L=20_5_5_128.json",
            "metrics_PROTONET_mode=single_target_p=0.75_L=20_5_5_128.json",
            "metrics_PROTONET_mode=single_target_p=1.0_L=20_5_5_128.json",
        ],
    }

def get_num_tokens(filename):
    """Derive the number of tokens from the filename.
    """
    # example: "metrics_CRUD_mode=single_target_5_5_128.json"
    num_tokens = filename.split(".")[0].split("_")[-1]
    num_documents = filename.split(".")[0].split("_")[-2]
    return int(num_tokens) * int(num_documents)

def get_paraphrase_identifier(filename):
    """For the paraphrase data, we're evaluating with N=5, so we're not looking at the 
       number of tokens. Instead, we're looking at how many queries are paraphrased.
    """
    if "p=0.25" in filename:
        return "0.25"
    elif "p=0.5" in filename:
        return "0.5"
    elif "p=0.75" in filename:
        return "0.75"
    elif "p=1.0" in filename:
        return "1.0"
    else:
        return "0.0"
    
def get_auc_cutoff(model_name, path):
    """Get the AUC cutoff for each model and dataset.
    
    Returns a dictionary with the following format:
    {
        num_tokens: {
            LLM: [trial1, trial2, ...],
        }
    }
    """
    metric_filenames = DATA[model_name]
    for metric_filename in metric_filenames:
        if args.mode == "multiple_target":
            if not os.path.exists(os.path.join(path, metric_filename.replace("single_target", args.mode))):
                return {}
    
    start = time.time()
    print("Processing: {} {}".format(os.path.basename(path), model_name))

    result = {}
    for metric_filename in metric_filenames:
        if args.mode == "multiple_target":
            metric_filename = metric_filename.replace("single_target", args.mode)
        
        metrics = json.load(open(os.path.join(path, metric_filename)))
        if args.mode == "paraphrase":
            identifier = get_paraphrase_identifier(metric_filename)
        else:
            identifier = get_num_tokens(metric_filename)

        result[identifier] = {}
        for k in metrics.keys():
            if k == "global": continue
            result[identifier][k] = [trial["roc_auc_cutoff"] for trial in metrics[k].values()]

    print("Finished: {} {} in {} seconds".format(os.path.basename(path), model_name, time.time() - start))
    return result

def main():
    fewshot_data_path = utils.fewshot_data_path
    all_results = {}

    arguments = []
    for dirname in DIRNAMES:
        full_path = os.path.join(fewshot_data_path, dirname)
        for model_name in DATA.keys():
            arguments.append((model_name, full_path))

    with Pool(40) as pool:
        out = pool.starmap(get_auc_cutoff, arguments)

    for (model_name, full_path) in arguments:
        dirname = os.path.basename(full_path)
        all_results[dirname] = all_results.get(dirname, {})
        
        res = out.pop(0)
        if len(res) == 0:
            continue

        all_results[dirname][model_name] = res

    with open(f"{args.mode}.json", "w+") as f:
        f.write(json.dumps(all_results))

    return 0

if __name__ == "__main__":
    sys.exit(main())
