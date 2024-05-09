# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

parser = ArgumentParser()
parser.add_argument("-i", "--input_file_name", default="single_target.json")
args = parser.parse_args()

METHOD_ORDER = [
    # FewShot
    "LUAR_CRUD",
    "LUAR_MACHINE",
    "LUAR_MULTIDOMAIN",
    "WEGMANN",
    "ROBERTA_MAML",
    "ROBERTA_PROTONET",
    "SBERT",
    # ZeroShot
    "ROBERTA",
    "OPENAI",
    "RANK",
    "LOGRANK",
    "ENTROPY",
    # Additions for Paraphrase Methods
    "LUAR_IUR_PARAPHRASE",
    "LUAR_IUR_PARAPHRASE_MULTIPLE_TARGET",
    "LUAR_CRUD_PARAPHRASE",
    "LUAR_CRUD_PARAPHRASE_MULTIPLE_TARGET",
    "PROTONET_PARAPHRASE",
]

METHOD_NAME_TO_LABEL = {
    "LUAR_CRUD": "LUAR_CRUD_Reddit",
    "LUAR_MULTIDOMAIN": "LUAR_Reddit,Twitter,StackExchange",
    "LUAR_MACHINE": "LUAR_AAC",
    "WEGMANN": "CISR",
    "SBERT": "SBERT",
    "ROBERTA": "RoBERTa (ZeroShot)",
    "ROBERTA_MAML": "RoBERTa_MAML",
    "ROBERTA_PROTONET": "RoBERTa_PROTONET",
    "OPENAI": "RoBERTa_OpenAI",
    # Additions for Paraphrase Methods
    "LUAR_IUR_PARAPHRASE": "LUAR_IUR_Paraphrase_Reddit",
    "LUAR_IUR_PARAPHRASE_MULTIPLE_TARGET": "LUAR_IUR_Paraphrase_Reddit_MultipleTarget",
    "LUAR_CRUD_PARAPHRASE": "LUAR_CRUD_Paraphrase_Reddit",
    "LUAR_CRUD_PARAPHRASE_MULTIPLE_TARGET": "LUAR_CRUD_Paraphrase_Reddit_MultipleTarget",
    "PROTONET_PARAPHRASE": "PROTONET_Paraphrase",
}

def get_method_data(AUC):
    """Returns a dictionary of the form:
    {
        num_tokens: {
            method_name: [auc1, auc2, ...]
        }
    }
    """
    method_data = {}

    for dirname in AUC.keys():
        for method in AUC[dirname].keys():
            for num_tokens in AUC[dirname][method].keys():
                LM_data = AUC[dirname][method][num_tokens]

                data = [value for value in LM_data.values()]
                data = [j for i in data for j in i]

                if num_tokens not in method_data.keys():
                    method_data[num_tokens] = {}
                method_data[num_tokens][method] = method_data[num_tokens].get(method, []) + data

    return method_data

def estimate_mean(data):
    """Estimates the mean of the data using bootstrap resampling.
    """
    res = bootstrap(data, np.mean, n_resamples=1000, method="basic")

    return (
        np.mean(data),
        res.confidence_interval.low,
        res.confidence_interval.high,
        res.standard_error,
    )
    
def print_table_1_results(df):
    df = df.groupby(["num_tokens", "method_name"]).agg(list)
    df.drop(columns=["low", "high"], inplace=True)

    new_order = deepcopy(METHOD_ORDER)
    new_order = [method_name for method_name in new_order if method_name in df.index.get_level_values("method_name")]

    header = "\\bf Number of Documents "
    for method_name in new_order:
        if method_name == "WEGMANN":
            method_name = "CISR"
        if "_" in method_name:
            split = method_name.split("_")
            method_name = f"{split[0]}\\_{split[1]}"
        header += f"& \\bf {method_name} "
    header += "\\\\\\toprule"
    print(header)

    # 5 & 10 posts
    for method_name in new_order:
        line = f"{method_name} & "

        num_tokens_or_examples = [str(128 * 5), str(128 * 10)]

        for x in num_tokens_or_examples:
            mean = df.loc[x, method_name]["mean"][0]
            stderr = df.loc[x, method_name]["stderr"][0]
            line += f"{mean:.4f} ({stderr:.4f}) &"
        line = line[:-1] + "\\\\"
        print(line)

def print_breakdown_by_dataset(AUC):
    dataset_map = {
        "amazon": "Amazon",
        "M4_arxiv": "M4 Arxiv",
        "M4_peerread": "M4 PeerRead",
        "M4_wikipedia": "M4 Wikipedia",
        "M4_wikihow": "M4 WikiHow",
    }

    print("dataset order: ", AUC.keys())
    for dataset_name in AUC.keys():
        latex_str_1 = dataset_map[dataset_name]
        latex_str_1 = latex_str_1 + " & 5"

        latex_str_2 = "& 10"
        for method_name in METHOD_ORDER:
            if method_name not in AUC[dataset_name].keys(): continue

            data = AUC[dataset_name][method_name]["640"]
            data = [value for value in data.values()]
            data = [j for i in data for j in i]
            data = (np.array(data), )
            mean, _, _, _ = estimate_mean(data)
            latex_str_1 += f" & {mean:.3f}"

            data = AUC[dataset_name][method_name]["1280"]
            data = [value for value in data.values()]
            data = [j for i in data for j in i]

            data = (np.array(data), )
            mean, _, _, _ = estimate_mean(data)
            latex_str_2 += f" & {mean:.3f}"

        latex_str_1 += " & 0.5 \\\\"
        latex_str_2 += " & 0.5 \\\\\\midrule"
        print(latex_str_1)
        print(latex_str_2)


def process_results_by_dataset(AUC):
    method_data = get_method_data(AUC)
    arguments = []
    for num_tokens in method_data.keys():
        for method_name in METHOD_ORDER:
            if method_name not in method_data[num_tokens].keys():
                continue
            data = method_data[num_tokens][method_name]
            data = (np.array(data),)
            arguments.append(data)
    print(f"len(args) = {len(arguments)}")
    with Pool(40) as p:
        results = p.map(estimate_mean, arguments)
    data = {
        "num_tokens": [],
        "method_name": [],
        "mean": [],
        "low": [],
        "high": [],
        "stderr": [],
    }
    i = 0
    for num_tokens in method_data.keys():
        for method_name in METHOD_ORDER:
            if method_name not in method_data[num_tokens].keys():
                continue
            mean, low, high, stderr = results[i]
            data["num_tokens"].append(num_tokens)
            data["method_name"].append(method_name)
            data["mean"].append(mean)
            data["low"].append(low)
            data["high"].append(high)
            data["stderr"].append(stderr)
            i += 1
    df = pd.DataFrame(data)

    if "paraphrase" not in args.input_file_name:
        print_table_1_results(df)
    
    if "simple" not in args.input_file_name:
        df["method_name"] = df["method_name"].apply(lambda x: METHOD_NAME_TO_LABEL[x])
        results_d = {}
        for _, row in df.iterrows():
            if row["method_name"] not in results_d.keys():
                results_d[row["method_name"]] = []
            results_d[row["method_name"]].append(row["mean"])
        output_file_name = os.path.splitext(args.input_file_name)[0] + "_results.json"
        with open(output_file_name, "w+") as fout:
            fout.write(json.dumps(results_d))

def main():
    AUC = json.load(open(args.input_file_name, "r"))
    process_results_by_dataset(AUC)
    if "paraphrase" not in args.input_file_name and "simple" not in args.input_file_name:
        print_breakdown_by_dataset(AUC)
    return 0

if __name__ == "__main__":
    sys.exit(main())
