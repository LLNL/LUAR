# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("-i", "--input_data_file", type=str, default="single_target_results.json")
args = parser.parse_args()

def main():
    data = json.loads(open(args.input_data_file, "r").read())

    _ = plt.figure()
    for model_name, values in data.items():
        N = len(values)
        if "paraphrase" in args.input_data_file:
            x = [0.0, 0.25, 0.5, 0.75, 1.0]
        else:
            x = range(1, N+1)
        y = values
        plt.plot(x, y, label=model_name)
    
    plt.legend()
    plt.ylabel("pAUC")
    if "paraphrase" in args.input_data_file:
        plt.xlabel("Proportion of Queries Paraphrased")
    else:
        plt.xlabel("Number of Documents")

    save_name = os.path.splitext(args.input_data_file)[0] + ".png"
    plt.savefig(save_name)
    plt.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())