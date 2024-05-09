# Copyright 2024 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd

human_data = pd.read_json("iclr2024_20240430.json", lines=True)
human_data["review_count"] = human_data.review_content.apply(len)
# repeat URL review count times
human_data["url"] = human_data[["url", "review_count"]].apply(lambda x: [x[0]] * x[1], axis=1)
urls = [j for i in human_data.url.tolist() for j in i]

human_data = human_data.review_content.tolist()
human_data = [j for i in human_data for j in i]

machine_data = []
with open("openai_responses.jsonl", "r") as fin:
    for line in fin:
        line_data = json.loads(line)

        finish_reason = line_data[1]["choices"][0]["finish_reason"]
        if finish_reason != "stop":
            continue
        machine_data.append(line_data[1]["choices"][0]["message"]["content"])

data = {
    "human": human_data,
    "machine": machine_data,
    "openreview_urls": urls,
}

with open("iclr2024_human_machine.json", "w") as fout:
    json.dump(data, fout)