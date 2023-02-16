# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import json
from tqdm import tqdm

dev_file = './data/raw_amazon/validation.jsonl'

dev_query = './data/raw_amazon/validation_queries.jsonl'
dev_target= './data/raw_amazon/validation_targets.jsonl'

def build_json_obj():
  out_map = {'author_id' : -1,
             'syms' : [],
             'hour' : [],
             'minute' : [],
             'day' : [],
             'action_type' : []}
  return out_map

with open(dev_query, 'w') as dev_qhandle:
  with open(dev_target, 'w') as dev_thandle:
    with open(dev_file, 'r') as dev_handle:
      for i in dev_handle:
        j = json.loads(i)
        curr_len = len(j['syms'])
        split_idx = int(curr_len/2)

        q_map = build_json_obj()
        t_map = build_json_obj()

        q_map['author_id'] = j['author_id']
        t_map['author_id'] = j['author_id']

        q_map['syms'] = j['syms'][:split_idx]
        t_map['syms'] = j['syms'][split_idx:]

        q_map['hour'] = j['hour'][:split_idx]
        t_map['hour'] = j['hour'][split_idx:]

        q_map['minute'] = j['minute'][:split_idx]
        t_map['minute'] = j['minute'][split_idx:]

        q_map['day'] = j['day'][:split_idx]
        t_map['day'] = j['day'][split_idx:]
        
        dev_qhandle.write(json.dumps(q_map) + '\n')
        dev_thandle.write(json.dumps(t_map) + '\n')

