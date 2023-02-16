# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import pickle5 as pickle
import json
import numpy as np
import sentencepiece as spm
from absl import logging 
from tqdm import tqdm
import gzip
from datetime import datetime
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("data_path", type=str, 
                    help="Path to the Amazon All_Amazon_Review.json.gz file.")

parser.add_argument("stats_path", type=str,
                    help="Path to the stats.npy file.")

args = parser.parse_args()

gz_path = args.data_path

logging.set_verbosity(logging.INFO)
logging.info('loading pickle')
try:
  amazon_count = pickle.load(open(args.stats_path, 'rb'))
except:
  amazon_count = defaultdict(int)
  with gzip.GzipFile(gz_path, 'r') as handle:
    for line in tqdm(handle, total=233055327):
      j = json.loads(line)
      amazon_count[j['reviewerID']] += 1
  np.save(args.stats_path, np.array(amazon_count), allow_pickle=True)

out_map = {}

def build_empty_json_obj():
  out_map = {'author_id' : -1,
             'syms' : [],
             'hour' : [],
             'minute' : [],
             'day' : []
             }
  return out_map
aid = -1
logging.info('iterating gzip')
with gzip.GzipFile(gz_path, 'r') as handle:
  for line in tqdm(handle, total=233055327):
    j = json.loads(line)
    if 'reviewerID' not in j or 'unixReviewTime' not in j or 'reviewText' not in j:
      continue
    assert j['reviewerID'] in amazon_count
    if amazon_count[j['reviewerID']] < 100:
      continue
    
    author = j['reviewerID']
    timestamp = j['unixReviewTime']
    text = j['reviewText']
    d = datetime.fromtimestamp(timestamp)
    if author in out_map:
      out_map[author]['syms'].append(text)
      out_map[author]['hour'].append(d.hour)
      out_map[author]['minute'].append(d.minute)
      out_map[author]['day'].append(d.day)
    else:
      aid += 1
      out = build_empty_json_obj()
      out['author_id'] = aid
      out['syms'].append(text)
      out['hour'].append(d.hour)
      out['minute'].append(d.minute)
      out['day'].append(d.day)
      out_map[author] = out

logging.info('Writing out to json')
with open('./data/raw_amazon/out_raw_100.jsonl', 'w+') as handle:
  for k, v in out_map.items():
    handle.write(json.dumps(v) + '\n')

