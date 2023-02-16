# Learning Universal Authorship Representations

This is the official repository for the EMNLP 2021 paper "Learning Universal Authorship Representations". The paper studies whether the authorship representations learned in one domain transfer to another. To do so, we conduct the first large-scale study of cross-domain transfer for authorship verification considering zero-shot transfers involving three disparate domains: Amazon Reviews, fanfiction short stories, and Reddit comments.

## Installation
Run the following commands to create an environment and install all the required packages:
```bash
python3 -m venv vluar
. ./vluar/bin/activate
pip3 install -U pip
pip3 install -r requirements.txt
```

## Downloading the Data and Pre-trained Weights

Once you've installed the environment, execute the following commands to download the SBERT pre-trained weights, download and preprocess the data:

### Pre-trained Weights

Follow the instructions [here](https://git-lfs.github.com) to install git lfs.

```bash
./scripts/download_sbert_weights.sh
```

### Reddit

Execute the following command to download the Reddit data:

```bash
./scripts/download_reddit_data.sh
```

### Amazon

The amazon data must be requested from [here](https://nijianmo.github.io/amazon/index.html#files) (the "raw review data" (34gb) dataset). Once the data has been downloaded, place the files under "./data/raw_amazon" and run the following command to pre-process the data:

```bash
./scripts/preprocess_amazon_data.sh
```

### Fanfiction

The fanfiction data must be requested from [here](https://zenodo.org/record/3724096#.YT942y1h1pQ). Once the data has been downloaded, place the data.jsonl and truth.jsonl files from the large dataset under "./data/pan_paragraph". Then, run the following command to pre-process the data:

```bash
./scripts/preprocess_fanfiction_data.sh
```

## Path Configuration
The application paths can be changed by modifying the variables in `file_config.ini`:
- **output_path**: Where the experiment results and model checkpoints will be saved. (Default ./output)
- **data_path**: Where the datasets should be stored. (Default ./data)
- **transformer_path**: Where the pretrained model weights for SBERT should be stored. (Default ./pretrained_weights)

We strongly encourage you to set your own paths.

## Reproducing Results

The commands for reproducing each table of results within the paper are found under "./scripts/reproduce/table_N.sh". 

## Training

The commands to train the SBERT model are shown below. There are two types of training: single domain and multi-domain. In short, single-domain models are trained on one dataset while multi-domain models are trained on two datasets. 

The dataset names available for training are:
* iur_dataset - The Reddit dataset from [here](https://aclanthology.org/D19-1178/).
* raw_all - The Reddit Million User Dataset (MUD).
* raw_amazon - The Amazon Reviews dataset.
* pan_paragraph - The PAN Short Stories dataset.


## Training Single-Domain Models

#### Reddit Comments
```bash
python main.py --dataset_name raw_all --do_learn --validate --gpus 4 --experiment_id reddit_model
```
#### Amazon Reviews
```bash
python main.py --dataset_name raw_amazon --do_learn --validate --experiment_id amazon_model
```
#### Fanfiction Stories
```bash
python main.py --dataset_name pan_paragraph --do_learn --validate --experiment_id fanfic_model
```

## Training Multi-Domain Models

### Reddit Comments + Amazon Reviews
```bash
python main.py --dataset_name raw_all+raw_amazon --do_learn --validate --gpus 4 --experiment_id reddit_amazon_model
```
### Amazon Reviews + Fanfiction Stories
```bash
python main.py --dataset_name raw_amazon+pan_paragraph --do_learn --validate --gpus 4 --experiment_id amazon_stories_model
```
#### Reddit Comments + Fanfiction Stories
```bash
python main.py --dataset_name raw_all+pan_paragraph --do_learn --validate --gpus 4 --experiment_id reddit_stories_model
```

## Evaluating
The commands to evaluate on each dataset is shown below. Replace <experiment_id> with the experiment identifier that was used during training. For example, if you followed the commands for single domain training shown above, valid experiment identifiers would be: reddit_model, amazon_model and fanfic_model. 

### Reddit Comments
```bash
python main.py --dataset_name raw_all --evaluate --experiment_id <experiment_id> --load_checkpoint
```

### Amazon Reviews
```bash
python main.py --dataset_name raw_amazon --evaluate --experiment_id <experiment_id> --load_checkpoint
```

### Fanfiction Stories
```bash
python main.py --dataset_name pan_paragraph --evaluate --experiment_id <experiment_id> --load_checkpoint
```

## Contributing

To contribute to LUAR, just send us a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).
When sending a request, make `main` the destination branch on the LUAR repository.

## Citation

If you use our code base in your work, please consider citing:

```
@inproceedings{uar-emnlp2021,
  author    = {Rafael A. Rivera Soto and Olivia Miano and Juanita Ordonez and Barry Chen and Aleem Khan and Marcus Bishop and Nicholas Andrews},
  title     = {Learning Universal Authorship Representations},
  booktitle = {EMNLP},
  year      = {2021},
}
```

## Contact

For questions about our paper or code, please contact [Rafael A. Rivera Soto](riverasoto1@llnl.gov).

## License

LUAR is distributed under the terms of the Apache License (Version 2.0).

All new contributions must be made under the Apache-2.0 licenses.

See LICENSE and NOTICE for details.

SPDX-License-Identifier: Apache-2.0

LLNL-CODE-844702
