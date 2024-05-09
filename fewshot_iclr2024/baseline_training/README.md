# Baseline Model Training

This directory contains code for training the finetuned RoBERTA model, few-shot Protonet, and MAML RoBERTa models. 
All models use `roberta-base` as the backbone.

## Datasets and Pre-trained Checkpoints

To download the AAC datasets, and pre-trained models, please see the README file at the root of this repository.

## Reproducing Baseline Training

Once the paths and data have been configured as discussed in the root README, the following commands may be used to reproduce our baseline training. 

To train the AI Detector (fine-tuned):
```bash
python finetune_roberta.py
```

To train the base PROTONET model, and the ablations discussed in the Appendix, run the following command:
```bash
./protonet.sh
```

To train the MAML model, run the following command:
```bash
python meta_train.py --fewshot_method MAML
```