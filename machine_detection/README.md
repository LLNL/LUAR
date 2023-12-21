# Machine Generated Text Detector

## Model Description
The model in this deployment package was developed to classify input text as either human-written or machine-generated. Given input text, the models output a score ranging from 0.0 to 1.0 representing the extent to which the input text is assessed by the model to be machine-generated (i.e., 1.0 being most likely to be machine-generated).  
 
The model takes the input text and extracts authorship embeddings provided by a pre-trained Learning Universal Authorship Representations (LUAR) model [Rivera Soto, et al. “Learning Universal Authorship Representations”, in Proceedings of EMNLP 2021.]. These authorship embeddings serve as input to a K-Nearest Neighbors classifier with k=100 trained using the fine-tuning dataset described below.
 
Please note that these classifiers are not infallible and should not be used as the sole method of determining the authorship of text. In early experiments on a test set of 9,619 machine generations and 11,974 human-written texts, the classifier was able to correctly classify 99.2% of machine-generated texts (TPR) while incorrectly classifying 7.4% of human-written texts as machine-generated (FPR).
 
## Datasets
The models use an unbalanced dataset of 95,564 human and 76,776 AI texts for either training the K-Nearest Neighbors classifier or fine-tuning. These data come from a variety of sources and include the data from several recent research publications on the development of AI text detection. The average length of a text is 1,388 characters. Sources include news, Reddit comments, reviews, and essays. Generating models include OPT, GPT, GPT-2, GPT-3, ChatGPT, PPLM, Grover, XLM, and XLNet.

Datasets:
* GPT3, Various Topics: https://github.com/openai/gpt-3
* ChatGPT,TOEFL essays: https://github.com/rexshijaku/chatgpt-generated-text-detection-corpus
* Various Models, News: https://github.com/AdaUchendu/Authorship-Attribution-for-Neural-Text-Generation
* ChatGPT, Hotel Reviews: https://osf.io/nrjcw/ 
* ChatGPT, Various Topics: https://huggingface.co/datasets/Hello-SimpleAI/HC3
Note: we used some additional data for training that is not publically available


## Limitations
The classifier is not reliable on short texts (<50 tokens). Even longer texts are sometimes incorrectly labeled by the classifier. Texts longer than 512 tokens (~2,048 characters) will be broken up into 512 token sections and truncated at 10240 tokens total. The sectioned document will be scored as a single document. The classifier was trained on only English text. Text that is very predictable cannot be reliably identified.  Machine-generated text can be intentionally changed to avoid detection. The classifier does best on text that is similar to the training set, and classifiers have been shown to be poorly calibrated on text very different from their training data. The classifier can mislabel both machine-generated and human-written text, and can be very confident in its decision.

## How To Run
Download the models by running the command below. You must download the KNN pkl file. If you would like to run this code fully locally, you must clone 2 HuggingFace repositories. If you are online you do not need to locally download the HuggingFace models.

```bash
sh download_models.sh
```

Next, install the required packages:

```bash
python3 -m venv vluarmachine
. ./vluarmachine/bin/activate
pip3 install -U pip
pip3 install -r requirements.txt
```

Then to make predictions on an input JSON lines file of texts, use the commands below. For training the KNN, you can optionally set the number of neighbors and whether to perform grid search. For inferece, you can specify the threshold desired to deem a text machine, the default is 0.7.

```bash
# For inference with the downloaded model:
python inference.py --file_name file.jsonl --threshold 0.7

# To train your own knn:
python train_luar_knn.py --file_name file.jsonl --n_neighbors 100
```

Each line in the test file should be of the format `{"text": "This is an example text."}`. If training the KNN, each line of the train file should be in the format `{"text": "This is a training example text", "labels" : 1}` where 1 = machine and 0 = human.

## Acknowledgements
This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #D2022-2205150003. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.


## License
This deliverable was developed by Lawrence Livermore National Laboratory (LLNL) under contract D2022-2205150003 for the Intelligence Advanced Research Projects Activity (IARPA)Human Interpretable Attribution of Text using Underlying Structure (HIATUS) program and is being delivered to the Government with UNLIMITED rights.
 
## Citations
@inproceedings{uar-emnlp2021,
  author    = {Rafael A. Rivera Soto and Olivia Miano and Juanita Ordonez and Barry Chen and Aleem Khan and Marcus Bishop and Nicholas Andrews},
  title     = {Learning Universal Authorship Representations},
  booktitle = {EMNLP},
  year      = {2021},
}
 
@misc{AITextClassifier, 
    month     = Jan,
    day       = 31,
    journal   = {AI Text Classifier},
    publisher = {OpenAI},
    url       = https://beta.openai.com/ai-text-classifier, 
    date      = 2023,
} 
 
Markowitz, David M., et al. “Linguistic Markers of Ai-generated Text Versus Human-generated Text: Evidence from Hotel Reviews and News Headlines.” PsyArXiv, 30 Jan. 2023. Web.
 
https://github.com/openai/gpt-2-output-dataset
 
@article{brown2020language,
    title={Language Models are Few-Shot Learners},
    author={Tom B. Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared Kaplan and Prafulla Dhariwal and Arvind Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell and Sandhini Agarwal and Ariel Herbert-Voss and Gretchen Krueger and Tom Henighan and Rewon Child and Aditya Ramesh and Daniel M. Ziegler and Jeffrey Wu and Clemens Winter and Christopher Hesse and Mark Chen and Eric Sigler and Mateusz Litwin and Scott Gray and Benjamin Chess and Jack Clark and Christopher Berner and Sam McCandlish and Alec Radford and Ilya Sutskever and Dario Amodei},
    year={2020},
    eprint={2005.14165},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
 
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
 
@article{guo2023hc3,
    title = "How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection",
    author = "Guo, Biyang  and Zhang, Xin and Wang, Ziyuan and Jiang, Minqi and Nie, Jinran and Ding, Yuxuan and Yue, Jianwei and Wu, Yupeng",
    journal={arXiv preprint arxiv:2301.07597}
    year = "2023",
}
 
@article{shijaku2023,
    author = {Shijaku, Rexhep and Canhasi, Ercan},
    year = {2023},
    pages = {},
    title = {ChatGPT Generated Text Detection},
    doi = {10.13140/RG.2.2.21317.52960}
}
 
@inproceedings{uchendu2020authorship,
  title={Authorship Attribution for Neural Text Generation},
  author={Uchendu, Adaku and Le, Thai and Shu, Kai and Lee, Dongwon},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={8384--8395},
  year={2020}
}
