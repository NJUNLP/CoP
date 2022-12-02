# CoP: Factual Inconsistency Detection by Controlling the Preference
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Active](http://img.shields.io/badge/Status-Active-green.svg)](https://tterb.github.io) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)


This directory contains code necessary to replicate the training and evaluation for the AAAI 2023 paper:

"**CoP: Factual Inconsistency Detection by Controlling the Preference**" by **[Shuaijie She](https://ricardokevins.github.io/), Xiang Geng and [Shujian Huang](http://nlp.nju.edu.cn/huangsj/), [Jiajun Chen](https://cs.nju.edu.cn/chenjiajun/index.htm)**.


# Dependencies and Setup
```
transformers            4.12.5
torch                   1.11.0
tensorboard             2.9.0
spacy                   3.2.3
en-core-web-sm          3.2.0
nltk                    3.7
rouge                   1.0.1
```



# Reproduce Step
## Preparation
Download Pretrain Model from Huggingface (for example [BARTCNN](https://huggingface.co/facebook/bart-large-cnn))

## How to Evaluation
Evaluate on Token-level task


Evaluate on Summary-level Task

Evaluate on Inconsistency Category Task

# How to Use

## How to Train with Prompt Tuning
Looking into PromptTuning folder

# Experimental Results
## Token-Level

|  Model  |  F1(%)  |
|  :----: | :----:  |
|  DAE-Weak  |  59.10  |
|  BARTSc  |  59.25  |
|  EntFA  |  60.23  |
|  CoP Zero-shot  |  63.72  |
|  DAE  |  65.00  |
|  CoP Few-shot  |  66.56  |
|  CoP Full-shot  |  69.61  |


## Summary-Level Inconsistency Detection
|  Model  |  QAGSCNN |  QAGSXSUM |  FRANKCNN |  FRANKXSUM |
|  :----:  | :----:  | :----:  | :----:  | :----:  |
|  BERTScore  |  0.1  |  0.1  |  0.1  |  0.1  |
|  QAGSScore  |  0.1  |  0.1  |  0.1  |  0.1  |
|  BARTScore  |  0.1  |  0.1  |  0.1  |  0.1  |
|  CoCoScore  |  0.1  |  0.1  |  0.1  |  0.1  |
|  Ours  |  0.1  |  0.1  |  0.1  |  0.1  |

## Inconsistency Category Detection

|  Model  |  Overall  |  EntE  |   OutE  |
|  :----:  | :----:  | :----:  | :----:  |
|  BARTSc  |  0.1  |  0.1  |  0.1  |
|  CoP Zero-shot  |  0.1  |  0.1  |  0.1  |
|  CoP Few-shot  |  0.1  |  0.1  |  0.1  |
|  CoP Full-shot  |  0.1  |  0.1  |  0.1  |

|  Model  |  Overall  |  CorefE  |   OutE  |
|  :----:  | :----:  | :----:  | :----:  |
|  BARTSc  |  0.1  |  0.1  |  0.1  |
|  CoP Zero-shot  |  0.1  |  0.1  |  0.1  |
|  CoP Few-shot  |  0.1  |  0.1  |  0.1  |
|  CoP Full-shot  |  0.1  |  0.1  |  0.1  |