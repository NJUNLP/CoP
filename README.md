# CoP: Factual Inconsistency Detection by Controlling the Preference
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Active](http://img.shields.io/badge/Status-Active-green.svg)](https://tterb.github.io) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)


This directory contains code necessary to replicate the training and evaluation for the AAAI 2023 paper:

"**CoP: Factual Inconsistency Detection by Controlling the Preference**" by **[Shuaijie She](https://ricardokevins.github.io/), Xiang Geng and [Shujian Huang](http://nlp.nju.edu.cn/huangsj/), Jiajun Chen**.


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
Download Pretrain Model
## How to Evaluation

## How to Train

# Experimental Results
## Token-Level

|  Model  |  F1(%)  |
|  :----: | :----:  |
|  BARTSc  |  0.1  |
|  CoP Zero-shot  |  0.1  |
|  CoP Few-shot  |  0.1  |
|  CoP Full-shot  |  0.1  |


## Summary-Level Inconsistency Detection
|  Model  |  Pearson(%)  |
|  :----:  | :----:  |
|  BARTSc  |  0.1  |
|  CoP Zero-shot  |  0.1  |
|  CoP Few-shot  |  0.1  |
|  CoP Full-shot  |  0.1  |

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