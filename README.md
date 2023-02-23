# CoP: Factual Inconsistency Detection by Controlling the Preference
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Active](http://img.shields.io/badge/Status-Active-green.svg)](https://tterb.github.io) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)


This directory contains code necessary to replicate the training and evaluation for the AAAI 2023 paper:

"**CoP: Factual Inconsistency Detection by Controlling the Preference**" by **[Shuaijie She](https://ricardokevins.github.io/), [Xiang Geng](https://scholar.google.com.hk/citations?hl=zh-CN&user=n6QnFS0AAAAJ), [Shujian Huang](http://nlp.nju.edu.cn/huangsj/) and [Jiajun Chen](https://cs.nju.edu.cn/chenjiajun/index.htm)**.



> **I'm reorganizing the code for simplicity and convenience. I will release it gradually.**

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



# Guide
## Preparation
Download Pretrain Model from Huggingface (for example [BARTCNN](https://huggingface.co/facebook/bart-large-cnn))

## How to Evaluation
### Evaluate on Token-level task
TD
### Evaluate on Summary-level Task
Using the script reproduce.sh
--TestOn support four data split mentioned in paper, including ['qagscnn','qagsxsum','frankcnn','frankxum']


### Evaluate on Inconsistency Category Task

## How to Use
We provide a simple inference usage as inference.sh (currently support Zero-shot token&summary Level Tasks)
```
1. Prepare data (a simple example in data/toy.json)
2. Specify Config in inference.sh
3. Create output Folder
4. Exec inference.sh
5. Check the result in output/result.json
```

## How to Train with Prompt Tuning
Looking into PromptTuning folder.

Our experiments were conducted on single 3090 and take around 10G V-Memory (based on BARTCNN)

 
# Citation
If you find our work useful, please consider citing our work.

```
@misc{she2022cop,
      title={CoP: Factual Inconsistency Detection by Controlling the Preference}, 
      author={Shuaijie She and Xiang Geng and Shujian Huang and Jiajun Chen},
      year={2022},
      eprint={2212.01611},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@article{to update with the AAAI2023 Processings,
  title={==},
  author={==},
  journal={==},
  year={==}
}
```

<!-- # Experimental Results
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

 -->
