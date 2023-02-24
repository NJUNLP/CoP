import argparse
import torch
import torch.nn as nn
from rouge import Rouge
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from transformers import BartTokenizer, BartForConditionalGeneration
from model.config import Encoder_prefix_prompt_length,Encoder_inter_prompt_length,Decoder_prefix_prompt_length,target_data_set,invalid_sample_id
from utils.wn import predict_with_sync_baseline
from utils.process import *
from utils.dataload import load
from utils.align import convert_bpe2_tokens_space
from utils.align import align_values_frombpe2tokens
from utils.entity_loss import pass_loss_Entity
from utils.wordnet import return_synonyms
from utils.overlap_predictor import NotInSource
from utils.overlap_predictor import returnNone_baseline
from utils.overlap_predictor import synonyms_notin
from utils.process import re_upper
from utils.process import filter_none
from transformers import BartTokenizer

def load_model(args):
    optional_mode = ['zero-shot']
    if args.mode not in optional_mode:
        print("hit unknow mode not in ",optional_mode)
        exit()
    from model.BaselineBARTScorer import BARTScorer
    model = BARTScorer(device='cuda:0', checkpoint=args.model_path)
    return model
    
def zero_shot_predictor(d,s,model):
    pre_score = model.score([d],[s],summary_level=True)
    after_score = model.score([s + " " + d],[s],summary_level=True)
    assert len(pre_score) == 1
    dif_scores = pre_score[0] - after_score[0]
    return dif_scores

def few_shot_predictor(d,s):
    pass

def ent_zero_shot_predictor(d,s,new_d):
    pass

def coref_zero_shot_predictor(d,s,new_d):
    pass

import json

from tabulate import tabulate
result = []

def main(args):
    model = load_model(args)
    print("Testing on {}".format(args.TestOn))
    if args.TestOn not in ['qagscnn','qagsxsum','frankcnn','frankxsum']:
        print("ERROR")
        exit()
    if args.TestOn == 'qagscnn' or args.TestOn == 'qagsxsum':
        if args.TestOn == 'qagscnn':
            f = open("./data/QAGSCNN.jsonl",'r',encoding = 'utf-8')
        if args.TestOn == 'qagsxsum':
            f = open("./data/QAGSXSUM.jsonl",'r',encoding = 'utf-8')
        lines = f.readlines()
        lines = [i.strip() for i in lines]

        source_lines = []
        target_lines = []
        human_scores = []
        
        for i in lines:
            data_dict = json.loads(i)
            source_lines.append(data_dict['text'])
            target_lines.append(data_dict['claim'])
            human_scores.append(data_dict['score'])

        predict_score = []
        from tqdm import tqdm 
        for i in tqdm(range(len(source_lines))):
            d = source_lines[i]
            s = target_lines[i]
            if args.mode == 'zero-shot':
                temp = zero_shot_predictor(d,s,model)
            predict_score.append(temp)

        assert len(predict_score) == len(source_lines)
        pearson, _ = pearsonr(predict_score, human_scores)
        print(pearson)

    if args.TestOn == 'frankcnn' or args.TestOn =='frankxsum':
        if args.TestOn == 'frankcnn':
            f = open("./data/FRANKCNN.jsonl",'r',encoding = 'utf-8')
        if args.TestOn == 'frankxsum':
            f = open("./data/FRANKXSUM.jsonl",'r',encoding = 'utf-8')
        lines = f.readlines()
        lines = [i.strip() for i in lines]
        source_lines = []
        target_lines = []
        human_scores = []

        extra_source = []
        for i in lines:
            data_dict = json.loads(i)
            source_lines.append(data_dict['text'])
            # Summary in Frank Dataset is all lowercased, while the BART-CNN generation model is case sensitive
            # so we simply re-upper case the summary for fair comparison
            data_dict['claim'] = data_dict['claim'].capitalize()
            data_dict['claim'] = re_upper(data_dict['text'],data_dict['claim'])
            target_lines.append(data_dict['claim'])
            human_scores.append(data_dict['score'])

        predict_score = []
        from tqdm import tqdm 
        print(len(source_lines))
        for i in tqdm(range(len(source_lines))):
            d = source_lines[i]
            s = target_lines[i]
            if args.mode == 'zero-shot':
                temp = zero_shot_predictor(d,s,model)
            predict_score.append(temp)
        
        assert len(predict_score) == len(source_lines)
        pearson, _ = pearsonr(predict_score, human_scores)
        print(pearson)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--TestOn",
        type=str,
        default=None,
        required=True
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True
    )
    args = parser.parse_args()
    main(args)