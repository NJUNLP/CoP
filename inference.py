# %%

import argparse

import torch
import torch.nn as nn
from rouge import Rouge
import numpy as np
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


sample_infor_dump = []
def load_model(args):
    optional_mode = ['zero-shot','full-shot']
    if args.mode not in optional_mode:
        print("hit unknow mode not in ",optional_mode)
        exit()
    
    if args.mode == "zero-shot":
        from model.BaselineBARTScorer import BARTScorer
        # from model.BRIO import BARTScorer
        # from model.PegasusScorer import BARTScorer
        high_bart_scorer = BARTScorer(device='cuda:0', checkpoint=arg.model_path)
    else:
        from model.PromptBARTScore import BARTScorer
        high_bart_scorer = BARTScorer(checkpoint=arg.model_path,PromptBART = True)
    return high_bart_scorer


count = 0

def predict_with_known_num(filtered_token_list,pre_score,after_score,summary,number):
    filtered_token_list,token2bpe,bpe2tokens = convert_bpe2_tokens_space([summary],filtered_token_list)

    pre_score[0] = align_values_frombpe2tokens(pre_score[0],token2bpe,bpe2tokens)
    pre_score= pre_score[0][1:-1]
    pre_score,temp_count = pass_loss_Entity(pre_score,filtered_token_list)

    after_score[0] = align_values_frombpe2tokens(after_score[0],token2bpe,bpe2tokens)
    after_score= after_score[0][1:-1]
    after_score,temp_count = pass_loss_Entity(after_score,filtered_token_list)

    diff_score = [j-i for i,j in zip(pre_score,after_score)]
    #diff_score,temp_count = pass_loss_Entity(after_score,filtered_token_list)
    return diff_score

def zero_shot_predictor(d,s):
    filtered_token_list,pre_score,_ = high_bart_scorer.score([d],[s])
    filtered_token_list,after_score,_ = high_bart_scorer.score([s + " " + d],[s])
    predict_score = predict_with_known_num(filtered_token_list,pre_score,after_score,summary,number)
    return predict_score

def few_shot_predictor(d,s):
    filtered_token_list,pre_score,_ = high_bart_scorer.score([d],[s])
    filtered_token_list,after_score,_ = high_bart_scorer.score([s + " " + d],[s],inserted = s)
    predict_score = predict_with_known_num(filtered_token_list,pre_score,after_score,summary,number)
    return predict_score

def predict(args):
    documents = []
    summarys = []
    result_list = []
    
    import json
    f = open(arg.data,'r')
    data = json.load(f)
    for i in data:
        documents.append(i['doc'])
        summarys.append(i['sum'])

    sample_id = -1

    corpus_predict = []
    total_pretent_predict_sample = 0

    from tqdm import tqdm
    valid_sample = 0

    total_predict_label = []
    total_not_in_source_predict = []
    total_prob_score = []

    #
    print(len(documents))

    dataset_level_information = []
    for i in tqdm(range(len(documents))):
        sample_id += 1
        if sample_id in invalid_sample_id:
            continue
        if sample_id in trainset:
            continue

        sentence_level += 1
        document = documents[i]
        summary = summarys[i]

        pre_sum = summary
        summary = summary.capitalize()
        summary = re_upper(document,summary)

        label = labels[i]
        target = [int(l) for l in label.split(" ")]
        number = sum(target)
            
        total_predict_label = total_predict_label + target

        
        valid_sample += 1
        d = document
        s = summary

        predict_label1 = NotInSource(d,s)
        total_not_in_source_predict = total_not_in_source_predict + predict_label1

        sample_info = {}

        if arg.mode == "zero-shot":
            predict_score,pre_score,after_score = zero_shot_predictor(d,s)
            sample_info['pre_score'] = pre_score
            sample_info['after_score'] = after_score
            total_after_score += after_score
            total_pre_score += pre_score

        if arg.mode == 'full-shot':
            predict_score = few_shot_predictor(d,s)
        

        total_prob_score += predict_score
        
        
        
        sample_info['data'] = dataset
        sample_info['predict_score'] = predict_score
        sample_info['not_in_score'] = predict_label1
        sample_info['target_label'] = target
        sample_info['document'] = d
        sample_info['summary'] = s
        
        dataset_level_information.append(sample_info)

    for no,pro,la in zip(total_not_in_source_predict,total_prob_score,total_predict_label):
        sample_info = {}
        sample_info['data'] = dataset
        sample_info['not_in'] = no
        sample_info['prob_s'] = pro
        sample_info['label'] = la
        corpus_predict.append(sample_info)

    not_in_predict = sum(total_not_in_source_predict)
    expect_predict_num = int(sum(total_predict_label) * 0.55) + sum(total_predict_label)-not_in_predict
    total_pretent_predict_sample += expect_predict_num
    idx2score = {}
    print(len(total_prob_score))
    for i in range(len(total_prob_score)):
        idx2score[i] = total_prob_score[i]
    idx2score = sorted(idx2score.items(), key = lambda kv:(kv[1], kv[0]),reverse=False)
    index = 0
    for i in idx2score:
        id = i[0]
        if total_not_in_source_predict[id] == 0:
            total_not_in_source_predict[id] = 1
            expect_predict_num -= 1
        if expect_predict_num == 0:
            break

    iter_index = 0
    for sample in dataset_level_information:
        sample['predicted_label'] = total_not_in_source_predict[iter_index:iter_index+len(sample['predict_score'])]
        iter_index += len(sample['predict_score'])

    f = open(arg.output_file_path,'w')
    import json
    json.dump(dataset_level_information,f,indent=2)



def main(args):
    #model = load_model(args)
    print(args.data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        required=True,
        help="The path of the source articles.",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        required=True,
        help="The path of the summaries to be evaluated.",
    )
    parser.add_argument(
        "--cmlm_model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mlm_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--knn_model_path",
        type=str,
        required=True,
    )
    
    args = parser.parse_args()
    main(args)