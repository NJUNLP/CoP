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
from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("/home/shesj/workspace/Data/PLM/BART")


def is_skip(input_text):
    encoded_tgt = tokenizer(
                    [input_text],
                    max_length=1024,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
    tgt_tokens = encoded_tgt['input_ids']

    token_list = []
    for j in tgt_tokens[0]:
        token_list.append(tokenizer._convert_id_to_token(j.cpu().numpy().tolist()))
    filtered_token_list = []
    for i in token_list:
        filtered_token_list.append(tokenizer.convert_tokens_to_string([i]).strip())
    filtered_token_list,token2bpe,bpe2tokens = convert_bpe2_tokens_space([input_text],filtered_token_list)
    if token2bpe is None:
        return True
    else:
        return False

def create_toy_data():
    f = open("/home/shesj/workspace/Data/Data/Sort_XSUM/TConvS2S/TConvS2S.source",'r')
    source = f.readlines()
    f = open("/home/shesj/workspace/Data/Data/Sort_XSUM/TConvS2S/TConvS2S.target",'r')
    target = f.readlines()
    max_sample = 10
    source = [i.strip() for i in source]
    target = [i.strip() for i in target]
    
    f = open("data/toy.json",'w')
    data = []
    for i,j in zip(source,target):
        data.append({'doc':i,'sum':j})

    data = data[:max_sample]
    import json
    json.dump(data,f,indent=2)

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
        high_bart_scorer = BARTScorer(device='cuda:0', checkpoint=args.model_path)
    else:
        from model.PromptBARTScore import BARTScorer
        high_bart_scorer = BARTScorer(checkpoint=args.model_path,PromptBART = True)
    return high_bart_scorer


count = 0

def predict_with_known_num(filtered_token_list,pre_score,after_score,summary):
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

def zero_shot_predictor(d,s,high_bart_scorer):
    filtered_token_list,pre_score,_ = high_bart_scorer.score([d],[s])
    filtered_token_list,after_score,_ = high_bart_scorer.score([s + " " + d],[s])
    predict_score = predict_with_known_num(filtered_token_list,pre_score,after_score,s)
    return predict_score

def few_shot_predictor(d,s,high_bart_scorer):
    filtered_token_list,pre_score,_ = high_bart_scorer.score([d],[s])
    filtered_token_list,after_score,_ = high_bart_scorer.score([s + " " + d],[s],inserted = s)
    predict_score = predict_with_known_num(filtered_token_list,pre_score,after_score,s)
    return predict_score

def zero_shot_SummaryPredictor(d,s,high_bart_scorer):
    pre_score = high_bart_scorer.score([d],[s],summary_level=True)
    after_score = high_bart_scorer.score([s + " " + d],[s],summary_level=True)
    assert len(pre_score) == 1
    dif_scores = pre_score[0] - after_score[0]
    return dif_scores
    
def predict(args,high_bart_scorer):
    documents = []
    summarys = []
    result_list = []
    
    import json
    f = open(args.data_path,'r')
    data = json.load(f)
    for i in data:
        documents.append(i['doc'])
        summarys.append(i['sum'])

    sample_id = -1

    corpus_predict = []
    total_pretent_predict_sample = 0

    from tqdm import tqdm
    valid_sample = 0

    total_not_in_source_predict = []
    total_prob_score = []

    print(len(documents))

    dataset_level_information = []
    for i in tqdm(range(len(documents))):
        sample_id += 1
        document = documents[i]
        summary = summarys[i]
        if is_skip(summary):
            print("HIT SKIP")
            continue
        pre_sum = summary
        if args.Recapital:
            summary = summary.capitalize()
            summary = re_upper(document,summary)


        
        valid_sample += 1
        d = document
        s = summary

        predict_label1 = NotInSource(d,s)
        total_not_in_source_predict = total_not_in_source_predict + predict_label1

        sample_info = {}

        if args.mode == "zero-shot":
            predict_score = zero_shot_predictor(d,s,high_bart_scorer)
            summary_level_score = zero_shot_SummaryPredictor(d,s,high_bart_scorer)

        if args.mode == 'full-shot':
            predict_score = few_shot_predictor(d,s,high_bart_scorer)
        

        total_prob_score += predict_score
        
        
        
        sample_info['predict_score'] = predict_score
        #sample_info['not_in_score'] = predict_label1
        sample_info['document'] = d
        sample_info['summary'] = s
        sample_info['summary_score'] = summary_level_score
        assert len(sample_info['predict_score']) == len(sample_info['summary'].split(" "))
        
        dataset_level_information.append(sample_info)

    for no,pro in zip(total_not_in_source_predict,total_prob_score):
        sample_info = {}
        sample_info['not_in'] = no
        sample_info['prob_s'] = pro
        corpus_predict.append(sample_info)

    not_in_predict = sum(total_not_in_source_predict)
    expect_predict_num = int(len(total_not_in_source_predict) * args.predict_raio)
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
    
    for sample in dataset_level_information:
        #print("hit")
        tokens = sample['summary'].split(' ')
        predic = sample['predicted_label']
        assert len(tokens) == len(predic)
        tokens_labeled = [i + "[" + str(j) + ']' for i,j in zip(tokens,predic)]
        sample['Pre_lab_summary'] = " ".join(tokens_labeled)
        del sample['predicted_label']
        del sample['predict_score']
    f = open(args.output_file_path,'w')
    import json
    json.dump(dataset_level_information,f,indent=2)




def main(args):
    model = load_model(args)
    predict(args,model)


if __name__ == "__main__":
    create_toy_data()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
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

    parser.add_argument(
        "--output_file_path",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--predict_raio",
        type=float,
        default=None,
        required=True
    )

    parser.add_argument("--Recapital",
                        action="store_true",
                        help="When Summary is lowercase, we Recapital it")

    args = parser.parse_args()
    main(args)