from promptBART import BartPromptForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from tqdm import tqdm 
import torch
from config import Encoder_prefix_prompt_length,Encoder_inter_prompt_length,Decoder_prefix_prompt_length,target_data_set,invalid_sample_id
from utils import pass_loss_Entity
import torch
import numpy as np
import random
from check_disk import get_free_space_mb
from torch.utils.tensorboard import SummaryWriter   
from align import convert_bpe2_tokens_space
from align import align_values_frombpe2tokens

task_name = 'New Task'
writer = SummaryWriter("./log/" + task_name)
log_w = open("./log/" + "{}.txt".format(task_name),'w')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # 设置随机数种子
setup_seed(42)

device = 'cuda:0'
checkpoint = '../../../../Data/PLM/BARTCNN'
model = BartPromptForConditionalGeneration.from_pretrained(checkpoint).to(device)
tokenizer = BartTokenizer.from_pretrained(checkpoint)
loss_fct = nn.NLLLoss(reduction='none', ignore_index=model.model.config.pad_token_id)
lsm = nn.LogSoftmax(dim=1)

def decode(token_index):
    token_list = []
    for j in token_index:
        token_list.append(tokenizer._convert_id_to_token(j.cpu().numpy().tolist()))
    filtered_token_list = []
    for i in token_list:
        filtered_token_list.append(tokenizer.convert_tokens_to_string([i]).strip())
    return filtered_token_list

# pretrained = torch.load('./data/prompt.pt')
# model.model.encoder.prefix_encoder.weight.data.copy_(pretrained)

for name, param in model.named_parameters():
    if 'prefix_encoder' in name:
        param.requires_grad = True
    elif 'prefix_decoder' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

    #model.resize_token_embeddings(len(tokenizer))


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.model.named_parameters()
                if not any(nd in n for nd in no_decay)],
        "weight_decay": 0},
    {"params": [p for n, p in model.model.named_parameters()
                if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3, eps=1e-8)
#lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=15000,num_training_steps=798000)

def load(target):
    f1 = open("../../../../Data/Data/XsumDetect/{}/{}.source".format(target,target),'r',encoding='utf-8')
    document = f1.readlines()

    f2 = open("../../../../Data/Data/XsumDetect/{}/{}.target".format(target,target),'r',encoding='utf-8')
    summary = f2.readlines()

    f3 = open("../../../../Data/Data/XsumDetect/{}/{}.label".format(target,target),'r',encoding='utf-8')
    label = f3.readlines()

    f4 = open("../../../../Data/Data/XsumDetect/{}/{}.ref".format(target,target),'r',encoding='utf-8')
    reference = f4.readlines()

    document = [i.strip() for i in document]
    summary = [i.strip() for i in summary]
    label = [i.strip() for i in label]
    reference = [i.strip() for i in reference]
    return document,summary,label,reference

documents = []
summarys = []
labels = []

def NotInSource(document,summary):
    document = document
    prediction_label = []
    for i in summary.split(' '):
        if i.replace('.','').replace('\'s','').replace(",",'') not in document:
            prediction_label.append(1)
        else:
            prediction_label.append(0)
    return prediction_label

filter_upper_token = ['the','a','this','there','an','in','on']
def re_upper(document,summary):
    tokens = summary.split(" ")
    upper_tokens = []
    for i in tokens:
        if i.capitalize().replace('.','').replace('\'s','').replace(",",'') in document and (i.lower() not in filter_upper_token):
            i = i.capitalize()
        upper_tokens.append(i)
    return " ".join(upper_tokens)

total_srcs = []
total_tgts = []
total_labs = []

# Read trainset split
f = open("split.txt",'r')
trainset = f.readlines()
trainset = [int(i.strip()) for i in trainset]

sample_id = -1
token_level_sample_number = 0

train_src = []
train_tgt = []
train_label = []

dev_src = []
dev_tgt = []
dev_label = []

test_src = []
test_tgt = []
test_label = []

for dataset in target_data_set:
    documents,summarys,labels,references = load(dataset)
    for i in range(len(documents)):
        sample_id += 1 
        if sample_id in invalid_sample_id:
            continue
        document = documents[i]
        summary = summarys[i]
        reference = references[i]
        label = labels[i]

        target = [int(l) for l in label.split(" ")]
        number = sum(target)

        summary = summary.capitalize()
        summary = re_upper(document,summary)
        
        encoded_tgt = tokenizer(
                    [summary],
                    max_length=1024,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
        tgt_tokens = encoded_tgt['input_ids']
        filtered_token_list = decode(tgt_tokens[0])
        
        filtered_token_list,token2bpe,bpe2tokens = convert_bpe2_tokens_space([summary],filtered_token_list)
        if token2bpe == None:
            print("hit BUG")
            exit()
            # print("INVALID SAMPLE : ",sample_id)
            # continue

        expand_label = []
        assert len(token2bpe) - len(target) == 2
        for index in range(1,len(token2bpe)-1):
            for l in range(len(token2bpe[index])):  
                if target[index-1] == 1: # Unfact
                    expand_label.append(-1)
                else:
                    expand_label.append(1)

        assert len(expand_label) == (len(bpe2tokens)-2)
        if sample_id in trainset:
            total_srcs.append(document)
            total_tgts.append(summary)
            total_labs.append(expand_label)
            token_level_sample_number += len(expand_label)

        else:
            test_src.append(document)
            test_tgt.append(summary)
            test_label.append(target)


train_index = list(range(len(total_srcs)))
random.shuffle(train_index)
for i in range(len(total_srcs)):
    if i in train_index[:400]:
        dev_src.append(total_srcs[i])
        dev_tgt.append(total_tgts[i])
        dev_label.append(total_labs[i])
    else:
        train_src.append(total_srcs[i])
        train_tgt.append(total_tgts[i])
        train_label.append(total_labs[i])

print(len(train_src))
print(len(dev_src))
print(len(test_src))
print(test_tgt[0])
print(test_label[0])
# token_level_sample_number = 0 
# for i in range(len(srcs)-400):
#     token_level_sample_number += len(labs[i])

print("Summary Level Sample : ",len(train_src))

from bartscore import BARTScorer
high_bart_scorer = BARTScorer(checkpoint='../../../../Data/PLM/BARTCNN',PromptBART = True,Dev = True)

def test(log_step):
    nh_correct = 0
    recall_total = 0
    precision_total = 0
    tot_tokens = 0
    ncorrect = 0
    nsamples = 0
    high_bart_scorer.model.model.load_state_dict(model.model.state_dict())
    total_prob= []
    total_label = []
    with torch.no_grad():
        avg_loss = 0
        update_step =  0
        for i in range(len(test_src)):
            filtered_token_list,pre_score,_ = high_bart_scorer.score([test_src[i]],[test_tgt[i]])
            filtered_token_list,after_score,_ = high_bart_scorer.score([test_tgt[i] + " " + test_src[i]],[test_tgt[i]],inserted = test_tgt[i])
            
            filtered_token_list,token2bpe,bpe2tokens = convert_bpe2_tokens_space([test_tgt[i]],filtered_token_list)

            pre_score[0] = align_values_frombpe2tokens(pre_score[0],token2bpe,bpe2tokens)
            pre_score= pre_score[0][1:-1]
            pre_score = pass_loss_Entity(pre_score,filtered_token_list)

            after_score[0] = align_values_frombpe2tokens(after_score[0],token2bpe,bpe2tokens)
            after_score= after_score[0][1:-1]
            after_score = pass_loss_Entity(after_score,filtered_token_list)

            diff_score = [i-j for i,j in zip(pre_score,after_score)]
            total_prob += diff_score
            

            for p in test_label[i]:
                if p == 1:
                    total_label.append(1)
                else:
                    total_label.append(0)
            dot_score = [t*k for t,k in zip(test_label[i],diff_score)]
            avg_loss += sum(dot_score)/(len(dot_score))

            update_step += 1
    idx2score = {}

    expect_predict_num = int(sum(total_label) * 0.55) + sum(total_label)
    for i in range(len(total_prob)):
        idx2score[i] = total_prob[i]
    idx2score = sorted(idx2score.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    index = 0
    #print(idx2score[:5])
    total_not_in_source_predict = [0 for i in range(len(total_prob))]
    for i in idx2score:
        id = i[0]
        total_not_in_source_predict[id] = 1
        expect_predict_num -= 1
        if expect_predict_num == 0:
            break

    target = total_label
    predict_label = total_not_in_source_predict
    print("predict",sum(predict_label))
    target = np.array(target)
    predict_label = np.array(predict_label)
    nh_correct += sum([1 for p, t in zip(predict_label, target) if p == 1 and t == 1])
    recall_total += sum(target == 1)
    precision_total += sum(predict_label == 1)
    tot_tokens += len(predict_label)
    ncorrect += sum(predict_label == target)
    nsamples += len(target)


    acc = float(ncorrect)/float(nsamples)
    recall = float(nh_correct)/float(recall_total)
    precision = float(nh_correct)/float(precision_total)
    f1 = 2 * precision * recall / (precision + recall)

    print("Acc {} Recall {} Precision {} F1 {} ".format(acc,recall,precision,f1))
    log_w.write("Acc {} Recall {} Precision {} F1 {}  \n".format(acc,recall,precision,f1))
    print('Real Testing Loss {0:1.5f}'.format(avg_loss/update_step)) 
    writer.add_scalar("TestLoss", avg_loss/update_step, log_step)
    writer.add_scalar("TestACC", acc, log_step)
    writer.add_scalar("TestRe", recall, log_step)
    writer.add_scalar("TestPr", precision, log_step)
    writer.add_scalar("TestF1", f1, log_step)
    #return avg_loss/update_step
    return avg_loss/update_step

def dev(log_step):
    nh_correct = 0
    recall_total = 0
    precision_total = 0
    tot_tokens = 0
    ncorrect = 0
    nsamples = 0
    high_bart_scorer.model.model.load_state_dict(model.model.state_dict())
    total_prob= []
    total_label = []
    with torch.no_grad():
        avg_loss = 0
        update_step =  0
        for i in range(len(dev_src)):
            filtered_token_list,pre_score,_ = high_bart_scorer.score([dev_src[i]],[dev_tgt[i]])
            filtered_token_list,after_score,_ = high_bart_scorer.score([dev_tgt[i] + " " + dev_src[i]],[dev_tgt[i]],inserted = dev_tgt[i])
            pre_score= pre_score[0][1:-1]          
            pre_score = pass_loss_Entity(pre_score,filtered_token_list)

            after_score= after_score[0][1:-1]
            after_score = pass_loss_Entity(after_score,filtered_token_list)

            diff_score = [i-j for i,j in zip(pre_score,after_score)]
            total_prob += diff_score

            for p in dev_label[i]:
                if p == -1:
                    total_label.append(1)
                else:
                    total_label.append(0)
            dot_score = [t*k for t,k in zip(dev_label[i],diff_score)]
            avg_loss += sum(dot_score)/(len(dot_score))

            update_step += 1
    idx2score = {}

    expect_predict_num = int(sum(total_label) * 0.55) + sum(total_label)
    for i in range(len(total_prob)):
        idx2score[i] = total_prob[i]
    idx2score = sorted(idx2score.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    index = 0
    total_not_in_source_predict = [0 for i in range(len(total_prob))]
    for i in idx2score:
        id = i[0]
        total_not_in_source_predict[id] = 1
        expect_predict_num -= 1
        if expect_predict_num == 0:
            break

    target = total_label
    predict_label = total_not_in_source_predict
    target = np.array(target)
    predict_label = np.array(predict_label)
    nh_correct += sum([1 for p, t in zip(predict_label, target) if p == 1 and t == 1])
    recall_total += sum(target == 1)
    precision_total += sum(predict_label == 1)
    tot_tokens += len(predict_label)
    ncorrect += sum(predict_label == target)
    nsamples += len(target)


    acc = float(ncorrect)/float(nsamples)
    recall = float(nh_correct)/float(recall_total)
    precision = float(nh_correct)/float(precision_total)
    f1 = 2 * precision * recall / (precision + recall)

    print("Acc {} Recall {} Precision {} F1 {} ".format(acc,recall,precision,f1))
    log_w.write("Acc {} Recall {} Precision {} F1 {}  \n".format(acc,recall,precision,f1))
    print('Testing Loss {0:1.5f}'.format(avg_loss/update_step)) 
    writer.add_scalar("DevLoss", avg_loss/update_step, log_step)
    writer.add_scalar("DevACC", acc, log_step)
    writer.add_scalar("DevRe", recall, log_step)
    writer.add_scalar("DevPr", precision, log_step)
    writer.add_scalar("DevF1", f1, log_step)
    #return avg_loss/update_step
    return avg_loss/update_step

train_src = train_src[:300]
train_tgt = train_tgt[:300]
train_label = train_label[:300]

def train(args):
    log_step = 0
    best_loss = 100000000000000
    EPOCH = 5000
    for E in range(EPOCH):
        train_size = len(train_src)
        #train_size = 200
        print(task_name)
        with tqdm(total=train_size, desc="Epoch {}".format(E)) as pbar: 
            avg_loss = 0
            update_step =  0
            train_index = list(range(len(train_src)))
            random.shuffle(train_index)
            print(train_index[:10])
            for i in train_index:
                model.train()
                src_encoded = tokenizer(
                                    [train_src[i]],
                                    max_length=1024-Encoder_prefix_prompt_length-Encoder_inter_prompt_length,
                                    truncation=True,
                                    padding=True,
                                    return_tensors='pt'
                                )
                src_tokens = src_encoded['input_ids'].to(device)
                src_attn_mask = src_encoded['attention_mask'].to(device)

                tgt_encoded = tokenizer(
                                    [train_tgt[i]],
                                    max_length=1024,
                                    truncation=True,
                                    padding=True,
                                    return_tensors='pt'
                                )
                tgt_tokens = tgt_encoded['input_ids'].to(device)
                tgt_attn_mask = tgt_encoded['attention_mask'].to(device)

                src_encoded2 = tokenizer(
                                        [train_tgt[i] + " " + train_src[i]],
                                        max_length=1024-Encoder_prefix_prompt_length-Encoder_inter_prompt_length,
                                        truncation=True,
                                        padding=True,
                                        return_tensors='pt'
                                    )
                src_tokens2 = src_encoded2['input_ids'].to(device)
                src_attn_mask2 = src_encoded2['attention_mask'].to(device)

                token_step = list(tgt_tokens.shape)[1]-2

                
                return_state = model(
                    input_ids=src_tokens,
                    attention_mask=src_attn_mask,
                    labels=tgt_tokens
                )
                logits = return_state['logits'].view(-1, model.model.config.vocab_size)
                loss = loss_fct(lsm(logits), tgt_tokens.view(-1))
                loss = loss.view(tgt_tokens.shape[0], -1)
                loss = loss[:,1:-1]

                
                return_state2 = model(
                    input_ids=src_tokens2,
                    attention_mask=src_attn_mask2,
                    labels=tgt_tokens,
                    prompt_input_id = token_step
                )
            
                logits2 = return_state2['logits'].view(-1, model.model.config.vocab_size)
                loss2 = loss_fct(lsm(logits2), tgt_tokens.view(-1))
                loss2 = loss2.view(tgt_tokens.shape[0], -1)
                loss2 = loss2[:,1:-1]
                loss = loss[0] - loss2[0]
                # 1 --》 unfact  with -1
                # 0--》 fact with 1
                #loss=torch.clamp(loss,-40,40)
                loss = loss * (torch.tensor(train_label[i]).reshape(1,-1).to(loss.device))
                loss = loss.mean()

                writer.add_scalar("LOSS", loss, log_step)
                log_step += 1
                avg_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #lr_scheduler.step()
                update_step += 1
    

                

                pbar.set_postfix({'loss' : '{0:1.5f}'.format(avg_loss/update_step)}) #输入一个字典，显示实验指标
                pbar.update(1)
                log_w.write('EPOCH {} Step {} Loss {} \n'.format(E,update_step,avg_loss/update_step))

                if update_step% 100 == 0:
                    test(log_step)
                    dev_loss = dev(log_step)
                    log_w.write("DevLoss  {} \n".format(dev_loss))              
                    if dev_loss <  best_loss:
                        best_loss = dev_loss
                        free_space = get_free_space_mb("/home")
                        print("FREE SPACE ON DEVICE {}".format(free_space))
                        if free_space < 15:
                            log_w.write("!!!!!!!!!HIT BUG WITH NO SPACE !!!!!!!!! With Loss {}  \n".format(best_loss))
                            print("!!!!!!!!!HIT BUG WITH NO SPACE !!!!!!!!! With Loss {}".format(best_loss))
                        else:
          
                            log_w.write("Saving model With Loss {}  \n".format(best_loss))
                            print("Saving model With Loss {}".format(best_loss))
                            model.save_model("../../../../Data/PLM/PromptBART/{}.pth".format(task_name))


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
        "--training_mode", # few-shot or full-shot
        type=str,
        default="full-shot",
        required=True
    )

    parser.add_argument(
        "--log_path",
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
        "--epoch",
        type=int,
        default=500,
        required=True
    )

    args = parser.parse_args()
    train(args)



