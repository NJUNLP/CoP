# %%
import torch
import torch.nn as nn
import traceback
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from scipy.stats import pearsonr, spearmanr, kendalltau

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint=None):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = PegasusTokenizer.from_pretrained('/home/shesj/workspace/Data/PLM/PEGASUS_CNN')
        self.model = PegasusForConditionalGeneration.from_pretrained('/home/shesj/workspace/Data/PLM/PEGASUS_CNN')
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self):
        """ Load model from paraphrase finetuning """
        self.model.load_state_dict(torch.load('models/bart.pth', map_location=self.device))

    def decode(self,token_index):
        """
        @description  : Use tokenizer to decode the token_index
        ---------
        @param  : 
            tokenindex: tensor
        -------
        @Returns  : token_list
        -------
        """
        token_list = []
        # print(type(tgt_tokens))
        # print(tgt_tokens.shape)
        #for temp in token_index:
            #print(temp[0])
        #print(token_index.shape)
        for j in token_index:
            token_list.append(self.tokenizer._convert_id_to_token(j.cpu().numpy().tolist()))
        filtered_token_list = []
        # print("TOKEN_LIST",token_list)
        # print("UNSTRIP filter ",[self.tokenizer.convert_tokens_to_string([i]) for i in token_list])
        for i in token_list:
            filtered_token_list.append(self.tokenizer.convert_tokens_to_string([i]).strip())
        '''
        TOKEN_LIST ['▁Rory', '▁mc', 'ilroy', '▁Will', '▁take', '▁a', '▁one', '-', 'shot', '▁lead', '▁into', '▁the', '▁final', '▁round', '▁of', '▁the', '▁w', 'gc', '-', 'hs', 'bc', '▁champions', '▁after', '▁card', 'ing', '▁a', '▁three', '-', 'under', '</s>']
        UNSTRIP filter  ['Rory', 'mc', 'ilroy', 'Will', 'take', 'a', 'one', '-', 'shot', 'lead', 'into', 'the', 'final', 'round', 'of', 'the', 'w', 'gc', '-', 'hs', 'bc', 'champions', 'after', 'card', 'ing', 'a', 'three', '-', 'under', '']
        '''
        return filtered_token_list[:-1]

    def score(self, srcs, tgts, batch_size=4,summary_level=False):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    

                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    

                    values, indices = loss.topk(1, dim=1, largest=True, sorted=True)
                    loss_values = loss.cpu().numpy().tolist()
                    loss = loss.sum(dim=1) / tgt_len

                    
                    filtered_token_list = self.decode(tgt_tokens[0])
                    # print(tgt_tokens[0])
                    # print(filtered_token_list)
                    # max_loss_token = [filtered_token_list[i] for i in indices.cpu().numpy().tolist()[0]]
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list
                    if summary_level:
                        loss_values = loss_values[0]

                        loss_values.sort(reverse=True)
                        
                        #loss_values = sum(loss_values[:11])/5
                        loss_values = sum(loss_values)/len(loss_values)
                        return [-loss_values]
                        #return score_list
                    
                    
                    return filtered_token_list,loss_values,indices.cpu().numpy().tolist()[0]
            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list



# bartscore = BARTScorer()
# import json
# TestOn = 'qagscnn'

# if TestOn == 'rank':
#     f = open("/root/SheShuaijie/workspace/Robust/data/data-dev.jsonl",'r',encoding = 'utf-8')
#     lines = f.readlines()
#     lines = [i.strip() for i in lines]

#     source_lines = []
#     target_lines = []
#     for i in lines:
#         data_dict = json.loads(i)
#         source_lines.append(data_dict['text'])
#         target_lines.append(data_dict['claim'])

#     scores = bartscore.score(source_lines,target_lines,batch_size=4)
#     assert len(scores) == len(source_lines)

#     postive_probs = []
#     negtive_probs = []

#     i = 0
#     while i < len(scores):
#         postive_probs.append(scores[i])
#         negtive_probs.append(scores[i+1])
#         i += 2

#     print(len(postive_probs))
#     print(len(negtive_probs))

#     acc = 0
#     for i,j in zip(postive_probs,negtive_probs):
#         if i > j:
#             acc += 1
#     print(acc/len(postive_probs))

# if TestOn == 'qagscnn':
#     f = open("/root/SheShuaijie/workspace/Robust/data/data-dev.jsonl",'r',encoding = 'utf-8')
#     lines = f.readlines()
#     lines = [i.strip() for i in lines]

#     source_lines = []
#     target_lines = []
#     human_scores = []
#     for i in lines:
#         data_dict = json.loads(i)
#         source_lines.append(data_dict['text'])
#         target_lines.append(data_dict['claim'])
#         human_scores.append(data_dict['score'])

#     scores = bartscore.score(source_lines,target_lines,batch_size=4)
#     assert len(scores) == len(source_lines)
#     pearson, _ = pearsonr(scores, human_scores)
#     print(pearson)
