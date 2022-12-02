from transformers import BartTokenizer, PegasusTokenizer
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration
# from transformers import BartTokenizer, PegasusTokenizer
# from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration

# IS_CNNDM = False # whether to use CNNDM dataset or XSum dataset
# ARTICLE_TO_SUMMARIZE = "Manchester United superstar Cristiano Ronaldo scored his 806th career goal in Old Trafford,\
#  breaking FIFA's all-time record for most goals in competitive matches in men's football history.\
#  It was the second of three goals the Portuguese attacker scored during the game,\
#  leading United to a 3-2 victory over Tottenham and finishing the day with 807 total career goals.\
#  The previous FIFA goal record was held by Josef Bican, with 805 goals."

# # Load our model checkpoints
# if IS_CNNDM:
#     model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
#     tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
# else:
#     model = PegasusForConditionalGeneration.from_pretrained('/home/shesj/workspace/Data/PLM/BRIOXSUM')
#     tokenizer = PegasusTokenizer.from_pretrained('/home/shesj/workspace/Data/PLM/BRIOXSUM')

# max_length = 1024 if IS_CNNDM else 512
# # generation example
# if IS_CNNDM:
#     article = ARTICLE_TO_SUMMARIZE.lower()
# else:
#     article = ARTICLE_TO_SUMMARIZE
# inputs = tokenizer([article], max_length=max_length, return_tensors="pt", truncation=True)
# # Generate Summary
# summary_ids = model.generate(inputs["input_ids"])
# print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
# exit()
import torch.nn as nn
import torch
class BARTScorer:
    def __init__(self, device='cpu', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length

        self.model = BartForConditionalGeneration.from_pretrained('/home/shesj/workspace/Data/PLM/BRIOCNN')
        self.tokenizer = BartTokenizer.from_pretrained('/home/shesj/workspace/Data/PLM/BRIOCNN')

        # self.model = PegasusForConditionalGeneration.from_pretrained('/home/shesj/workspace/Data/PLM/BRIOXSUM')
        # self.tokenizer = PegasusTokenizer.from_pretrained('/home/shesj/workspace/Data/PLM/BRIOXSUM')
        # max_input_len = self.tokenizer.max_len_single_sentence
        # print(max_input_len)
        # exit()

        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def get_candidate(self,logits):
        """
        @description  : Get a sub largest prob String Output
        ---------
        @param  : Prob Distribution
        -------
        @Returns  : decocded max-prob token-id
        -------
        """
        prob_ = self.lsm(logits)
        #print(prob_.shape)
        values, indices = prob_.topk(10, dim=1, largest=True, sorted=True)
        #print(values)
        #print(indices.shape)
        return indices


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
        for i in token_list:
            filtered_token_list.append(self.tokenizer.convert_tokens_to_string([i]).strip())

        return filtered_token_list
        
        
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
                    # print(src_tokens.shape)
                    # #print()
                    # print(tgt_tokens.shape)
                    # print("YYY")
                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    
                    #print(self.tokenizer.convert_tokens_to_string(tgt_tokens))
                    #print(output)
                    # print("TEST")
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    

                    values, indices = loss.topk(1, dim=1, largest=True, sorted=True)
                    loss_values = loss.cpu().numpy().tolist()
                    loss = loss.sum(dim=1) / tgt_len

                    
                    # filtered_token_list = self.decode(tgt_tokens[0])
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
