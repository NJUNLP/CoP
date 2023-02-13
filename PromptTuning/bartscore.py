import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from config import Encoder_prefix_prompt_length,Encoder_inter_prompt_length,Decoder_prefix_prompt_length,target_data_set,invalid_sample_id

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024-Encoder_prefix_prompt_length-Encoder_inter_prompt_length, checkpoint='facebook/bart-large-cnn',PromptBART = False,Dev=False):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.PromptBART = PromptBART
        if PromptBART:
            from promptBART import BartPromptForConditionalGeneration
            self.model = BartPromptForConditionalGeneration.from_pretrained(checkpoint,output_attentions=True)
            if Dev is False:
                print("loading from local Model")
                #self.load("/home/shesj/workspace/Data/PLM/PromptBART/temp.pth")
                
                self.load("/home/shesj/workspace/Data/PLM/PromptBART/rebuildShuffle40+40.pth")
                
   
        else:
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint,output_attentions=True)

        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

        

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        print("LOADING")
        self.model.model.load_state_dict(torch.load(path, map_location=self.device))
        print("LOADED")

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
        
    def score(self, srcs, tgts, batch_size=4,inserted = None,summary_level=False):
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
                    insert_position = list(tgt_tokens.shape)[1]-2
                    if inserted and self.PromptBART:
                        output = self.model(
                            input_ids=src_tokens,
                            attention_mask=src_mask,
                            labels=tgt_tokens,
                            prompt_input_id = insert_position
                        )
                    else:
                        output = self.model(
                            input_ids=src_tokens,
                            attention_mask=src_mask,
                            labels=tgt_tokens
                        )
                    
                    #print(self.tokenizer.convert_tokens_to_string(tgt_tokens))
                    #print(output)

                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    

                    values, indices = loss.topk(1, dim=1, largest=True, sorted=True)
                    loss_values = loss.cpu().numpy().tolist()
                    loss = loss.sum(dim=1) / tgt_len

                    
                    filtered_token_list = self.decode(tgt_tokens[0])
                    max_loss_token = [filtered_token_list[i] for i in indices.cpu().numpy().tolist()[0]]
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list
                    if summary_level:
                        return score_list
                    return filtered_token_list,loss_values,indices.cpu().numpy().tolist()[0]
                    
            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list