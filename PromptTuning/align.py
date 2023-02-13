#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : Align tokenized Gpt2-bpe-tokens with Space-tokenized tokens
@Date               : 2021/12/20 22:54:19
@Author             : Kevinpro
@version            : 1.0
'''

def do_align(filtered_tokens,bpe_tokens):
    matched_id = 0
    token2bpe = {}
    bpe2tokens = {}
    cur_pool = []
    cur_string = ""
    
    index = 0
    for i in bpe_tokens:
        if i.strip() in filtered_tokens[matched_id]:
            cur_pool.append(index)
            bpe2tokens[index] = matched_id
            cur_string = cur_string + i
            if cur_string == filtered_tokens[matched_id]:
                token2bpe[matched_id] = cur_pool
                cur_pool = []
                cur_string = ""
                matched_id += 1
            index += 1
        else:
            # print("===============hit bug==============")
            # print(i," ",filtered_tokens[matched_id])
            # print(filtered_tokens)
            # print(bpe_tokens)
            # print(cur_pool,cur_string)
            # print(bpe2tokens)
            # print(token2bpe)
            # exit()
            return None,None,None
            #exit()
            
    
    # for i in token2bpe:
    #     print(filtered_tokens[i],"|".join([bpe_tokens[j] for j in token2bpe[i]]))
    return filtered_tokens,token2bpe,bpe2tokens

def convert_bpe2_tokens_re(raw_string,bpe_tokens,bpe_ids = None):
    """
    @description  : Align single bpe-id to raw-string(split with re-expression)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """

    import regex as re
    pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    
    raw_string = raw_string[0].strip()
    space_tokenized = re.findall(pat, raw_string)
    # filter all " " character in Token:
    filtered_tokens = [i.replace(" ","") for i in space_tokenized]
    filtered_tokens = ['<s>'] + filtered_tokens + ["</s>"]
    return do_align(filtered_tokens,bpe_tokens)

def convert_bpe2_tokens_space(raw_string,bpe_tokens,bpe_ids = None):
    """
    @description  : Align single bpe-id to raw-string(split by spcae)
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """

    
    raw_string = raw_string[0].strip()
    space_tokenized = raw_string.split()
    # filter all " " character in Token:
    filtered_tokens = [i.replace(" ","") for i in space_tokenized]
    filtered_tokens = ['<s>'] + filtered_tokens + ["</s>"]
    return do_align(filtered_tokens,bpe_tokens)

def align_values_frombpe2tokens(values,token2bpe,bpe2tokens):
    """
    @description  : convert and combine bpe-values to tokens-values
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # input-values is align with bpe's shape
    tokens_values = []
    # print(values)
    # print(token2bpe)
    # print(bpe2tokens)
    # exit()
    for i in token2bpe:
        temp_values = [values[j] for j in token2bpe[i]]
        # Combine way : Max or Min or Sum
        #temp_value = sum(temp_values)
        #temp_value = sum(temp_values)/len(temp_values)
        temp_value = max(temp_values)
        tokens_values.append(temp_value)
    assert len(tokens_values) == len(token2bpe)
    return tokens_values


    
    

def testcase():
    from transformers import BartTokenizer
    checkpoint = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(checkpoint)
    testdata = ["Sainsbury's has reported a fall in half-year profits, helped by a rise in home sales at Argos and Argos."]
    encoded_src = tokenizer(
        testdata,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    src_tokens = encoded_src['input_ids']
    token_list = []
    token_index = src_tokens[0]
    for j in token_index:
        token_list.append(tokenizer._convert_id_to_token(j.cpu().numpy().tolist()))
    filtered_token_list = []
    for i in token_list:
        filtered_token_list.append(tokenizer.convert_tokens_to_string([i]).strip())

    filtered_tokens,token2bpe,bpe2tokens = convert_bpe2_tokens_re(testdata,filtered_token_list)
    filtered_tokens,token2bpe,bpe2tokens = convert_bpe2_tokens_space(testdata,filtered_token_list)

#testcase()