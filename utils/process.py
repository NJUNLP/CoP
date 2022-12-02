from nltk.tokenize import sent_tokenize

def filter_none(not_in_source,score,label):
    p_n = []
    p_s = []
    p_l = []
    for i,j,k in zip(not_in_source,score,label):
        if j is not None:
            p_n.append(i)
            p_s.append(j)
            p_l.append(k)
    return p_n,p_s,p_l

filter_upper_token = ['the','a','this','there','an','in','on']
def re_upper(document,summary):
    #document = document.split(' ')
    tokens = summary.split(" ")
    upper_tokens = []
    #print(document)
    #print("canadian".capitalize().replace('.','').replace('\'s','').replace(",",'') in document)
    for i in tokens:
        #print(i)
        if i.capitalize().replace('.','').replace('\'s','').replace(",",'') in document and (i.lower() not in filter_upper_token):
            #print('hit')
            i = i.capitalize()
        upper_tokens.append(i)
    #print(" ".join(upper_tokens))
    
    return " ".join(upper_tokens)

def get_filter_RougeScore(hypothesis,reference):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    #return (scores[0]['rouge-1']['p']+scores[0]['rouge-2']['p']+scores[0]['rouge-l']['p'])/3
    return scores[0]['rouge-2']['p']

def sentence_token_nltk(input_str):
    #print(input_str)
    sent_tokenize_list = sent_tokenize(input_str)
    return sent_tokenize_list

def filter_sentence(input_document,hypo):
    sentence_list = sentence_token_nltk(input_document)
    coherence_score = {}
    for index,sentence in enumerate(sentence_list):
        try:
            score = get_filter_RougeScore(sentence,hypo)
        except:
            score = -1
        coherence_score[index] = score

    sorted_result = sorted(coherence_score.items(), key = lambda kv:(kv[1], kv[0]),reverse = True)
    filterd = sorted_result[:7]
    sentence_id = [i[0] for i in filterd]
    sentence_id.sort()
    filterd_list =  [sentence_list[i] for i in sentence_id]
    #print(filterd_list)
    return " ".join(filterd_list).strip()

def align_re_space(input_string):
    import regex as re
    pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    
    raw_string = input_string.strip()
    re_tokens = re.findall(pat, raw_string)
    print(re_tokens)
    print(raw_string.split(" "))
    exit()