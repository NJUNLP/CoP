import en_core_web_sm
nlp = en_core_web_sm.load()

count = 0
def pass_loss_Entity(score,tokens):
    global count
    tokens = tokens[1:-1]
    # print(score)
    # print(tokens)
    # summary = "The former chief executive of the San Francisco court, Charney Charney, has been cleared by a court in California."
    sents = ' '.join(tokens)
    doc = nlp(sents)
    data_dict = {}
    extract_features = []
    for X in doc.ents:
        extract_features.append(X.text)
    
    complex_entity = []
    for i in extract_features:
        if len(i.split()) > 1:
            count += 1
            complex_entity.append(i)
    
    
    if len(complex_entity)>=1:
        for j in complex_entity:
            entity_length = len(j.split())
            for i in range(len(tokens)-entity_length+1):
                span = tokens[i:i+entity_length]
                if span == j.split():
                    span_score = score[i:i+entity_length]
                    max_score = max(span_score)
                    for p in range(entity_length):
                        score[i + p] = max_score
        
    return score