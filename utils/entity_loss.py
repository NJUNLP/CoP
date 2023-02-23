import en_core_web_sm
nlp = en_core_web_sm.load()
# import en_core_web_trf
# nlp = en_core_web_trf.load()
'''
PERSON:      People, including fictional.
NORP:        Nationalities or religious or political groups.
FAC:         Buildings, airports, highways, bridges, etc.
ORG:         Companies, agencies, institutions, etc.
GPE:         Countries, cities, states.
LOC:         Non-GPE locations, mountain ranges, bodies of water.
PRODUCT:     Objects, vehicles, foods, etc. (Not services.)
EVENT:       Named hurricanes, battles, wars, sports events, etc.
WORK_OF_ART: Titles of books, songs, etc.
LAW:         Named documents made into laws.
LANGUAGE:    Any named language.
DATE:        Absolute or relative dates or periods.
TIME:        Times smaller than a day.
PERCENT:     Percentage, including ”%“.
MONEY:       Monetary values, including unit.
QUANTITY:    Measurements, as of weight or distance.
ORDINAL:     “first”, “second”, etc.
CARDINAL:    Numerals that do not fall under another type.
'''

exclude = ['GPE','LOC','LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL']
# 
def find_entity(input_sent):
    doc = nlp(input_sent)
    extract_features = []
    for X in doc.ents:
        if X.label_ not in exclude:
            extract_features.append(X.text)
    return extract_features

# Entities can be viewed as a whole, sharing the same consistency
# So we simply assign the maximum probability difference
# eg. San  Fransico with score [a,b] (a>b)
# Will be processed to [a,a]

def pass_loss_Entity(score,tokens):
    count = 0
    tokens = tokens[1:-1]
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
    return score,count