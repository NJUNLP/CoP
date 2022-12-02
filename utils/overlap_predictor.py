
from utils.wordnet import return_synonyms
from utils.wn import predict_with_sync_baseline

def returnNone_baseline(document,summary):
    document = document
    prediction_label = []
    for i in summary.split(' '):
        prediction_label.append(0)
    return prediction_label

def synonyms_notin(document,summary):
    document = document
    prediction_label = []
    for i in summary.split(' '):
        s = return_synonyms(i.replace('.','').replace('\'s','').replace(",",''))
        s.append(i.replace('.','').replace('\'s','').replace(",",''))
        flag = 1
        for t in s:
            if t in document:
                prediction_label.append(0)
                flag = 0
                break
        if flag:
            prediction_label.append(1)

    return prediction_label

def NotInSource(document,summary):
    document = document
    prediction_label = []
    for i in summary.split(' '):
        if i.replace('.','').replace('\'s','').replace(",",'') not in document:
            prediction_label.append(1)
        else:
            prediction_label.append(0)
    return prediction_label