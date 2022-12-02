import nltk
from nltk.corpus import wordnet

def return_synonyms(input_word):
    synonyms = []

    for syn in wordnet.synsets(input_word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return list(set(synonyms))