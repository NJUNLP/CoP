import nltk
from nltk.corpus import wordnet as wn

# print(nltk.pos_tag(nltk.word_tokenize("This Court has jurisdiction to consider the merits of the case.")))

def mapping_src(tag):
    consider_tags_prefix = ["NN", "VB", "RB", "JJ"]
    maps = ["n", "v", "r", "s"]
    # consider_tags_prefix = ["NN"]
    # maps = ["n"]
    for j in range(len(consider_tags_prefix)):
        if tag[:2] == consider_tags_prefix[j]:
            return maps[j]
    return None


def find_all_source_lemmas(tok_src_sent, tok_src_pos_tags):
    all_lemmas = set()
    for word, pos in zip(tok_src_sent, tok_src_pos_tags):
        tag = mapping_src(pos)
        if tag is not None:
            try:
                lemmas = wn.synset("{}.{}.01".format(word, tag)).lemma_names()
                all_lemmas.update(lemmas)
            except:
                pass
    return all_lemmas

def mapping(tag):
    consider_tags_prefix = ["NN", "VB", "RB", "JJ"]
    maps = [wn.NOUN, wn.VERB, wn.ADV, wn.ADJ]
    # consider_tags_prefix = ["NN"]
    # maps = [wn.NOUN]

    for j in range(len(consider_tags_prefix)):
        if tag[:2] == consider_tags_prefix[j]:
            return maps[j]
    return None

def convert_token_labels_to_raw_labels(tok_labels, tok_sent, raw_sent):
    atom = []
    tags = []
    pointer = 0

    labels = []
    for tok, tag in zip(tok_sent, tok_labels):
        tok = tok.replace("``","\"")
        atom.append(tok)
        tags.append(tag)
        if "".join(atom) == raw_sent[pointer].replace("``","\""):
            labels.append(tags[0])
            atom = []
            tags = []
            pointer += 1
    if len(labels) != len(raw_sent):
        print(tok_labels)
        print(tok_sent)
        print(raw_sent)
    assert len(labels) == len(raw_sent)

    return labels

def find_synonyms(tok_tgt_sent, tok_tgt_pos_tags, tok_src_sent, tok_src_pos_tags):
    all_src_lemma = find_all_source_lemmas(tok_src_sent, tok_src_pos_tags)
    is_synonyms = []
    all_synonyms = []
    for word, pos in zip(tok_tgt_sent, tok_tgt_pos_tags):
        tag = mapping(pos)
        if tag is None:
            is_synonyms.append(0)
            continue
        synonyms = wn.synsets(word, pos=tag)
        if len(synonyms) > 0:
            syn_lemmas = set([lemma for ss in synonyms for lemma in ss.lemma_names()])
            xx = syn_lemmas.intersection(all_src_lemma)
            all_synonyms.append(xx)
            if len(xx) > 0:
                is_synonyms.append(1)
            else:
                is_synonyms.append(0)
        else:
            is_synonyms.append(0)
    assert len(is_synonyms) == len(tok_tgt_sent)
    return is_synonyms, all_synonyms



def predict_with_sync_baseline(document,summary):
    source_ = nltk.pos_tag(nltk.word_tokenize(document))
    target_ = nltk.pos_tag(nltk.word_tokenize(summary))
    source_token = [i[0] for i in source_]
    source_pos = [i[1] for i in source_]

    target_token = [i[0] for i in target_]
    target_pos = [i[1] for i in target_]


    has_synonyms, synonyms = find_synonyms(target_token, target_pos, source_token, source_pos)
    pred_label = [1-ss for ss in has_synonyms]
    pred_syn_labels = convert_token_labels_to_raw_labels(pred_label, target_token, summary.strip().split())
    # print(pred_syn_labels)
    # print(len(summary.strip().split()))
    return pred_syn_labels