'''
Created on Jun 7, 2017

@author: dicle
'''


import sys
sys.path.append("..")

from gensim import corpora


from time import time 

import os
import gensim

import string
import nltk
import pandas as pd
import numpy as np


from keyword_extraction import candidate_extraction
from dataset import corpus_io, io_utils
from language_tools import stopword_lists
import text_categorization.prototypes.feature_extraction.text_preprocessor as txtprep

import language_tools.polyglot_NER as NER

import pickle

txtfoldername = "docs"
titfoldername = "titles"

termsep = " # "


def downcase(words):
    words_ = [w.lower().replace('i̇' , "i")  for w in words]
    return words_

def _prep_texts(texts, lang, stem=False):
    
    
    allwords = [nltk.tokenize.wordpunct_tokenize(text) for text in texts]
    allwords = [downcase(words) for words in allwords]
    if stem:
        allwords = [txtprep.stem_words(words, lang) for words in allwords] 
    stopwords = stopword_lists.get_stopwords(lang=lang)  
    allwords = [[word for word in words if word not in stopwords] for words in allwords] 
    allwords = [[word for word in words if word not in string.punctuation] for words in allwords]
    return allwords



def prep_texts(texts, lang, preprocessor):
    
    allwords = [preprocessor.tokenize(text) for text in texts]
    
    return allwords



# takes a list of docs (each doc a string)
# returns [[phrase(i,j)]] where phrase(i,j) is the jth top phrase (sorted by tf*idf weight) in doc(i) (each row is for a doc)
# lda, lsi, nmf can be applied to model the topics
def train_tfidf_model(texts, lang, candidate_type='words', stemming=True, top_n_phrases=10,
                      dumpfolder=""):
    
    # extract candidates from each text in texts, either chunks or words
    '''
    if candidate_type == 'chunks':
        boc_texts = [candidate_extraction.extract_candidate_chunks(text, lang) for text in texts]
    elif candidate_type == 'words':
        boc_texts = [candidate_extraction.extract_candidate_words(text, lang) for text in texts]
        # stem?
        # strip punctuation?
        if stem:
            boc_texts = [txtprep.stem_words(boc_text, lang) for boc_text in boc_texts]
    ''' 
    
    preprocessor = txtprep.Preprocessor(lang=lang, stopword=True, more_stopwords=None,
                 spellcheck=False,
                 stemming=stemming,
                 remove_numbers=False,
                 deasciify=False,
                 remove_punkt=True,
                 lowercase=True)
    #allwords = prep_texts(texts, lang, preprocessor)
    allwords = _prep_texts(texts, lang, stemming)

        
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(allwords)
    corpus = [dictionary.doc2bow(doc_words) for doc_words in allwords]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    #corpus_tfidf = tfidf[corpus]
    
    # sort each row (doc vector) by the tf*idf values (the cells contain (word_id, tf*idf_weight))
    #sorted_matrix = [sorted(docvect, key=lambda x : x[1], reverse=True)[:top_n_phrases] for docvect in corpus_tfidf]
    # get phrases by ids
    #doc_top_phrases = [[(dictionary[pid], round(weight, 2)) for (pid, weight) in docvect] for docvect in sorted_matrix]


    if dumpfolder:
        dictionary.save(os.path.join(dumpfolder, "dict"))
        tfidf.save(os.path.join(dumpfolder, "tfidf_model"))
        pickle.dump(preprocessor, open(os.path.join(dumpfolder, "preprocessor.b"), "wb"))
    
    #return doc_top_phrases, tfidf, corpus_tfidf



# takes a list of docs (each doc a string)
# returns [[phrase(i,j)]] where phrase(i,j) is the jth top phrase (sorted by tf*idf weight) in doc(i) (each row is for a doc)
# lda, lsi, nmf can be applied to model the topics
def train_tfidf_model2(texts, lang, candidate_type='words', stemming=True, top_n_phrases=10,
                      dumpfolder=""):
    
    # extract candidates from each text in texts, either chunks or words
    '''
    if candidate_type == 'chunks':
        boc_texts = [candidate_extraction.extract_candidate_chunks(text, lang) for text in texts]
    elif candidate_type == 'words':
        boc_texts = [candidate_extraction.extract_candidate_words(text, lang) for text in texts]
        # stem?
        # strip punctuation?
        if stem:
            boc_texts = [txtprep.stem_words(boc_text, lang) for boc_text in boc_texts]
    ''' 
    
    preprocessor = txtprep.Preprocessor(lang=lang, stopword=True, 
                                        more_stopwords=["nan", "mailto", "from", "to", "cc", "com",
                                                        "merhaba", "sayın", "sayin"],
                 spellcheck=False,
                 stemming=stemming,
                 remove_numbers=True,
                 deasciify=True,
                 remove_punkt=True,
                 lowercase=True)
    allwords = prep_texts(texts, lang, preprocessor)
    #allwords = _prep_texts(texts, lang, stemming)

        
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(allwords)
    corpus = [dictionary.doc2bow(doc_words) for doc_words in allwords]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    # sort each row (doc vector) by the tf*idf values (the cells contain (word_id, tf*idf_weight))
    sorted_matrix = [sorted(docvect, key=lambda x : x[1], reverse=True)[:top_n_phrases] for docvect in corpus_tfidf]
    # get phrases by ids
    doc_top_phrases = [[(dictionary[pid], round(weight, 2)) for (pid, weight) in docvect] for docvect in sorted_matrix]


    if dumpfolder:
        dictionary.save(os.path.join(dumpfolder, "dict"))
        tfidf.save(os.path.join(dumpfolder, "tfidf_model"))
        pickle.dump(preprocessor, open(os.path.join(dumpfolder, "preprocessor.b"), "wb"))
    
    return doc_top_phrases, tfidf, corpus_tfidf
    

def find_keywords(newtext, lang, stem, dumpfolder, top_n_phrases=10):

    dict_ = corpora.Dictionary.load(os.path.join(dumpfolder, "dict"))
    tfidf_model = gensim.models.TfidfModel.load(os.path.join(dumpfolder, "tfidf_model"))
    #prep = pickle.load(open(os.path.join(dumpfolder, "preprocessor.b"), "rb"))
    
    #newwords = prep_texts([newtext], lang, prep)[0]
    newwords = _prep_texts([newtext], lang, stem)[0]
    '''
    newwords = candidate_extraction.extract_candidate_words(newtext, lang)
    if stem:
        newwords = txtprep.stem_words(newwords, lang)
    '''
    print(newwords)
    
    newbow = dict_.doc2bow(newwords)
    newvec = tfidf_model[newbow]

    N = min(top_n_phrases, len(newvec))
    top_keywords = [(str(dict_[tid]), w) for tid,w in sorted(newvec, key=lambda x : x[1], reverse=True)][:N]

    return newbow, newvec, top_keywords


def train_with_corpus(texts, titles, lang, stemming, top_n_phrases, dumpfolder):
    
    
    txtfpath = io_utils.ensure_dir(os.path.join(dumpfolder, txtfoldername))
    titfpath = io_utils.ensure_dir(os.path.join(dumpfolder, titfoldername))
    
    # train with texts
    train_tfidf_model(texts=texts, lang=lang, candidate_type="words", stemming=stemming, top_n_phrases=top_n_phrases, 
                      dumpfolder=txtfpath)


    # train with titles
    train_tfidf_model(texts=titles, lang=lang, candidate_type="words", stemming=stemming, top_n_phrases=top_n_phrases, 
                      dumpfolder=titfpath)
    
    


# extract keywords from 1) text 2) title using the saved tfidf model and extract named entities using polyglot
def extract_all_keywords(text, title, lang, stem, dumpfolder, top_n_phrases):
    
    txtmodelpath = os.path.join(dumpfolder, txtfoldername)
    titlemodelpath = os.path.join(dumpfolder, titfoldername)
    
    _,_,txt_keywords = find_keywords(text, lang, stem, txtmodelpath, top_n_phrases)
    print(txt_keywords)
    txt_keywords = [k.strip() for k,_ in txt_keywords]
    
    _,_,title_keywords = find_keywords(title, lang, stem, titlemodelpath, top_n_phrases=2)
    title_keywords = [k.strip() for k,_ in title_keywords]
    
    named_entities = NER.get_named_entities(text, lang)
    is_capital = lambda c : c[0] in string.ascii_uppercase + "İÜÖÇŞ"
    named_entities = [(e, tag) for e,tag in named_entities if is_capital(e)]
    #ne_dict = dict([(e.strip().lower().replace('i̇' , "i"), tag) for e,tag in named_entities])
    ne_dict = dict(named_entities)
    named_entities = [k.strip() for k,_ in named_entities]
    

    final_keywords = merge_term_sets_single_doc(txt_keywords, title_keywords, named_entities, N_per_set=5)

    final_keywords_tagged = []
    for keyword in final_keywords:
        if keyword in ne_dict.keys():
            final_keywords_tagged.append((keyword, ne_dict[keyword]))
        else:
            final_keywords_tagged.append((keyword, "TAG"))

    return txt_keywords, title_keywords, named_entities, final_keywords_tagged

    




# kwargs!
def merge_term_sets_single_doc(txt_terms, title_terms, nameds, N_per_set=5):        

    terms = []
    N1 = int(min(len(txt_terms), N_per_set))
    print(txt_terms)
    t1_ = txt_terms[:N1]
    
    N2 = int(min(len(title_terms), N_per_set))
    t2_ = title_terms[:N2]
    
    set1 = []
    set1.extend(t1_)
    for i in t2_:
        if i not in set1:
            set1.append(i)
    
    
    set2 = []
    set2.extend(nameds)
    t3_ = [i.lower().replace('i̇' , "i") for i in nameds]
    
    for i in set1:
        if i not in t3_:
            set2.append(i)
    
       
    return set2

# kwargs!
def merge_term_sets_all_docs(docs_tfidf_words, titles_tfidf_words, docs_named_entities, N_per_set=5):        

    docs_terms = []
    
    for t1, t2, t3 in zip(docs_tfidf_words, titles_tfidf_words, docs_named_entities):
        
        docs_terms.append(merge_term_sets_single_doc(t1, t2, t3, N_per_set))
    
    return docs_terms



# for a single doc, both are lists.
def _measure_tag_keyword_overlap(tags, keywords):    
    
    tags_ = [t.lower().replace('i̇' , "i").strip() for t in tags]
    #tags_.extend(tags)
    tags_ = list(set(tags_))
    #tags_ = ["-".join(t.split()) for t in tags_]
    
    keywords_ = [t.lower().replace('i̇' , "i").strip() for t in keywords]
    #keywords_.extend(keywords)
    keywords_ = list(set(keywords_))
    
    intersection = [i for i in keywords_ if i in tags_]
    overlap_val = 0.0
    if len(tags) > 0:
        overlap_val = len(intersection) / len(tags)
    
    return intersection, overlap_val

def get_docs_tags_overlap(docs_tags, docs_keywords):
    
    docs_intersections = []
    docs_overlap_vals = []
    
    for tags, keywords_ in zip(docs_tags, docs_keywords):
        keywords = [w for w,_ in keywords_]
        print(tags, " -- ", keywords)
        
        
        intersection, overlap_val = _measure_tag_keyword_overlap(tags, keywords)
        print("overlap: ", overlap_val, " ---- ", intersection)
        docs_intersections.append(intersection)
        docs_overlap_vals.append(overlap_val)
    
    return docs_intersections, docs_overlap_vals




def test_corpus(texts, titles, tags, lang, stem, dumpfolder, top_n_phrases,
                outpath):
    
    txt_keys = []
    title_keys = []
    named_entities = []
    final_keywords = []
    
    for text, title in zip(texts, titles):
        print(title)
        t1 = time()
        txtk, titlek, nes, finalk_ = extract_all_keywords(text, title, lang, stem, dumpfolder, top_n_phrases)
        t2 = time()
        print("   ", t2-t1," sec.")
        print(txtk, titlek, nes)
        txt_keys.append(txtk)
        title_keys.append(titlek)
        named_entities.append(nes)
        final_keywords.append(finalk_)
    
    #final_keywords = merge_term_sets_all_docs(txt_keys, title_keys, named_entities, N_per_set=5)
    intersection, overlap_vals = get_docs_tags_overlap(tags, final_keywords)
    
    
    '''
    output = np.array([texts, titles, tags, 
                       txt_keys, title_keys, named_entities,
                       final_keywords,
                       intersection,
                       overlap_vals], dtype=object).T
    df = pd.DataFrame(output, columns=["texts", "titles", "tags", 
                                       "TXT_KEYS", "TITLE_KEYS", "NEs",
                                       "FINAL_KEYWORDS",
                                       "intersection",
                                       "overlap_value"])
    '''
    
    print(len(txt_keys), len(title_keys), len(final_keywords))
    fk = [[term for term,_ in ks] for ks in final_keywords]
    print(len(fk))
    print(txt_keys[:3], title_keys[:3])
    print(final_keywords)
    print(fk)
    
    df = pd.DataFrame()
    df["texts"] = texts
    df["titles"] = titles
    df["tags"] = tags
    df["TXT_KEYS"] = txt_keys
    df["TITLE_KEYS"] = title_keys
    df["NEs"] = named_entities
    df["FINAL_KEYWORDS"] = fk
    df["intersection"] = intersection
    df["overlap_value"] = overlap_vals
    
    df.to_csv(outpath, sep="\t")
    
    #print(final_keywords)



