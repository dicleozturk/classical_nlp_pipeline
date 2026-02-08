'''
Created on Oct 12, 2016

@author: dicle
'''

import os

import nltk
import gensim
import networkx
from itertools import takewhile, tee

import pandas as pd

from time import time 

from rake_keyphrase_extractor import rake
import six
import operator
import random

from keyword_extraction import candidate_extraction
from dataset import corpus_io
from language_tools import stopword_lists
import text_categorization.prototypes.feature_extraction.text_preprocessor as txtprep
from tr_chunker.TR_chunker import chunker_main2




# takes a list of docs (each doc a string)
# returns [[phrase(i,j)]] where phrase(i,j) is the jth top phrase (sorted by tf*idf weight) in doc(i) (each row is for a doc)
# lda, lsi, nmf can be applied to model the topics
def score_keyphrases_by_tfidf(texts, lang, candidate_type='chunks', stem=True, top_n_phrases=10):
    
    # extract candidates from each text in texts, either chunks or words
    if candidate_type == 'chunks':
        boc_texts = [candidate_extraction.extract_candidate_chunks(text, lang) for text in texts]
    elif candidate_type == 'words':
        boc_texts = [candidate_extraction.extract_candidate_words(text, lang) for text in texts]
        # stem?
        # strip punctuation?
        if stem:
            boc_texts = [txtprep.stem_words(boc_text, lang) for boc_text in boc_texts]
    
    elif candidate_type == "all":
        unigrams = [candidate_extraction.extract_candidate_words(text, lang) for text in texts]
        if stem:
            unigrams = [txtprep.stem_words(word, lang) for word in unigrams]
        chunks = [candidate_extraction.extract_candidate_chunks(text, lang) for text in texts]
        
        boc_texts = [i+j for i,j in zip(unigrams, chunks)]
        #boc_texts = [random.shuffle(keys) for keys in boc_texts]
        
        print("xxx")
        print(len(unigrams), unigrams[:2])
        print(len(chunks), chunks[:2])
        
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    # sort each row (doc vector) by the tf*idf values (the cells contain (word_id, tf*idf_weight))
    sorted_matrix = [sorted(docvect, key=lambda x : x[1], reverse=True)[:top_n_phrases] for docvect in corpus_tfidf]
    # get phrases by ids
    doc_top_phrases = [[(dictionary[pid], round(weight, 2)) for (pid, weight) in docvect] for docvect in sorted_matrix]

    return doc_top_phrases





def score_keyphrases_by_textrank(text, lang, n_keywords=50):
    
    
    # tokenize for all words, and extract *candidate* words

    words = [word.lower().replace('i̇' , "i")
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    candidates = candidate_extraction.extract_candidate_words(text, lang)
 
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = round(avg_pagerank, 2)
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)
    
    return sorted(keyphrases.items(), key=lambda x: x[1], reverse=True)
    
    


# stopword and phrase pattern based keyphrase extraction
def score_keyphrases_by_rake(text, lang, top_n_phrases=10):
    
    sentenceList = rake.split_sentences(text)

    # generate candidate keywords
    #stoppath = "SmartStoplist.txt"
    #stopwordpattern = rake.build_stop_word_regex_path(stoppath)
    stopwords = stopword_lists.get_stopwords(lang)
    stopwordpattern = rake.build_stop_word_regex_list(stopwords)
    phraseList = rake.generate_candidate_keywords(sentenceList, stopwordpattern, max_words_length=3)
    
    # calculate individual word scores
    wordscores = rake.calculate_word_scores(phraseList)
    
    # generate candidate keyword scores
    keywordcandidates = rake.generate_candidate_keyword_scores(phraseList, wordscores)
    
    # sort candidates by score to determine top-scoring keywords
    sortedKeywords = sorted(six.iteritems(keywordcandidates), key=operator.itemgetter(1), reverse=True)
    
    # for example, you could just take the top third as the final keywords
    return sortedKeywords[0:top_n_phrases]


# doc_phrase_weight_list = [(word(i,j), weight)] for the jth word of doc_i
def print_top_phrases(doc_phrase_weight_list, print_weight=False):   
    
    for i,tuples in enumerate(doc_phrase_weight_list):
        print("Doc ",i," : ", end="")
        
        if print_weight:
            wlist = [(word, round(weight, 2)) for word, weight in tuples]
        else:
            wlist = [word for word,_ in tuples]
        print(wlist)
        


# doc_phrase_weight_list = [(word(i,j), weight)] for the jth word of doc_i
def get_top_phrases(doc_phrase_weight_list, print_weight=False):   
    
    doc_terms = []
    for i,tuples in enumerate(doc_phrase_weight_list):
        #print("Doc ",i," : ", end="")
        
        if print_weight:
            wlist = [(word, round(weight, 2)) for word, weight in tuples]
        else:
            wlist = [word for word,_ in tuples]
        doc_terms.append((i, str(wlist)))
    
    return doc_terms
        
        
    
def run_method(method, instances, lang):
    
    weighted_phrases = [method(text, lang) for text in instances]
    weighted_phrases = [(word, round(val, 2)) for word,val in weighted_phrases]
    return weighted_phrases


def extract_keywords_allmethods(instances, lang, col_prefix, topNterms=15):
     
    df = pd.DataFrame()
         
    print("tfidf scoring")
    '''
    Algorithm (traditional tf*idf weighting scheme):
    Gerard Salton and Christopher Buckley. 1988. Term-weighting approaches in automatic text retrieval. Information Processing and Management, 24(5):513–523.
     
    Implementation:
    gensim: https://radimrehurek.com/gensim/index.html   
    '''
    
    
    
    '''
    t00 = time()
    
    doc_topphrases_tfidf = score_keyphrases_by_tfidf(instances, lang, top_n_phrases=topNterms)
    
    df[col_prefix+"_terms_tfidf_chunk"] = [str(i) for i in doc_topphrases_tfidf]
    
    t0 = time()
    print("tfidf took ", t0-t00,"sec.")
    '''
 
    t0 = time()
    doc_topphrases_tfidf = score_keyphrases_by_tfidf(instances, lang, candidate_type="all", top_n_phrases=topNterms)
    
    df[col_prefix+"_terms_tfidf_word"] = [str(i) for i in doc_topphrases_tfidf]
       
    
    t1 = time()
    print("tfidf took ", t1-t0,"sec.")
    print(doc_topphrases_tfidf)
    print()
    
    
    
     
    print("textrank scoring (pagerank applied to the graph of phrases)")
    '''
    Algorithm:
    Rada Mihalcea and Paul Tarau, TextRank: Bringing Order into Texts, in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2004), Barcelona, Spain, July 2004.
     
    Implementation:
    http://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/  
    '''
    doc_topphrases_textrank = [score_keyphrases_by_textrank(text, lang, n_keywords=topNterms) for text in instances]
    print(doc_topphrases_textrank)
    df[col_prefix+"_terms_textrank"] = [str(i) for i in doc_topphrases_textrank]
    print()
     
     
    t2 = time()
    print("textrank took ", t2-t1, "sec.")
     
  
     
    print("scoring with rake based on stopword and phrase patterns") 
    '''
    Algorithm:
    Rose, S., Engel, D., Cramer, N., & Cowley, W. (2010). Automatic Keyword Extraction from Individual Documents. In M. W. Berry & J. Kogan (Eds.), Text Mining: Theory and Applications: John Wiley & Sons.
     
    Implementation:
    1) https://github.com/zelandiya/RAKE-tutorial
    2) https://www.airpair.com/nlp/keyword-extraction-tutorial
    @todo: see for a neat implementation http://sujitpal.blogspot.com.tr/2013/03/implementing-rake-algorithm-with-nltk.html
    '''
    doc_topphrases_rake = [score_keyphrases_by_rake(text, lang, top_n_phrases=topNterms) for text in instances]
    df[col_prefix+"_terms_rake"] = [str(i) for i in doc_topphrases_rake]
    t3 = time()
    print("rake took ", t3-t2, "sec.")
    print(doc_topphrases_rake)
    print()
    
    
    
    print("Only chunks")
    chunks = [chunker_main2.chunk_text(text) for text in instances]
    df[col_prefix+"_chunks"] = [str(i) for i in chunks]
    t4 = time()
    print("chunking took ", t4-t3, "sec.")
     
    return df





    '''
    instances = ["Merhaba, bugün ne güzel bir gün bu güzel koltukta!", "Sade kahve şekersiz bir kaşıkla nasıl içilir?", "Temiz su plastik şişede duran sakin suyu içmek çok susuzken çok acil!"]
    
    t_all = score_keyphrases_by_tfidf(instances, lang="tr", candidate_type="all")
    
    t_c = score_keyphrases_by_tfidf(instances, lang="tr", candidate_type="chunks")
    t_w = score_keyphrases_by_tfidf(instances, lang="tr", candidate_type="words")
    
    
    print(len(t_c), t_c)
    print()
    print(len(t_w), t_w)
    print()
    
    print(len(t_all), t_all)

    

    k_df = extract_keywords_allmethods(instances, lang="tr", col_prefix="k", topNterms=20)
    
    outfolder = "<PATH>"
    k_df.to_csv(os.path.join(outfolder, "sample_tfidf-all.csv"), sep="\t", index=False)
    '''
    
    main1()






  

    