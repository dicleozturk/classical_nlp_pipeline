'''
Created on May 31, 2017

@author: dicle
'''


import sys
sys.path.append("..")


import os

import pandas as pd

import keyword_extraction.comparison_topic_extraction as topic_extractors
import text_categorization.prototypes.feature_extraction.text_preprocessor as txtprep
import language_tools.polyglot_NER as NER

# topical_words = [[(doc_i-term_j, term_j-weight),], ]
# join multiwords with "-"
def process_topical_words(docs_topical_words, lang, stem=True):
    
    terms = []
    
    for doc_topical_words in docs_topical_words:
        
        doc_terms = []
        
        for term, _ in doc_topical_words:
                        
            term_ = term
                                    
            if stem and (" " not in term_):
                term_ = txtprep.stem_words([term_], lang)[0]
            
            term_ = "-".join(term_.split())
            doc_terms.append(term_)
        
        terms.append(doc_terms)
    
    return terms



def extract_tfidf_heaviests(texts, lang, termsep=" # "):
    
    N_per_set = 5
    
    # tfidf - words
    
    docs_tfidf_words = topic_extractors.score_keyphrases_by_tfidf(texts, lang, candidate_type="words", stem=True, top_n_phrases=15)
    docs_tfidf_words2 = process_topical_words(docs_tfidf_words, lang, stem=True)
    #docs_tfidf_words2 = [termsep.join(doc_terms) for doc_terms in docs_tfidf_words2]
    
    
    # textrank
    docs_textrank_words = [topic_extractors.score_keyphrases_by_textrank(text, lang, n_keywords=50)
                                                for text in texts]
    docs_textrank_words2 = process_topical_words(docs_textrank_words, lang, stem=True)
    print(docs_textrank_words2)
    # limit the number of terms in one extracted_keyword
    f = lambda x : len(x.split("-")) < 4
    docs_textrank_words2 = [list(filter(f, terms)) for terms in docs_textrank_words2]
    print(docs_textrank_words2)
    
    # named entities
    docs_named_entities = [NER.get_named_entities(text, lang) for text in texts]
    docs_named_entities2 = process_topical_words(docs_named_entities, lang, stem=False)
    
    
    # merge
    '''
    tfidf = [terms[:N_per_set] for terms in docs_tfidf_words2]
    tfidf = tfidf + [list(map(lambda x : x.lower(), terms)) for terms in tfidf]
    tfidf = list(set(tfidf))
    textrank = [terms[:N_per_set] for terms in docs_textrank_words2]
    textrank = textrank + [list(map(lambda x : x.lower(), terms)) for terms in textrank]
    textrank = list(set(textrank))
    ne = docs_named_entities2 + [list(map(lambda x : x.lower(), terms)) for terms in docs_named_entities2]
    ne = list(set(ne))
    '''
    
    
    
    return docs_tfidf_words2, docs_textrank_words2, docs_named_entities2


    
# kwargs!
def merge_term_sets(docs_tfidf_words2, docs_textrank_words2, docs_named_entities2, N_per_set=5):        

    docs_terms = []
    
    for t1, t2, t3 in zip(docs_tfidf_words2, docs_textrank_words2, docs_named_entities2):
    
        t1_ = t1[:N_per_set]
        t2_ = t2[:N_per_set]
        t3_ = [i.lower().replace('i̇' , "i") for i in t3]
        
        terms = []
        terms.extend(t3_)
        
        for i in t1_:
            if i not in terms:
                terms.append(i)
        
        for i in t2_:
            if i not in terms:
                terms.append(i)
        docs_terms.append(terms)
    
    return docs_terms
    
# for a single doc, both are lists.
def _measure_tag_keyword_overlap(tags, keywords):    
    
    tags_ = [t.lower().replace('i̇' , "i") for t in tags]
    tags_.extend(tags)
    tags_ = list(set(tags_))
    tags_ = ["-".join(t.split()) for t in tags_]
    
    keywords_ = [t.lower().replace('i̇' , "i") for t in keywords]
    keywords_.extend(keywords)
    keywords_ = list(set(keywords_))
    
    intersection = [i for i in keywords_ if i in tags_]
    overlap_val = 0.0
    if len(tags) > 0:
        overlap_val = len(intersection) / len(tags)
    
    return intersection, overlap_val

def get_docs_tags_overlap(docs_tags, docs_keywords):
    
    docs_intersections = []
    docs_overlap_vals = []
    
    for tags, keywords in zip(docs_tags, docs_keywords):
        
        intersection, overlap_val = _measure_tag_keyword_overlap(tags, keywords)
        docs_intersections.append(intersection)
        docs_overlap_vals.append(overlap_val)
    
    return docs_intersections, docs_overlap_vals

        print()
        
        
        
    
    