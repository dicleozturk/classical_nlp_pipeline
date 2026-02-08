'''
Created on Sep 6, 2017

@author: dicle
'''

import sys
sys.path.append("..")

import nltk


def pos_tag_sentences(text, lang):

    postagged_sentences = []

    if lang in ["en", "eng", "english"]:
        postagged_sentences = nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                    for sent in nltk.sent_tokenize(text))
        
        
    elif lang in ["tr", "turkish"]:
        import language_tools.tr_postag_syntaxnet as tr_syntaxnet
        #from language_tools import tokenization
        #sentences = tokenization.tr_sentence_tokenizer(text)
        #print(len(sentences))
        #postagged_sentences = tr_syntaxnet.postag_sentences(sentences)
        postagged_sentences = tr_syntaxnet.postag_text(text)

    return postagged_sentences







    print()
    
    
    print(pos_tag_sentences(text="i want this doing", lang="en"))
    
    