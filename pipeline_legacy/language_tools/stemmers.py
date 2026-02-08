'''
Created on Nov 7, 2018

@author: dicle
'''

import sys
sys.path.append("..")

from language_tools import tr_stemmer, en_stemmer, ar_stemmer, fr_stemmer

def stem_words(words, lang, tr_alt_stemming=False):
    roots = []
    if lang in ["tr", "turkish"]:
        roots = tr_stemmer.stem_words(words, alternative_stemming=tr_alt_stemming)
    
    if lang in ["en", "english"]:
        roots = [en_stemmer.stem1(word) for word in words]
    
    if lang in ["ar", "arabic", "arab"]:
        roots = [ar_stemmer.stem(word) for word in words]
    
    if lang in ["fr", "french"]:
        roots = [fr_stemmer.stem1(word) for word in words]
    return roots

