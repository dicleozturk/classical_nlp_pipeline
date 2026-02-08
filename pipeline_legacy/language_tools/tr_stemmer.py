'''
Created on Dec 21, 2016

@author: dicle
'''

import sys
sys.path.append("..")

import os
import re
import subprocess
from time import time

import nltk.tokenize as nltktokenizer
import snowballstemmer
#from polyglot.text import Word

from language_tools import TOOL_CONSTANTS

#import tr_morph_analyzer.scripts.disambiguate as tr_morph

import language_tools.tr_morph_analyzer.scripts.disambiguate as tr_morph

###### using method: https://github.com/coltekin/TRmorph   ####
# -- faster version in langauge_tools/tr_morph_analyzer; using scripts from https://github.com/coltekin/TRmorph.
def stem(word):
    
    return tr_morph.stem_word(word)


'''
 alternative_stemming returns the longest root; returns araştırma, not araştır.
'''
def stem_words(word_list, alternative_stemming=False):
    
    if alternative_stemming:
        return tr_morph.stem_word_list_alternatives(word_list)
    else:
        return tr_morph.stem_word_list(word_list)   


def stem_words_in_text(text, tokenize=False):
    words = []
    if tokenize:
        words = nltktokenizer.wordpunct_tokenize(text)
    else:
        words = text.split()
    return stem_words(words)
##################################################

'''
method: https://github.com/coltekin/TRmorph
--slower version
'''
def _stem(word):                                                            
    # return  subprocess.Popen("echo '" + word + "' | flookup Documents/tools/tr_morph/coltekin/TRmorph/stem.fst", shell=True, stdout=subprocess.PIPE).stdout.read().split()
    
    # problems with apostrophe
    apost_pattern = r"[\"'’´′ʼ]"
    w = re.sub(apost_pattern, "", word)
    
    '''
    items = subprocess.Popen("echo '" + w + "' | flookup " + TOOL_CONSTANTS.PATH_TO_STEM_FST, 
                             shell=True, stdout=subprocess.PIPE).stdout.read().split()
    '''
    proc = subprocess.Popen("echo '" + w + "' | flookup " + TOOL_CONSTANTS.PATH_TO_STEM_FST,
                             shell=True, stdout=subprocess.PIPE).stdout
    items = proc.read().split()
    proc.close()
    items = [str(i, "utf-8") for i in items]
    # print(items)
    root = items[-1]
    
    tag_pattern = r"\<(\w+:?\w*)+\>"  # "\<\w{1,4}\>"
    root = re.sub(tag_pattern, "", root)

    
    if root.endswith("?"):  # no root could be found, especially for NEs.
        return word
    else:
        return root




def stem2(word):
    
    
    stemmer = snowballstemmer.stemmer("turkish")
    stemmed = stemmer.stemWord(word)
    
    if stemmed == "fatur":
        stemmed = "fatura"
    elif stemmed == "hatt":
        stemmed = "hat"
    
    return stemmed


# hasim sak's morphological analyser
def stem3(word):

    ''' # doesn't work
    #command = "python2 " + TOOL_CONSTANTS.PATH_TO_SAK_PARSER_PY + " " + word
    command = "python2 <PATH> geliyorum" 
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
    items = proc.read().split()
    proc.close()
    return items
    # '''
    

def stem4(word):
    w = Word(word, language="tr")
    return w.morphemes[0]

    print(stem("yapıyorum"))

