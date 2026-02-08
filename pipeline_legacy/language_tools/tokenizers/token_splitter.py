'''
Created on Nov 2, 2018

@author: dicle
'''


import sys
sys.path.append("..")


import re
import string
import nltk


def is_punctuation(word):
    punkts = string.punctuation
    tested_chars = [i for i in word if i in punkts]
    return len(word) == len(tested_chars)

def eliminate_empty_strings(wordlist):
    l = [w.strip() for w in wordlist]
    l = [w for w in l if len(w) > 0]
    return l


def ends_with_2dots(txt):
    
    c = re.compile("\.{2,}$")
    search = c.search(txt)
    if search and len(txt) > len(search.group()):
        return txt[:-len(search.group())], search.group()
    else:
        return None

def text_to_words(text, lang="turkish", remove_punkt=False):
     
    language = "english"
    if lang in ["en", "eng", "english"]:
        language = "english"
    elif lang in ["tr", "turkish"]:
        language = "turkish"
        
    
    tokens = nltk.tokenize.word_tokenize(text, language=language)
        
    tokens = eliminate_empty_strings(tokens)

    tokens_ = []
    # to separate ".." from the word - word_tokenize() doesn't handle it; it can do only 1 or 3+ dots.
    for token in tokens:
        dots = ends_with_2dots(token)
        if dots:
            word, suffix = dots
            tokens_.append(word)
            tokens_.append(suffix)
        else:
            tokens_.append(token)

          
    if remove_punkt:
        tokens_ = [token for token in tokens_ if not is_punctuation(token)]

    return tokens_

    print(text_to_words("Hava, nihayet güzel.. ne güzel...", remove_punkt=False))
    
    
    
