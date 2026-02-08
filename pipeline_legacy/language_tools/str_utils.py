'''
Created on May 28, 2019

@author: dicle
'''


import sys
sys.path.append("..")


import string
import re




def ends_with(str_, pattern):
    
    c = re.compile(pattern+"$")
    if c.search(str_):
        return True
    else:
        return False

def is_punctuation(word):
    punkts = string.punctuation
    tested_chars = [i for i in word if i in punkts]
    return len(word) == len(tested_chars)




def eliminate_empty_strings(wordlist):
    l = [w.strip() for w in wordlist]
    l = [w for w in l if len(w) > 0]
    return l

def remove_surrounding_punctuation(word):
    p = "[,\.\!\;]"
    
    w = word
    if re.match("^"+p, w):
        w = w[1:]
    if re.search(p+"$", w):
        w = w[:-1]
    return w



