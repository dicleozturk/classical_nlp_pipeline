'''
Created on Jun 19, 2017

@author: dicle
'''


import os
import pandas as pd

import re
from dataset import io_utils
import language_tools.language_identification as langid

# 1- remove nan
# 2- remove empty
# 3- remove <5 words
# 4- remove non-tr
def filter_unwanted(textdf, textcol,
                    word_limit=5):
    
    df = textdf.copy()
    #print(df.columns)
    print(df.shape)
    df = df[df[textcol].notnull()]
    
    texts = df[textcol].tolist()
    texts = [str(t).strip() for t in texts]
    df[textcol] = texts
    
    not_empty = lambda x : len(x) > 0
    df = df.loc[df[textcol].apply(not_empty), :]
    
    is_long = lambda x : len(x.split()) > word_limit
    df = df.loc[df[textcol].apply(is_long), :]
    
    
    # takes too long!
    '''
    is_tr = lambda x : langid.is_in_lang(x, "tr")
    df = df.loc[df[textcol].apply(is_tr)]
    '''
    print(df.shape)
    return df




# replace urls with URL
# replace dates with DATE
# replace numbers with NUM
def replace_with_symbols(texts):
    
    URL = "*URL*"
    NUM = "*NUM*"
    DATE = "*DATE*"
    USER = "*USER*"
    
    urlp = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    texts = [re.sub(urlp, URL, text) for text in texts]
    
    datep = "(\d{1,4}[\.\-\/]\d{1,2}[\.\-\/]\d{1,4})"
    texts = [re.sub(datep, DATE, text) for text in texts]
    
    nump = "\d+[\.\/,]?\d+"
    texts = [re.sub(nump, NUM, text) for text in texts]
    
    texts = [re.sub("<@USER>|@<USER>", USER, text) for text in texts]
    texts = [re.sub("<URL>", URL, text) for text in texts]
    
    return texts
    
    
    
    
    