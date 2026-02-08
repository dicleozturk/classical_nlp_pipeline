'''
Created on Dec 11, 2017

@author: dicle
'''

import sys
sys.path.append("..")



import nltk.stem as stemmer


def stem1(word):
    
    snowball_stemmer = stemmer.SnowballStemmer("french")
    return snowball_stemmer.stem(word)




    print(stem1("vacances"))
    
    