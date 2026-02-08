'''
Created on Oct 26, 2017

@author: dicle
'''

import sys
sys.path.append("..")

import time
import gensim
import numpy as np

def load_embeddings(model_path, binary=True):

    print("Loading the embeddings from ", model_path)
    t0 = time.time()
    #model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False, encoding="utf-8")  #, encoding='ISO-8859-1')
    model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(model_path, binary=binary, encoding="utf-8")   #'ISO-8859-1')
    t1 = time.time()
    print("Loading finished. Took ", str(t1-t0), "sec.")
    
    return model


'''
 v1 and v2 are vectors (word embeddings)
'''
def euclidean_distance(v1, v2):
    dist = np.linalg.norm(v1-v2)
    return round(dist, 4)
