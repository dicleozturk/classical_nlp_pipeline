'''
Created on Jul 9, 2018

@author: dicle
'''

import sys
from embeddings import embd_utils
sys.path.append("..")


import numpy as np
import time
import gensim


from language_tools import tr_tokenizer


class EmbeddingReader:
    
    embds_path = ""
    embeddings = None
    embd_length = None
    UNK_vect = None
    tokenizer = None
    
    def __init__(self, embds_path, tokenizer=tr_tokenizer.get_tokenized_sentences, 
                 binary=True, encoding="utf-8"):
        self.embds_path = embds_path
        self.embeddings = None
        self.embd_length = None
        self.UNK_vect = None #np.random.uniform(-0.25, 0.25, self.embd_length)
        
        self.binary = binary
        self.encoding = encoding
        
        self.tokenizer = tokenizer
    
    def load_embeddings(self):

        print("Loading the embeddings from ", self.embds_path)
        t0 = time.time()
        #model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False, encoding="utf-8")  #, encoding='ISO-8859-1')
        #self.embeddings = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(self.embds_path, binary=binary, encoding="utf-8")   #'ISO-8859-1')
        self.embeddings = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(self.embds_path, 
                                                                                       binary=self.binary, encoding=self.encoding)   #'ISO-8859-1')
        t1 = time.time()
        print("Loading finished. Took ", str(t1-t0), "sec.")
        self.embd_length = self.embeddings.vector_size
        #self.UNK_vect = np.random.uniform(-0.25, 0.25, self.embd_length)
        
    
    def get_word_vector(self, word):
        
        try:
            return self.embeddings[word]
        except KeyError:
            #vect = np.random.uniform(-0.25, 0.25, self.embd_length)
            vect = np.random.uniform(-0.25, 0.25, self.embd_length) #self.UNK_vect
            return vect
    
    
    
    # avg    - from tokens
    def get_doc_vector_from_words(self, doc_words):
        
        if not self.embeddings:
            self.load_embeddings()
            
            
        doc_vect = [0]*self.embd_length
        for w in doc_words:
            word_vect = self.get_word_vector(w)
            #print(type(word_vect))
           
            doc_vect = doc_vect + word_vect
    
        doc_vect = doc_vect / len(doc_words)
        return doc_vect
    
    
    def get_doc_vector(self, text):
        
        if not self.embeddings:
            self.load_embeddings()
        
        tokenized_sentences = self.tokenizer(text)
        tokens = [token for sentence in tokenized_sentences for token in sentence]
        #print(tokens)
        
        doc_vect = self.get_doc_vector_from_words(tokens)
        return doc_vect
    
    
    # to load embeddings once for a dataset
    def get_doc_vectors_of_collection(self, texts):
        
        all_vects = []
        for text in texts:
            v = self.get_doc_vector(text)
            all_vects.append(v)
        
        return all_vects



embeddings_param = {"name" : "tweets_size-50",
                    "path" : "<PATH>",
                    "binary" : False,
                    "tokenizer" : tr_tokenizer.get_tokenized_sentences
                    }


def f(t, tokenizer):
    return tokenizer(t)

    text1 = "Bakın biz non-newtonyen orange polimer gibiyiz bize yumuşakça nazikçe gelene her daim misafirperver davranız bize"
    text2 = "Erdoğan'ın yazlık sarayı için inşaat alanı 20 hektardan 65 hektara çıkarılıyor."
    
    
    #print(f(text1, embeddings_param["tokenizer"]))
    
    '''
    embeddings_reader = EmbeddingReader(embds_path=embeddings_param["path"],
                                        tokenizer=embeddings_param["tokenizer"],
                                        binary=embeddings_param["binary"])
    
    
    
    dvect1 = embeddings_reader.get_doc_vector(text1)
    '''
    
    
    embeddings_param_en = {"name" : "en_embd",
                    "path" : "<PATH>",                 
                    "binary" : True,
                    "tokenizer" : tr_tokenizer.get_tokenized_sentences
                }
    en_embeddings_reader = EmbeddingReader(embds_path=embeddings_param_en["path"],
                                           tokenizer=embeddings_param_en["tokenizer"],
                                           binary=embeddings_param_en["binary"])
    
    doc1 = "Hello, this is for english embeddings."
    doc2 = "And this is for turkish embeddings."
    doc3 = "I'm listening to music"
    v1 = en_embeddings_reader.get_doc_vector(doc1)
    v2 = en_embeddings_reader.get_doc_vector(doc2)
    v3 = en_embeddings_reader.get_doc_vector(doc3)
    d12 = embd_utils.euclidean_distance(v1, v2)
    d13 = embd_utils.euclidean_distance(v1, v3)
    d23 = embd_utils.euclidean_distance(v3, v2)
    print(v1)
    print(d12, d13, d23)
    print()
    
    
    
    