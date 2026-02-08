'''
Created on Jul 4, 2018

@author: dicle
'''

import sys
sys.path.append("..")

import os
import re
import glob
from sklearn.externals import joblib

import gensim
import logging
import time

import tr_splitter_tokenizer as tr_tokenizer

def read_lines(path):
    
    f = open(path, "r")
    lines = f.readlines()
    return lines


# returns the names of the files and dirs in the given directory *path*
def getfilenames_of_dir(path, removeextension=True):
    files = os.listdir(path)
    filenames = []
    for fileitem in files:
        if os.path.isfile(path + os.sep + fileitem):
            if removeextension:
                filename = fileitem.split(".")[0]  # remove extension if any
            else:
                filename = fileitem
            filenames.append(filename)
        
    return filenames

def getfoldernames_of_dir(path):
    files = os.listdir(path)
    foldernames = []
    for fileitem in files:
        if os.path.isdir(path + os.sep + fileitem):
            foldernames.append(fileitem)
    return foldernames


# ensures if the directory given on *f* exists. if not creates it.
def ensure_dir(f):
    # d = os.path.dirname(f)
    if not os.path.exists(f):
        os.makedirs(f)
    return f 



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


def tokenize_texts(texts):
    
    all_sentences = []
    for text in texts:
        
        tokenized_sentences = tr_tokenizer.get_tokenized_sentences(text)
        all_sentences.extend(tokenized_sentences)
    
    return all_sentences


def get_pickled_sentences(smainfolder):
    
 
    sentences = []
    for p in glob.glob(os.path.join(smainfolder, "*.b")):
        
        file_sentences = joblib.load(p)
        #sentences.append(file_sentences)
        sentences.extend(file_sentences)

    return sentences




def paramdict_to_str(paramdict):

   
    f = lambda x : str(x[0])+"-"+str(x[1])
    return "_".join(list(map(f, paramdict.items())))



'''
params: 
_size=50, _min_count=3, _window=5,
                        _alpha=0.025,
                        _max_vocab_size=1500000
'''
def generate_embeddings2(sentences, outfolder, modelname,
                        params=dict(size=50, min_count=3, window=5,
                        alpha=0.025, max_vocab_size=1500000)):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) #"-".join(str(datetime.now()).split())
    
    paramstr = paramdict_to_str(params)
    
    outpath = os.path.join(outfolder, paramstr+"_"+modelname+"_"+timestamp+".txt")
    
    
    print("\nGenerating word vectors..")
    
    _size = params["size"]
    _min_count = params["min_count"]
    _window = params["window"]
    _max_vocab_size = params["max_vocab_size"]
    _alpha = params["alpha"]
    _trainer = params["trainer"]
    _neg_samp = params["neg_sampling"]
    
    t0 = time.time()
    model = gensim.models.Word2Vec(sentences, 
                                   size=_size, min_count=_min_count, window=_window,
                                   max_vocab_size=_max_vocab_size, alpha=_alpha,
                                   sg=int(_trainer!="cbow"), hs=0, negative=_neg_samp)
    
    
    t1 = time.time()
    print("Generating vectors finished. Took ", str(t1-t0), "sec.")
    
    
    print("\nSaving the vectors in ", outpath)
    model.wv.save_word2vec_format(outpath, binary=False)
    t2 = time.time()
    print("Recording finished. Took ", str(t2-t1), "sec.")
    #model.wv.save(outpath)

    return outpath



def load_embeddings(model_path, efficient=True, binary=False):

    print("Loading the embeddings from ", model_path)
    t0 = time.time()
    #model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False, encoding="utf-8")  #, encoding='ISO-8859-1')
    model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(model_path, binary=binary, encoding="utf-8")   #'ISO-8859-1')
    t1 = time.time()
    print("Loading finished. Took ", str(t1-t0), "sec.")
    #model = gensim.models.Word2Vec.load_word2vec_format(mpath)
    #words = ["şiir", "gazete", "Ankara", "Diyarbakır", "ve"]
    
    '''
    if efficient:
        model = model.wv
    '''
    return model 




