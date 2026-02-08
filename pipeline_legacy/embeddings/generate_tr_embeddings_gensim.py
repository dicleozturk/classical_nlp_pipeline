'''
Created on Aug 18, 2017

@author: dicle
'''
# import modules & set up logging
import gensim, logging

import time
from datetime import datetime

import os
import pandas as pd
import joblib
import glob



from language_tools import tr_tokenizer




def file_to_sentences(filepath):
    
    content = open(filepath, "r").read()
    
    '''
    import nltk.data
    tr_tokenizer = nltk.data.load('tokenizers/punkt/turkish.pickle')
    sentences = tr_tokenizer.tokenize(content)
    sentences = [s.strip() for s in sentences]
    return sentences
    '''
    return tr_tokenizer.get_tokenized_sentences(content)





def f(args, args2, *kwargs):
    
    for key in kwargs:
        print(key)



def generate_embeddings(sentences, outfolder, modelname,
                        _size=50, _min_count=3, _window=5,
                        _alpha=0.025,
                        _max_vocab_size=1500000):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) #"-".join(str(datetime.now()).split())
    outpath = os.path.join(outfolder, "s-"+str(_size)+"_min-count-"+str(_min_count)+"_"+modelname+"_"+timestamp+".txt")
    
    
    print("\nGenerating word vectors..")
    t0 = time.time()
    model = gensim.models.Word2Vec(sentences, 
                                   vector_size=_size, min_count=_min_count, window=_window,
                                   max_vocab_size=_max_vocab_size, alpha=_alpha,
                                   sg=1, hs=0, negative=10)
    
    
    t1 = time.time()
    print("Generating vectors finished. Took ", str(t1-t0), "sec.")
    
    
    print("\nSaving the vectors in ", outpath)
    model.wv.save_word2vec_format(outpath, binary=False)
    t2 = time.time()
    print("Recording finished. Took ", str(t2-t1), "sec.")
    #model.wv.save(outpath)

    return outpath


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
    
    t0 = time.time()
    model = gensim.models.Word2Vec(sentences, 
                                   size=_size, min_count=_min_count, window=_window,
                                   max_vocab_size=_max_vocab_size, alpha=_alpha,
                                   sg=1, hs=0, negative=5)
    
    
    t1 = time.time()
    print("Generating vectors finished. Took ", str(t1-t0), "sec.")
    
    
    print("\nSaving the vectors in ", outpath)
    model.wv.save_word2vec_format(outpath, binary=False)
    t2 = time.time()
    print("Recording finished. Took ", str(t2-t1), "sec.")
    #model.wv.save(outpath)

    return outpath


def csv_to_sentences(df, textcol, picklepath=None):
    
    texts = df[textcol].tolist()
    
    allsentences = []
    for text in texts:
        tsentences = tr_tokenizer.get_tokenized_sentences(text)
        allsentences.extend(tsentences)

    if picklepath:
        joblib.dump(allsentences, picklepath)

    return allsentences







def load_embeddings(model_path, efficient=True, binary=True):

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
    return model   # for efficient memory use: 
                        # If you’re finished training a model (=no more updates, only querying), then switch to the gensim.models.KeyedVectors instance in wv
                            # >>> word_vectors = model.wv
                            # >>> del model





def get_vectors_of(words, vectors_path, binary=False):    
    import numpy as np
    vects = load_embeddings(vectors_path, binary=binary)
    selected_vects = []
    
    for w in words:
        try:
            v = vects[w]
        except KeyError:
            v = np.random.uniform(-0.25, 0.25, vects.vector_size)
        
        selected_vects.append(v)
    
    

'''

def load_embeddings(model_path):

    print("Loading the embeddings from ", model_path)
    t0 = time.time()
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False, encoding="utf-8")  #, encoding='ISO-8859-1')
    t1 = time.time()
    print("Loading finished. Took ", str(t1-t0), "sec.")
    #model = gensim.models.Word2Vec.load_word2vec_format(mpath)
    words = ["şiir", "gazete", "Ankara", "Diyarbakır", "ve"]
    
    vocab = model.vocab
    print(type(vocab))
    print(list(vocab.items())[0])
    for w in words:
        print()
        if w not in vocab:
            print(w, " not in vocab")
        else:
            print(w, " --- most similar words --- : ", model.most_similar([w]))

'''   

# prints the similar words according to the model for each word given in words.
def get_similar_words(words, model):

    vocab = model.vocab
    for w in words:
        print()
        if w not in vocab:
            print(w, " not in vocab")
        else:
            print(w, " --- most similar words --- : ", model.most_similar([w]))
    
    



def tokenize_dataset_and_save():


    #incsvfolder = "<PATH>"
    rel_incsvfolder = "../data/raw_data/prep"
    incsvfolder = os.path.abspath(rel_incsvfolder)
    fnames = ["prep_tr_news_200MB.csv", "prep_ntv.csv"]
    
    #fnames = io_utils.getfilenames_of_dir(incsvfolder, removeextension=False)
    textcol = "description"

    
    #sentences_folder = "<PATH>"
    rel_sentences_folder = "../data/raw_data/sentences2"
    sentences_folder = os.path.abspath(rel_sentences_folder)
    
    print("\nSplit texts into sentences and tokens; record in ", sentences_folder)
    t0 = time.time()
    
    for fname in fnames:
        
        t00 = time.time()
        df = pd.read_csv(os.path.join(incsvfolder, fname), sep="\t")
        print(fname, df.shape)
        spicklepath = os.path.join(sentences_folder, fname+"_sentences.b")
        csv_to_sentences(df, textcol, spicklepath)
        t01 = time.time()
        print("    ", round(t01-t00, 2), " sec. ", fname)
        
    t1 = time.time()
    print("Tokenizing took ", t1-t0," sec.\n")
    
    

def get_pickled_sentences(smainfolder):
    
 
    sentences = []
    for p in glob.glob(os.path.join(smainfolder, "*.b")):
        
        file_sentences = joblib.load(p)
        #sentences.append(file_sentences)
        sentences.extend(file_sentences)

    return sentences
 



def tokenize_texts(texts):
    
    all_sentences = []
    for text in texts:
        
        tokenized_sentences = tr_tokenizer.get_tokenized_sentences(text)
        all_sentences.extend(tokenized_sentences)
    
    return all_sentences




    
