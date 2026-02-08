'''
Created on Jul 4, 2018

@author: dicle
'''
import sys
sys.path.append("..")

import os

from sklearn.externals import joblib


from dataset import io_utils
from embeddings import prep_filter
import embeddings.generate_tr_embeddings_gensim3 as embd_prod



#########################




'''
 1- sentence tokenize text lists
 2- word tokenize sentences in texts lists


in: lists of texts
out: lists of sentences which are lists of words, as pickled.
'''



def _prep_tokenize_save_texts(textspath, picklepath):
                            
    
    
    texts = io_utils.read_lines(textspath)
    
    # 1- Replace symbols
    print("  replacing symbols")
    texts = prep_filter.replace_with_symbols(texts)
    
    # 2- Tokenize texts to sentences and words [[w,]]
    print("  tokenizing")
    sents = embd_prod.tokenize_texts(texts)
    
    if picklepath:
        joblib.dump(sents, picklepath)
        
  
        
    
    
def prep_tokenize_save_dataset(texts_folder,
                               picklefolder): 
    
    fnames = io_utils.getfilenames_of_dir(texts_folder, removeextension=False)
    
    for fname in fnames:
        
        print("\n", fname)
        inp = os.path.join(texts_folder, fname)
        picklepath = os.path.join(picklefolder, "sents_"+fname[:-4]+".b")  
        _prep_tokenize_save_texts(inp, picklepath)
        

    print("Finished.")



##################################


def run_embedder(pickled_sentences_folder,
                 outfolder,
                 modelname):
    
    sentences = embd_prod.get_pickled_sentences(pickled_sentences_folder)

    params=dict(size=50, min_count=3, window=5,
                        alpha=0.025, max_vocab_size=1500000)
    
    embd_prod.generate_embeddings2(sentences, outfolder, modelname, params)





    
    