'''
Created on May 22, 2017

@author: dicle
'''

import os

import numpy as np
import pandas as pd
import random
import gensim


import embeddings.generate_tr_embeddings_gensim3 as utils
from dataset import io_utils


# gold_analogy_pairs = [[(w11, w12), (w21, w22)]]
def evaluate_analogy(model, gold_analogy_pairs):
        
    extracted_analogy_pairs = []
    
    for pair_item in gold_analogy_pairs:
        pair1, pair2 = pair_item
        p1 = list(pair1)
        p2_ = pair2[0]
        try:
            #analogous_words = model.most_similar(positive=p1, negative=p2_)
            sim1 = model.similarity(pair1[0], pair1[1])
            sim2 = model.similarity(pair2[0], pair2[1])
            n_sim = model.n_similarity(list(pair1), list(pair2))
        except KeyError:
            sim1 = -1
            sim2 = -1
            n_sim = -1
        
        #extracted_analogy_pairs.append([pair1, (analogous_words, pair2[1])])
        #print(pair1, pair2, " -- >", analogous_words)
        from collections import OrderedDict
        row = OrderedDict.fromkeys(["w11", "w12", "sim1", "w21", "w22", "sim2", "n_sim"])
        row["w11"] = pair1[0]
        row["w12"] = pair1[1]
        row["sim1"] = sim1
        row["w21"] = pair2[0]
        row["w22"] = pair2[1]
        row["sim2"] = sim2
        row["n_sim"] = n_sim
                                       
        extracted_analogy_pairs.append(row)
    
    return extracted_analogy_pairs



def get_words_closest_to_vect(model, vect):
    
    closest_words = model.similar_by_vector(vect)
    #closest_word = wordvect[0]
    return closest_words


def evaluate_by_analogy(model, gold_analogy_path):
    
    gold_analogy_pairs = open(gold_analogy_path, "r").readlines()
    #gold_analogy_pairs = random.sample(gold_analogy_pairs, 20)
    
       
    result = []
    
    for line in gold_analogy_pairs:
        
        items = line.split()
        
        if len(items) == 4:
            
            #vects = [model.word_vec(word) for word in items[:3]]
            
            vects = []
            for word in items[:3]:
                try:
                    vect = model.word_vec(word)
                except KeyError:
                    #vect = np.zeros(model.vector_size)  # random??
                    vect = np.random.uniform(-0.25, 0.25, model.vector_size) 
                finally:
                    vects.append(vect)
                    
            diff_pair1 = vects[1] - vects[0]
            target_vect = diff_pair1 + vects[2]
            
            closests = get_words_closest_to_vect(model, target_vect)
            print(closests)
            closests = [(w, sim) for w,sim in closests if w != items[2]]
            
            print(items)
            print()
            w = closests[0][0]
            sim = closests[0][1]
            result.append({"gold" : items[3],
                                   "closest" : w,
                                   "closest_sim" : sim,
                                   "w11" : items[0],
                                   "w12" : items[1],
                                   "w21" : items[2]
                                   })
            '''
            for w, sim in closests:
                if w not in items[:3]:
                    result.append({"gold" : items[3],
                                   "closest" : w,
                                   "closest_sim" : sim,
                                   "w11" : items[0],
                                   "w12" : items[1],
                                   "w21" : items[2]
                                   })
                    break
            '''
    return result       
    

# the file on fpath has <w11 w12 w21 w22> per line where the similarity of w11 and w12 is analogous to that of w21 and w22.
# returns [[(w11, w12), (w21, w22)]]
def read_analogy_pairs(fpath):
    
    similar_pairs = []
    
    f = open(fpath, "r")
    lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        items = line.split()
        if len(items) == 4:
            pair1 = items[:2]
            pair2 = items[2:]
            similar_pairs.append([tuple(pair1), tuple(pair2)])
    
    f.close()
    return similar_pairs


# pairs: [[(w11, w12), (w21, w22)]]
# save it to the file at filepath, <w11 w12 w21 w22> per line
def save_analogy_pairs(pairs, filepath):
    
    f = open(filepath, "w")
    for item in pairs:
        ws = []
        ws.extend(item[0])
        ws.extend(item[1])
        f.write(" ".join(ws))
    
    f.close()
    



# syndf = [(closest, closest_sim, gold, w11, w12, w21)] is a df.
# find the rate of matching synonyms
def performance(syndf):
    
    c = 0
    for i in syndf.index.values:
        
        gold_word = syndf.loc[i, "gold"]
        found_word = syndf.loc[i, "closest"]

        if gold_word == found_word:
            c += 1
    
    acc = float(c) / syndf.shape[0]
    return c, acc





   
    