'''
Created on Oct 25, 2018

@author: dicle
'''

import sys
sys.path.append("..")

import nltk
import math
import numpy as np

import copy


from language_tools.tokenizers import token_splitter, sentence_splitter
from keyword_extraction.keyphrase_extraction_methods.helpers import embds_conf
from keyword_extraction.keyphrase_extraction_methods.helpers.embeddings_reader import EmbeddingReader



'''
The method adapted from Wang, Rui, Wei Liu, and Chris McDonald. 
                              "Corpus-independent generic keyphrase extraction using word embedding vectors." 
                              Software Engineering Research Conference. Vol. 39. 2014.

'''



'''
Corpus, list of docs with a list of sentences.
extract keyphrases in each doc.
 1- make a list of all words
 2- find word frequencies
 3- find word co-occ freq.s of window k or sentence
 4- compute f, dice, attr for word pairs
 5- make graph, compute S for each word
 6- keyphrases:
  i- select adjacent words as phrases
  ii- select highest n scoring (with total word scores) as keyphrases 
'''




class KeyphraseModel:
    
    def __init__(self):
        return

class Word:
    
    word = ""
    frequency = 0.0
    positions = None    #[(doc_id, sentence_id, word_index)]
    vertex_score = 0.0
    
    embedding = None
    
    def __init__(self, word, freq, pos, vscore):
        
        self.word = word
        self.frequency = freq
        self.positions = pos
        self.vertex_score = vscore
        
        self.embedding = None #self.get_embedding()
    
    
    
    # modify!!    
    def set_embedding(self, embeddings_set):
        vect = embeddings_set.get_word_vector(self.word)
        self.embedding = vect
        return self.embedding
        #return np.random.uniform(-0.25, 0.25, 25)

        
    def __str__(self):
        return self.word
    
    def __eq__(self, other):
        return self.word == other.word
    
    def tostr(self):
        return self.word + " : " + str(self.positions)
    
    def __cmp__(self, other):
        return self.word > other.word
    
    def __hash__(self):
        return hash(self.word)
    
    '''
     return [(doc_id, sent_id)] pairs
    '''
    def get_position_index(self):
        l = []
        for docid, sents in self.positions:
            for sentid,_ in sents:
                l.append((docid, sentid))
        
        return l



def get_word_index(word, wordlist):
    
    word_index = -1
    try:
        word_index = wordlist.index(word)
    except ValueError:
        pass
    
    return word_index



def get_word_positions_in_docbase(word, doc_base):

    positions = []   # [(doc_id, sent_id, word_index)]
    for docid, sentences in doc_base:
        
        for sentid, words in sentences:
            
            w_index = get_word_index(word, words)
            if w_index > -1:
                positions.append((docid, sentid, w_index))
    
    return positions
            
            


'''
 corpus = [doc] where doc is a text.
'''
def model_words(corpus, lang):
    
    all_words = []
    doc_base = []   # [ doc_id, [sentence_id : [word_i]]]
    for i,doc in enumerate(corpus):

        doc_tokens = token_splitter.text_to_words(doc, lang=lang, remove_punkt=True)
        all_words.extend(doc_tokens)
        
        doc_sentences = sentence_splitter.text_to_sentences(doc, lang=lang)
        doc_base_sentences = []
        for sent_id,sentence in enumerate(doc_sentences):
            sent_tokens = token_splitter.text_to_words(sentence, lang=lang, remove_punkt=False)
            doc_base_sentences.append((sent_id, sent_tokens))
        doc_base.append((i, doc_base_sentences))
    '''
    for i,doc in enumerate(corpus):

        doc_tokens = txtprep.tokenizer(doc, remove_punkt=True)
        all_words.extend(doc_tokens)
        
        doc_sentences = tr_sentence_splitter.text_to_sentences(doc)
        doc_base_sentences = []
        for sent_id,sentence in enumerate(doc_sentences):
            sent_tokens = txtprep.tokenizer(sentence, remove_punkt=False)
            doc_base_sentences.append((sent_id, sent_tokens))
        doc_base.append((i, doc_base_sentences))
    '''
    
    # consider stemming!!
    wordfreqs = nltk.FreqDist(all_words)
    distinct_words = list(wordfreqs.keys())
    
    words = []
    for word in distinct_words:
        freq = wordfreqs.freq(word)
        positions = get_word_positions_in_docbase(word, doc_base)
        vertex_score = 0.0
        words.append(Word(word=word, freq=freq, pos=positions, vscore=vertex_score))
    
    return words




'''
 words = [Word]
 window: k (int) (>0) or sentence if k=0 
'''
def build_coocc_matrix(words, window=0):
       
    
    coocc_matrix = {}  # {(w1, w2) : ncoocc}
                   
    for w1 in words:
        for w2 in words:

            if (w1 != w2) and ((w2, w1) not in coocc_matrix.keys()):
                n_coocc = 0
                w1_pos = w1.positions
                w2_pos = w2.positions
                w1_pos_ = [(i,j) for i,j,_ in w1_pos]  # w/o word_index [(doc_id, sent_id)]
                w2_pos_ = [(i,j) for i,j,_ in w2_pos]
                sent_coocc = [i for i in w1_pos_ if i in w2_pos_]
                n_coocc_sent = len(sent_coocc)
                
                                 
                if window == 0:
                    n_coocc = n_coocc_sent
                elif window > 0:                    
                    w1_neighbours = [(i,j,k+n) for n in range(1,window+1) for i,j,k in w1_pos]
                    n_coocc_window = [i for i in w1_neighbours if i in w2_pos]
                    n_coocc = len(n_coocc_window)
                
                #if n_coocc > 0:
                coocc_matrix[(w1, w2)] = n_coocc
                
                    
                        
                 
    return coocc_matrix                



'''
def build_coocc_matrix(words, window=0):
       
    
    coocc_matrix = {}  # {(w1, w2) : ncoocc}
                   
    for w1 in words:
        for w2 in words:

            if (str(w1) != str(w2)) and ((str(w2),str(w1)) not in coocc_matrix.keys()):
                n_coocc = 0
                w1_pos = w1.positions
                w2_pos = w2.positions
                w1_pos_ = [(i,j) for i,j,_ in w1_pos]  # w/o word_index [(doc_id, sent_id)]
                w2_pos_ = [(i,j) for i,j,_ in w2_pos]
                sent_coocc = [i for i in w1_pos_ if i in w2_pos_]
                n_coocc_sent = len(sent_coocc)
                
                                 
                if window == 0:
                    n_coocc = n_coocc_sent
                elif window > 0:                    
                    w1_neighbours = [(i,j,k+n) for n in range(1,window+1) for i,j,k in w1_pos]
                    n_coocc_window = [i for i in w1_neighbours if i in w2_pos]
                    n_coocc = n_coocc_window
                
                if n_coocc > 0:
                    coocc_matrix[(str(w1), str(w2))] = n_coocc
                        
                
                 
    return coocc_matrix     
'''


'''
 v1 and v2 are vectors (word embeddings)
'''
def euclidean_distance(v1, v2):
    dist = np.linalg.norm(v1-v2)
    return round(dist, 4)

def compute_f(words):
    
    f_matrix = {}
    
    for w1 in words:
        for w2 in words:
            if (w1 != w2) and ((w2, w1) not in f_matrix.keys()):
                
                f_value = 0
                
                freq_w1 = w1.frequency
                freq_w2 = w2.frequency
                
                vect_w1 = w1.embedding
                vect_w2 = w2.embedding
                
                
                word_distance = euclidean_distance(vect_w1, vect_w2)
                dsquare = word_distance ** 2
                
                if not math.isclose(dsquare, 0):
                    
                    f_value = (freq_w1 * freq_w2) / dsquare
                
                f_matrix[(w1, w2)] = f_value
    
    return f_matrix
                
            
def compute_dice(c_matrix):
    dice_matrix = dict.fromkeys(c_matrix.keys())   
    
    for pair, co_freq_val in c_matrix.items():
        
        w1, w2 = pair
        freq_w1 = w1.frequency
        freq_w2 = w2.frequency
        dice_val = 0.0
        
        denom = freq_w1 * freq_w2
        if not math.isclose(denom, 0):
            dice_val = (2*co_freq_val) / denom
        
        dice_matrix[pair] = dice_val
        
        '''
        if co_freq_val == 0.0:
            print("---------- ", w1, w2, dice_val)
        '''
        
    return dice_matrix


def compute_attraction(dice_matrix, f_matrix):
    
    attr_matrix = dict.fromkeys(dice_matrix.keys())
    
    for pair in attr_matrix.keys():
        
        attr_matrix[pair] = dice_matrix[pair] * f_matrix[pair]
    
    return attr_matrix



# pairs = [(w1, w2)]
# return [w1, w2,...]  with no duplicates
def get_pair_items(pairs):
    nodes = []
    for a,b in pairs:
        nodes.append(a)
        nodes.append(b)
    nodes = list(set(nodes))
    return nodes

def build_graph(attr_matrix):
        
    # init graph
    nodes = get_pair_items(attr_matrix.keys())
    graph = dict.fromkeys(nodes, {})  # {wi : {wj : attr_val}}  where attr_val > 0
    
    for (w1, w2), attr_val in attr_matrix.items():
        if attr_val > 0:
            graph[w1][w2] = attr_val
            graph[w2][w1] = attr_val
    
    return graph
            

# apply modified pagerank
#  graph has attr values as edge weights
def build_attr_rank_graph(graph):  
    
    dumping_factor = 0.85    # should be in a file!!
    error_rate = 0.0001
    
    nodes = list(set(list(graph.keys())))  # unique node list
    sgraph = dict.fromkeys(nodes, 1.0)  # {wi : score}; initially all score values are 0.
    
    total_diff = 0.0
    i = 0
    while total_diff < error_rate:
        for node_i, edges in graph.items():
            
            #print(node_i.word)
            
            total_i = 0.0
            for node_j, attr_ij in edges.items():
                if node_j != node_i:
                    neighbours_j = graph[node_j]
                    total_j = 0.0
                    for node_k, attr_jk in neighbours_j.items():
                        if node_k != node_j:
                            total_j += attr_jk
                
                    total_i += (attr_ij * sgraph[node_j]) / total_j
                    #print("  ", node_j.word, attr_ij, total_i)
            
            score_i = (1-dumping_factor) + dumping_factor * total_i
            #print(total_i, "  ", score_i)
            prev_score_i = sgraph[node_i]
            diff_i = score_i - prev_score_i
            total_diff += diff_i
            
            sgraph[node_i] = score_i 
            i += 1
            #print(" ", node_i.word, "  ", prev_score_i)
            #print("iteration ",i, diff_i, total_diff)
            
    
    return sgraph


def make_phrases(sgraph, c_matrix):
    
    pairs_scores = {} # {(wi, wj) : (score_i, score_j)} -- co-occurring words and the words' own scores
    for (w1, w2), val in c_matrix.items():
        if val > 0.0:
            pairs_scores[(w1, w2)] = (sgraph[w1], sgraph[w2])
    
    return pairs_scores


def select_phrases(phrases_scores, select_n=None):

    phrases = {} # {(w1, w2) : score}
    for (w1, w2),(score1, score2) in phrases_scores.items():
        
        phrase_score = score1 + score2
        phrases[(w1, w2)] = phrase_score
    
    
    sphrases = sorted(phrases.items(), reverse=True, key=lambda x : x[1])
    if select_n:
        sphrases = sphrases[:select_n]
    return sphrases




def keyphrase_extraction_pipeline(docs, lang, window=1, nphrases=None):
    
    if lang in ["tr", "turkish"]:
        embeddings_param = embds_conf.embeddings_param_tr
    else:
        embeddings_param = embds_conf.embeddings_param_en
    
    embeddings_reader = EmbeddingReader(embds_path=embeddings_param["path"],
                                        tokenizer=embeddings_param["tokenizer"],
                                        binary=embeddings_param["binary"])
    embeddings_reader.load_embeddings()
    
    words = model_words(docs, lang) 
    # assign embeddings
    for w in words:
        w.set_embedding(embeddings_reader)
        
        
    print("0000000")
    print("building cmatrix")
    cmatrix = build_coocc_matrix(words, window=window)    
    print("building fmatrix")
    fmatrix = compute_f(words)
    print("building dmatrix")
    dmatrix = compute_dice(cmatrix)
    print("building amatrix")
    amatrix = compute_attraction(dmatrix, fmatrix)
    graph = build_graph(amatrix)
    score_graph = build_attr_rank_graph(graph)
    candidate_phrases = make_phrases(score_graph, cmatrix)
    selected_phrases = select_phrases(candidate_phrases, nphrases)
    phrases = [(w1.word, w2.word) for (w1, w2),_ in selected_phrases]

    '''
    for (w1,w2),score in selected_phrases:
        print(w1, w2, "  ", score)
    '''
    return phrases


    #main()
    docs = ["""Renault, binek otomobil ve hafif ticari modellerinde sıfır faiz ve cazip fırsatlar sunuyor.   Clio HB ve Clio Sport Tourer modellerinin benzinli manuel versiyonlarında müşteri araç satış bedelinin en az yarısını takas ya da nakit olarak peşin ödeyebiliyor.   Kalan yarısı için maksimum 36 bin TL kredi için 12 ay vadede yüzde sıfır faiz fırsatı bulunuyor.   Ayrıca bu modeller için liste fiyatı üzerinden 3 bin 500 TL ekstra indirim de sunuluyor.""",
            """Başkentin kavurucu sıcaklarından bunalan vatandaşlar soluğu, Yenimahalle Belediyesi'nin 5 yıldızlı otel konforunu aratmayan olimpik yüzme havuzlarında alıyor.   Yenimahalle Belediyesi'nin Bülent Ecevit Yüzme Havuzu ve Sosyal Tesisleri, Turgut Özakman Yüzme havuzu ve Sosyal Tesisleri ve Hacı Sabancı Yüzme Havuzu ve Sosyal Tesisleri olmak üzere 3 ayrı bölgede hizmet veren havuzları vatandaşların akınına uğruyor."""
            ]
    lang = "tr"
    window = 1
    nphrases = None
    
    
    en_texts = ["Recent studies on Turkish NER make use of rather statistical and lately neural approaches. Thesystems mostly rely on the features mostly found in formal text while social media data or textsfrom other domains may not carry such features or might be very different stylistically [5] and toovercome such noise or to extend to other domains 1) normalization 2) domain adaptation issuggested [6, 8]. Currently, using word embeddings beside named entity features is very popular and said to be overcoming such problems.",
                "We went through a subset of studies on NER (details follow below), largely on Turkish NER. It appears that methods employing (neural) word embeddings (vector representations of words using neural networks) together with a sequence tagger or a neural network give very good results and are open to much more improvements compared to classical methods. Indeed, such an approach is applicable to other languages without difficulty."
                ]
    lang = "en"
       
    en_phrases = keyphrase_extraction_pipeline(en_texts, lang, window, nphrases)
    from pprint import pprint
    pprint(en_phrases)
    
    
    