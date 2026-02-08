'''
Created on Oct 25, 2018

@author: dicle
'''

import sys
sys.path.append("..")

import nltk


from language_tools.tokenizers import token_splitter, sentence_splitter


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
    
    def __init__(self, word, freq, pos, vscore):
        
        self.word = word
        self.frequency = freq
        self.positions = pos
        self.vertex_score = vscore
    
    def __str__(self):
        return self.word
    
    def __eq__(self, other):
        return self.word == other.word
    
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
def get_words(corpus, lang):
    
    all_words = []
    doc_base = []   # [(doc_id, (sentence_id, [word_i])]
    
    for i,doc in enumerate(corpus):

        doc_tokens = token_splitter.text_to_words(doc, lang=lang, remove_punkt=True)
        all_words.extend(doc_tokens)
        
        doc_sentences = sentence_splitter.text_to_sentences(doc, lang=lang)
        doc_base_sentences = []
        for sent_id,sentence in enumerate(doc_sentences):
            sent_tokens = token_splitter.text_to_words(sentence, lang=lang, remove_punkt=True)
            doc_base_sentences.append((sent_id, sent_tokens))
        doc_base.append((i, doc_base_sentences))
    '''
    for i,doc in enumerate(corpus):

        doc_tokens = txtprep.tokenizer(doc, remove_punktTrue=True)
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
def build_coocc_matrix(words, doc_base, window=0):
    
    w_texts = [w.word for w in words]
    coocc_matrix = []  # [(w1, w2, ncoocc)]
    for w1 in words:
        for w2 in words:

            if w1 != w2:

                w1_index = w1.get_position_index()
                w2_index = w2.get_position_index()
                co_occ = [i for i in w1_index if i in w2_index]
                n_coocc_sent = len(co_occ)
                                   
                
                if window == 0:
                    coocc_matrix.append((str(w1), str(w2), n_coocc_sent))
                
                '''
                if window == 0:
                    w1_index = w1.get_position_index()
                    w2_index = w2.get_position_index()
                    co_occ = [i for i in w1_index if i in w2_index]
                    n_coocc = len(co_occ)
                    coocc_matrix.append((str(w1), str(w2), n_coocc))
                elif window > 0:
                '''   
                    
    



    w1 = Word("ali", 0, [], 1)
    w2 = Word("ali", 1, [], 0)
    print(w1 == w2)
    print(str(w1))
    
    
    