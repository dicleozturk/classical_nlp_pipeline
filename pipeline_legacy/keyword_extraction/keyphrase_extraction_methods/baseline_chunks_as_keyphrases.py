'''
Created on Nov 2, 2018

@author: dicle
'''

import itertools, nltk, string


from language_tools import pos_tagger
from language_tools import stopword_lists



def extract_candidate_chunks(text, lang):
    
    
    if lang in ["en", "english", "eng"]:
        grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    
    
        # exclude candidates that are stop words or entirely punctuation
        punct = set(string.punctuation)
        stop_words = set(stopword_lists.get_stopwords(lang)) #set(nltk.corpus.stopwords.words('english'))
        # tokenize, POS-tag, and chunk using regular expressions
        chunker = nltk.chunk.regexp.RegexpParser(grammar)
        tagged_sents = pos_tagger.pos_tag_sentences(text, lang)
        #print("nsents: %d" % len(tagged_sents))
        all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                        for tagged_sent in tagged_sents))
        #print("nchunks: %d" % len(all_chunks))
    
        # join constituent chunk words into a single chunked phrase
        # i = 'i̇'    PROBLEM IN DOWNCASING İ  
        candidates = [' '.join(word for word, pos, chunk in group).lower().replace('i̇' , "i") for key, group in itertools.groupby(all_chunks, lambda x : x[2] != 'O') if key]
    
        #print("ncandidates: %d" % len(candidates))
    
        candidates = [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]
        #candidates = [cand for cand in candidates if not all(char in punct for char in cand)]
    
        #print("ncandidates: %d" % len(candidates))
    
    elif lang in ["tr", "turkish"]:
        
        import tr_chunker.TR_chunker.chunker_main2 as trchunker
        candidates = trchunker.chunk_text(text)
    
    return candidates



    text = "Hello, there is a green apple on the blue table which is so nice.."
    
    print(extract_candidate_chunks(text, lang="en"))
    
    
    