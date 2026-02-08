'''
Created on Oct 12, 2016

@author: dicle
'''


import itertools, nltk, string

from language_tools import stopword_lists
from language_tools.tr_postag_syntaxnet import postag_sentences


'''
Adapted from http://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/

'''



'''
def extract_candidate_words(text, lang, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    
    stop_words = set(stopword_lists.get_stopwords(lang))   #set(nltk.corpus.stopwords.words('english'))
    
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates


def extract_candidate_chunks(text, lang, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(stopword_lists.get_stopwords(lang)) #set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    #print("nsents: %d" % len(tagged_sents))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    #print("nchunks: %d" % len(all_chunks))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower() for key, group in itertools.groupby(all_chunks, lambda x : x[2] != 'O') if key]

    #print("ncandidates: %d" % len(candidates))

    candidates = [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]
    #print("ncandidates: %d" % len(candidates))
    return candidates

'''


def pos_tag_sentences(text, lang):

    postagged_sentences = []

    if lang in ["en", "eng", "english"]:
        postagged_sentences = nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                    for sent in nltk.sent_tokenize(text))
    elif lang in ["tr", "turkish"]:
        import language_tools.tr_postag_syntaxnet as tr_syntaxnet
        #from language_tools import tokenization
        #sentences = tokenization.tr_sentence_tokenizer(text)
        #print(len(sentences))
        #postagged_sentences = tr_syntaxnet.postag_sentences(sentences)
        postagged_sentences = tr_syntaxnet.postag_text(text)

    return postagged_sentences



def extract_candidate_words(text, lang):
    
    good_tags = set(['JJ','JJR',  # comparative
                   'JJS',  # superlative
                   'NN',
                   'NNP',  # proper
                   'NNS', # plural noun
                   'NNPS'  # plural proper noun
                   ])
    
    if lang in ["en", "eng", "english"]:
        good_tags = set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])
    elif lang in ["tr", "turkish"]:
        good_tags = set(['ADJ', 'NOUN', 'PROPN'])
    else:
        return []
                      
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    
    stop_words = set(stopword_lists.get_stopwords(lang))   #set(nltk.corpus.stopwords.words('english'))
    
    # tokenize and POS-tag words
    '''
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    '''
    
    tagged_words = itertools.chain.from_iterable(pos_tag_sentences(text, lang))
    
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower().replace('i̇' , "i") for word, tag in tagged_words
                  if tag in good_tags and word.lower().replace('i̇' , "i") not in stop_words
                  and not all(char in punct for char in word)]

    return candidates


def extract_candidate_chunks(text, lang):
    
    
    if lang in ["en", "english", "eng"]:
        grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    
    
        # exclude candidates that are stop words or entirely punctuation
        punct = set(string.punctuation)
        stop_words = set(stopword_lists.get_stopwords(lang)) #set(nltk.corpus.stopwords.words('english'))
        # tokenize, POS-tag, and chunk using regular expressions
        chunker = nltk.chunk.regexp.RegexpParser(grammar)
        tagged_sents = pos_tag_sentences(text, lang)
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




'''
def extract_candidate_chunks(text, lang):
    
    
    if lang in ["en", "english", "eng"]:
        grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    elif lang in ["tr", "turkish"]:
        grammar = r'KT: {(<ADJ>* <(NOUN|PROPN)>+ <(ADP|SCONJ)>)? <ADJ>* <(NOUN|PROPN)>+}'
    else:
        return []
    
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(stopword_lists.get_stopwords(lang)) #set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = pos_tag_sentences(text, lang)
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
    return candidates


'''

def extract_candidate_chunks2(text, lang):
    
    
    if lang in ["en", "english", "eng"]:
        #grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
        #grammar = r"""KT: {<NN.*|JJ>*<NN.*>}"""
        
        
        grammar = r'KT: {(<JJ>* <CD>* <NN.*>+ <IN>)? <JJ>* <CD>* <NN.*>+}'
        grammar = r'KT: {((<JJ>* <CD>*)* (<CD>* <JJ>*)* <NN.*>+ <IN>)? (<JJ>* <CD>*)* (<CD>* <JJ>*)* <NN.*>+}'
        grammar = r'KT: {((<JJ>* <CD>*)* (<CD>* <JJ>*)* <NN.*>+)+}'
        
    elif lang in ["tr", "turkish"]:
        grammar = r'KT: {(<ADJ>* <(NOUN|PROPN)>+ <(ADP|SCONJ)>)? <ADJ>* <(NOUN|PROPN)>+}'
    else:
        return []
    
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(stopword_lists.get_stopwords(lang)) #set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = pos_tag_sentences(text, lang)
    #print(tagged_sents)
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
    return candidates



def extract_candidate_chunks3(text, lang):
    
    
    if lang in ["en", "english", "eng"]:
        #grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
        grammar = r"""KT: {<NN.*|JJ>*<NN.*>}"""
        #grammar = r'KT: {(<JJ>* <CD>* <NN.*>+ <IN>)? <JJ>* <CD>* <NN.*>+}'

        
    elif lang in ["tr", "turkish"]:
        grammar = r'KT: {(<ADJ>* <(NOUN|PROPN)>+ <(ADP|SCONJ)>)? <ADJ>* <(NOUN|PROPN)>+}'
    else:
        return []
    
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(stopword_lists.get_stopwords(lang)) #set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = pos_tag_sentences(text, lang)
    #print(tagged_sents)
    #print("nsents: %d" % len(tagged_sents))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    #print("nchunks: %d" % len(all_chunks))
    #print(all_chunks)
    # join constituent chunk words into a single chunked phrase
    # i = 'i̇'    PROBLEM IN DOWNCASING İ  
    candidates = [' '.join(word for word, pos, chunk in group).lower().replace('i̇' , "i") for key, group in itertools.groupby(all_chunks, lambda x : x[2] != 'O') if key]

    #print("ncandidates: %d" % len(candidates))

    candidates = [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]
    #candidates = [cand for cand in candidates if not all(char in punct for char in cand)]

    #print("ncandidates: %d" % len(candidates))
    return candidates


def get_chunks():
    
    # nltk or blipparsertree
    return


    #text = "A clause is a finite verb and all its arguments, i.e. a main verb and everything that depends on it. If you have a sentence with a single main verb, the entire sentence is one clause. Conjunctions and relative pronouns typically introduce new clauses."
    text = "I like the most beautiful apples and the very delicious cherries"
    x = extract_candidate_chunks(text, lang="english")
    print(x)


    
    tr_text = "Başta inşaat olmak üzere birçok sektörün canlanacağı, nakliye masraflarının düşeceği, İzmirli işadamlarının daha sık İstanbul ve Avrupa'ya ulaşacağı, kentin daha fazla yatırım alacağı, daha çok turist çekeceği belirtildi.Türkiye'nin en büyük gıda ihracatının gerçekleştirildiği Ege Bölgesi ihracatına büyük katkı sağlayacak otoyolun, zamanla yarışılan gıda ürünleri ihracatçılarına büyük avantajlar sağlayacağı, otoyolun sadece iki şehri değil, aynı zamanda Ege ve Marmara Ekonomisini birbirine bağlayacağı kaydedildi."
    #tr_text = "Bugün güzel bir gün! Güneş burada, ısınan hava içimizde.."
    y = extract_candidate_chunks(tr_text, lang="turkish")
    print(extract_candidate_words(tr_text, lang="tr"))
    print(y)
    
    
    '''
    print("en:")
    print(pos_tag_sentences(text, lang="en"))
    print("tr:")
    print(pos_tag_sentences(tr_text, lang="tr"))
    '''
    
    