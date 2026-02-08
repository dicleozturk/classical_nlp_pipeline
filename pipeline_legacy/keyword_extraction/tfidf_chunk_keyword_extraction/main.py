'''
Created on May 23, 2019

@author: dicle
'''

import gensim





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
    
    tagged_words = itertools.chain.from_iterable(pos_tagger.pos_tag_sentences(text, lang))
    
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower().replace('i̇' , "i") for word, tag in tagged_words
                  if tag in good_tags and word.lower().replace('i̇' , "i") not in stop_words
                  and not all(char in punct for char in word)]

    return candidates



# takes a list of docs (each doc a string)
# returns [[phrase(i,j)]] where phrase(i,j) is the jth top phrase (sorted by tf*idf weight) in doc(i) (each row is for a doc)
# lda, lsi, nmf can be applied to model the topics
def score_keyphrases_by_tfidf(texts, lang, candidate_type='chunks', stem=True, top_n_phrases=10):
    
    # extract candidates from each text in texts, either chunks or words
    if candidate_type == 'chunks':
        boc_texts = [extract_candidate_chunks(text, lang) for text in texts]
    elif candidate_type == 'words':
        boc_texts = [extract_candidate_words(text, lang) for text in texts]
        # stem?
        # strip punctuation?
        if stem:
            boc_texts = [txtprep.stem_words(boc_text, lang) for boc_text in boc_texts]
    
    elif candidate_type == 'all':
        unigrams = [extract_candidate_words(text, lang) for text in texts]
        if stem:
            unigrams = [txtprep.stem_words(word, lang) for word in unigrams]
        chunks = [extract_candidate_chunks(text, lang) for text in texts]
        
        boc_texts = [i+j for i,j in zip(unigrams, chunks)]
        #boc_texts = [random.shuffle(keys) for keys in boc_texts]

        
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    # sort each row (doc vector) by the tf*idf values (the cells contain (word_id, tf*idf_weight))
    sorted_matrix = [sorted(docvect, key=lambda x : x[1], reverse=True)[:top_n_phrases] for docvect in corpus_tfidf]
    # get phrases by ids
    doc_top_phrases = [[(dictionary[pid], round(weight, 2)) for (pid, weight) in docvect] for docvect in sorted_matrix]

    return doc_top_phrases


def score_keyphrases_by_tfidf_single(text, lang, 
                                     candidate_type='all', stem=True, top_n_phrases=10):
    
    if lang in ["tr", "turkish"]:
        aux_sentences = ["Merhaba, nasılsınız?", "Bugün hava ne güzel..", "Buraya standard cümleler eklememiz lazım."]
    else:
        aux_sentences = ["Hello, how are you?", "The weather is so nice today..", "We need standard auxiliary words to make this corpus stronger."]
    texts = [text] + aux_sentences
    
    docs_top_phrases = score_keyphrases_by_tfidf(texts, lang, candidate_type, stem, top_n_phrases)

    return docs_top_phrases[0]




    print()