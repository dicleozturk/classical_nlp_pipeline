'''
Created on Nov 2, 2018

@author: dicle
'''


import sys
sys.path.append("..")


import gensim
import itertools, nltk, string
from pprint import pprint

from language_tools.TokenHandler import TrTokenHandler
from language_tools import pos_tagger
from language_tools import stopword_lists
import text_categorization.prototypes.feature_extraction.text_preprocessor as txtprep





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




# doc_phrase_weight_list = [(word(i,j), weight)] for the jth word of doc_i
def print_top_phrases(doc_phrase_weight_list, print_weight=False):   
    
    for i,tuples in enumerate(doc_phrase_weight_list):
        print("Doc ",i," : ", end="")
        
        if print_weight:
            wlist = [(word, round(weight, 2)) for word, weight in tuples]
        else:
            wlist = [word for word,_ in tuples]
        
        pprint(wlist)
        


        
def extract_keyphrases_tfidf(texts, lang, candidate_type="all", top_n_phrases=15):
    
    doc_topphrases_tfidf = score_keyphrases_by_tfidf(texts, lang, 
                                                     candidate_type=candidate_type, top_n_phrases=top_n_phrases)
    
    
    print_top_phrases(doc_topphrases_tfidf, print_weight=True)



def post_process_keyphrases(keywords, lang,
                            stopword=True, more_stopwords=None,
                             stemming=True, 
                             remove_numbers=T, deasciify=False, 
                             remove_punkt=True,
                             lowercase=True):

    
    
    TrTokenHandler()
    
    


    from time import time
    docs = ["""Renault, binek otomobil ve hafif ticari modellerinde sıfır faiz ve cazip fırsatlar sunuyor.   Clio HB ve Clio Sport Tourer modellerinin benzinli manuel versiyonlarında müşteri araç satış bedelinin en az yarısını takas ya da nakit olarak peşin ödeyebiliyor.   Kalan yarısı için maksimum 36 bin TL kredi için 12 ay vadede yüzde sıfır faiz fırsatı bulunuyor.   Ayrıca bu modeller için liste fiyatı üzerinden 3 bin 500 TL ekstra indirim de sunuluyor.""",
            """Başkentin kavurucu sıcaklarından bunalan vatandaşlar soluğu, Yenimahalle Belediyesi'nin 5 yıldızlı otel konforunu aratmayan olimpik yüzme havuzlarında alıyor.   Yenimahalle Belediyesi'nin Bülent Ecevit Yüzme Havuzu ve Sosyal Tesisleri, Turgut Özakman Yüzme havuzu ve Sosyal Tesisleri ve Hacı Sabancı Yüzme Havuzu ve Sosyal Tesisleri olmak üzere 3 ayrı bölgede hizmet veren havuzları vatandaşların akınına uğruyor."""
            ]
    lang = "tr"
    
    #extract_keyphrases_tfidf(docs, lang, candidate_type="chunks", top_n_phrases=50)
    print()
    
    text = docs[1]
    
    ### deneme 23 Mayıs 19
    
    text = """"Mardin'in Savur ilçesinde oturan Nur Sema Demir, 31 Temmuz'da açıklanan üniversite sınavında, sayısal bölüm sorularından 253, sözelden 278 ve temel yeterlilik testinde 324 puan aldı. Aldığı puana göre tercih yapan Demir, 1 ay sonra açıklanan yerleştirme sonucunda sayısal puanının, '0' olduğunu ve 4 yıllık bir fakülte beklerken, 18'inci tercihi olan 2 yıllık Moda Tasarımı Bölümü'ne yerleştirildiğini öğrenince şok yaşadı. Demir ailesi, ilk sınav sonucunun yerleştirme sonucundan çok farklı olduğununu, 4 yıllık bir okula yerleştirilmesini sağlayan sayısal puanın ikinci sonuç belgesinde sıfır gösterildiğini ve sonuç belgesindeki doğru ve yanlış soru sayılarının farklı olduğunu görünce ÖSYM'ye itirazda bulundu. ÖSYM tarafından Nur Sema Demir'e gönderien yanıtta, itiraz için geç kaldığı belirtildi. Demir ailesi, yapılan yanlış nedeniyle kızlarının mağdur edildiğini söyledi."
    """
    
    ###
    lang = "tr"
    candidate_type = "all"
    stem = False
    top_n_phrases = 15
    
    t1 = time()
    key_phrases = score_keyphrases_by_tfidf_single(text, lang, candidate_type, stem, top_n_phrases)
    pprint(key_phrases)
    t2 = time()
    
    print(round(t2-t1, 2), "sec.")
    