'''
Created on Nov 2, 2018

@author: dicle
'''


import sys
sys.path.append("..")


import re
import gensim
import itertools, nltk, string
from pprint import pprint

from language_tools.TokenHandler import TrTokenHandler
from language_tools import pos_tagger, stemmers
from language_tools import stopword_lists
import text_categorization.prototypes.feature_extraction.text_preprocessor as txtprep
from language_tools import tr_sentence_splitter





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
    
    
    tagged_words_list = pos_tagger.pos_tag_sentences(text, lang)
    
    '''
    # label mismarked gerunds (çalışmak, yapmak, araştırma) as NOUN, mislabelled by syntaxnet as VERB
    if lang in ["tr", "turkish"]:
        _list = []
        for tagged_sentence in tagged_words_list:
            __list = []
            for word, tag in tagged_sentence:
                if tag == "VERB" and (re.search(r"m[ae]k?$", word)):
                    tag = "NOUN"
                __list.append((word, tag))
            _list.append(__list)
        tagged_words_list = _list
    '''
    
    
    # filter on certain POS tags and lowercase all words
    tagged_words = itertools.chain.from_iterable(tagged_words_list)
    candidates = [word.lower().replace('i̇' , "i") for word, tag in tagged_words
                  if tag in good_tags and word.lower().replace('i̇' , "i") not in stop_words
                  and not all(char in punct for char in word)]
    
    print("candidates - b.p: ", candidates)
    candidates = [remove_surrounding_punctuation(word) for word in candidates]
    print("candidates - a.p: ", candidates)
    return candidates


def remove_surrounding_punctuation(word):
    p = "[,\.\!\;]"
    w = word
    if re.match("^"+p, w):
        w = w[1:]
    if re.search(p+"$", w):
        w = w[:-1]
    return w


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


'''
 adds auxiliary texts to the single text inputs
'''
def check_corpus(texts, lang):
    
    if type(texts) == str:
        texts = [texts]
    if len(texts) == 1:
        
        if lang in ["tr", "turkish"]:
            aux_sentences = ["Merhaba, nasılsınız?", "Bugün hava ne güzel..", "Buraya standard cümleler eklememiz lazım."]
        else:
            aux_sentences = ["Hello, how are you?", "The weather is so nice today..", "We need standard auxiliary words to make this corpus stronger."]
        texts.extend(aux_sentences)
    return texts

def search_in_list_of_wordlists(word, list_of_wordlists):
    for wordlist in list_of_wordlists:
        if word in wordlist:
            return True
    return False

# takes a list of docs (each doc a string)
# returns [[phrase(i,j)]] where phrase(i,j) is the jth top phrase (sorted by tf*idf weight) in doc(i) (each row is for a doc)
# lda, lsi, nmf can be applied to model the topics
def score_keyphrases_by_tfidf_chunk_unigram_inclusion(texts, lang, stem=True, top_n_phrases=10,
                              tr_alt_stemming=True):
    
    
    # if there is a single text, append some auxiliary texts so that we can calculate IDF.
    texts = check_corpus(texts, lang)
    
    print(type(texts))
    print(texts)
    # extract candidates from each text in texts, either chunks or words

    unigrams = [extract_candidate_words(text, lang) for text in texts]
    if stem:
        unigrams = [stemmers.stem_words(word_list, lang, tr_alt_stemming) for word_list in unigrams]
    chunks = [extract_candidate_chunks(text, lang) for text in texts]
    
    # merge unigrams and chunks. do not copy unigrams that appear in chunk items
    boc_texts = []
    #chunk_words = [[c.split() for c in text_chunks] for text_chunks in chunks]
    
    for i in range(len(chunks)):
        doc_unigrams = unigrams[i]
        doc_chunks = chunks[i]
        doc_chunks_split = [c.split() for c in doc_chunks]
        doc_boc_texts = doc_chunks.copy()
        for unigram in doc_unigrams:
            if not search_in_list_of_wordlists(unigram, doc_chunks_split):
                doc_boc_texts.append(unigram)
        boc_texts.append(doc_boc_texts)
    
    #boc_texts = [random.shuffle(keys) for keys in boc_texts]

    #print(boc_texts)
    # remove apostrophes and empty strings
    boc_texts = [[word.split("'")[0] for word in boc_text] for boc_text in boc_texts]
    boc_texts = [[word.replace("'", "").replace('"', "").strip() for word in text] for text in boc_texts]
    boc_texts = [[word for word in text if len(word.strip())>1] 
                                                        for text in boc_texts]
    
    print("\n****\nchunks")
    print(unigrams)
    print(chunks)
    print(boc_texts)
    print("***********")
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

    return doc_top_phrases[0]




# takes a list of docs (each doc a string)
# returns [[phrase(i,j)]] where phrase(i,j) is the jth top phrase (sorted by tf*idf weight) in doc(i) (each row is for a doc)
# lda, lsi, nmf can be applied to model the topics
def score_keyphrases_by_tfidf(texts, lang, candidate_type='chunks', stem=True, top_n_phrases=10,
                              tr_alt_stemming=True):
    
    # extract candidates from each text in texts, either chunks or words
    if candidate_type == 'chunks':
        boc_texts = [extract_candidate_chunks(text, lang) for text in texts]
    elif candidate_type == 'words':
        boc_texts = [extract_candidate_words(text, lang) for text in texts]
        # stem?
        # strip punctuation?
        if stem:
            boc_texts = [stemmers.stem_words(boc_text, lang, tr_alt_stemming) for boc_text in boc_texts]
    
    elif candidate_type == 'all':
        unigrams = [extract_candidate_words(text, lang) for text in texts]
        if stem:
            unigrams = [stemmers.stem_words(word_list, lang, tr_alt_stemming) for word_list in unigrams]
        chunks = [extract_candidate_chunks(text, lang) for text in texts]
        
        boc_texts = [i+j for i,j in zip(unigrams, chunks)]
        #boc_texts = [random.shuffle(keys) for keys in boc_texts]

    #print(boc_texts)
    # remove apostrophes and empty strings
    boc_texts = [[word.split("'")[0] for word in boc_text] for boc_text in boc_texts]
    boc_texts = [[word.replace("'", "").replace('"', "").strip() for word in text] for text in boc_texts]
    boc_texts = [[word for word in text if len(word.strip())>1] 
                                                        for text in boc_texts]
    
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


# takes a list of docs (each doc a string)
# returns [[phrase(i,j)]] where phrase(i,j) is the jth top phrase (sorted by tf*idf weight) in doc(i) (each row is for a doc)
# lda, lsi, nmf can be applied to model the topics

# fix: treat sentences as docs, then merge the keywords by aggregating the scores of the common keywords among sentences
def __score_keyphrases_by_tfidf(sentences, lang, candidate_type='chunks', stem=True, top_n_phrases=10):
    
    # extract candidates from each text in texts, either chunks or words
    if candidate_type == 'chunks':
        boc_texts = [extract_candidate_chunks(text, lang) for text in sentences]
    elif candidate_type == 'words':
        boc_texts = [extract_candidate_words(text, lang) for text in sentences]
        # stem?
        # strip punctuation?
        if stem:
            boc_texts = [txtprep.stem_words(boc_text, lang) for boc_text in boc_texts]
    
    elif candidate_type == 'all':
        unigrams = [extract_candidate_words(text, lang) for text in sentences]
        if stem:
            unigrams = [txtprep.stem_words(word, lang) for word in unigrams]
        chunks = [extract_candidate_chunks(text, lang) for text in sentences]
        
        boc_texts = [i+j for i,j in zip(unigrams, chunks)]
        #boc_texts = [random.shuffle(keys) for keys in boc_texts]

    #print(boc_texts)
        
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
    
    # aggregate common keywords
    all_phrases = []
    for doc_phrases in doc_top_phrases:
        all_phrases.extend(doc_phrases)
    all_words = [w for w,_ in all_phrases]
    keyword_scores = dict.fromkeys(all_words, [])
    for w,score in all_phrases:
        keyword_scores[w] = keyword_scores[w] + [score]
    
    keyword_scores_agg = [(w, sum(scores)/len(scores)) for w,scores in keyword_scores.items()]
    doc_top_phrases = keyword_scores_agg

    return doc_top_phrases


def __score_keyphrases_by_tfidf_single(text, lang, 
                                     candidate_type='all', stem=True, top_n_phrases=10):
    
    
    sentences = tr_sentence_splitter.text_to_sentences(text)
    
    if len(sentences) > 1:
        keyphrases = __score_keyphrases_by_tfidf(sentences, lang, candidate_type, stem, top_n_phrases)
        all_keyphrases = []
        for kp in keyphrases:
            all_keyphrases.extend(kp)
        return all_keyphrases
    else:
        if lang in ["tr", "turkish"]:
            aux_sentences = ["Merhaba, nasılsınız?", "Bugün hava ne güzel..", "Buraya standard cümleler eklememiz lazım."]
        else:
            aux_sentences = ["Hello, how are you?", "The weather is so nice today..", "We need standard auxiliary words to make this corpus stronger."]
        texts = [text] + aux_sentences
    
        docs_top_phrases = __score_keyphrases_by_tfidf(texts, lang, candidate_type, stem, top_n_phrases)

        return docs_top_phrases[0]


def score_keyphrases_by_tfidf_single(text, lang, 
                                     candidate_type='all', stem=True, top_n_phrases=10, tr_alt_stemming=True):
    
    if lang in ["tr", "turkish"]:
        aux_sentences = ["Merhaba, nasılsınız?", "Bugün hava ne güzel..", "Buraya standard cümleler eklememiz lazım."]
    else:
        aux_sentences = ["Hello, how are you?", "The weather is so nice today..", "We need standard auxiliary words to make this corpus stronger."]
    texts = [text] + aux_sentences
    
    docs_top_phrases = score_keyphrases_by_tfidf(texts, lang, candidate_type, stem, top_n_phrases, tr_alt_stemming)

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
                             remove_numbers=True, deasciify=False, 
                             remove_punkt=True,
                             lowercase=True):

    
    
    TrTokenHandler()
    
    


    text = """"
        Uzay Yolu filmlerinde sık sık izleyicinin karşısına çıkan 'ışınlanma' teknolojisi, yavaş ancak emin adımlarla gelişiyor. Son olarak Çinli bilim insanları, Dünya'nın yörüngesine bir fotonu ışınlamayı başardıklarını açıkladı. Yapılan duyuruda, 500 kilometre uzaklıktaki bir kuantum uydusuna fotonun başarıyla ışınlandığı belirtildi.Işığın temel birimi olan partiküllere foton adı veriliyor. Başarıyla sonuçlanan deney ile dünya ve uzay arasındaki ilk kuantum veri ağı da kurulmuş oldu. Kuantum ışınlaması maddenin enerjiye dönüştürülerek uzay-zamanda hareket ettirilmesi olarak tanımlanıyor. Kuantum ışınlamasında foton çiftleri kullanılıyor. Foton çiftleri arasındaki mesafe ne olursa olsun, bir tanesinin gösterdiği tepki diğerini de etkiliyor. Çinli bilim insanlarının gerçekleştirdiği son ışınlama ise bugüne kadarki en uzun mesafeli kuantum ışınlaması olarak kayda geçmiş durumda.Çin, başarıyla sonuçlanan deneyin kuantum interneti için ilk adım olduğunu duyurdu. Partiküllerin fiziksel temas olmaksızın veri aktarımı gerçekleştirebildiği kuantum internetinin, hayata geçtiğinde veri aktarım hızında da devrim niteliğinde bir gelişme olacağı ddüşünülüyor. Kuantum ışınlamasının tıp alanında da kullanılabileceği teorisi üzerinde duruluyor.Gelecekte, organları oluşturan partiküllerin ışınlama yoluyla kopyalanarak veri halinde depolanması ve ihtiyaç duyulduğunda bu veri depolarındaki organ parçacıklarıyla tedavilerin yapılabileceği üzerinde duruluyor
    """
    lang = "tr"
    stem = True
    top_n_phrases = None
    tr_longer_stemming = False
    
    text = """"ABD'de yapılan bir .araştırma araştırma, yaban .arılarının (eşek arıları) düşünülenden çok daha zeki olduklarına ve mantık yürütebildiklerini ortaya koydu.Michigan Üniversitesi'nde yapılan araştırmaya göre, kağıt yaban arısı adlı eşek arısı türü geçişli çıkarsama olarak bilinen bir akıl yürütme yöntemiyle, bilinen ilişkileri, bilinmeyen ilişkilerden ayırt edebiliyor. Bir başka ifadeyle bu arılar, tıpkı insanlar gibi X, Y'den, Y de Z'den büyükse, X, Z'den büyüktür' çıkarımını yapabiliyor. Mantıksal çıkarım yapabilen ilk omurgasız hayvanlarİnsanların dışında kuş, maymun ve balık gibi omurgalı hayvanların da bu yeteneğe sahip oldukları biliniyor. Ancak araştırma ilk kez bir omurgasız hayvanın da böyle bir yeteneği olduğunu gösteriyor. Evrim biyoloğu Elizabeth Tibbetts öncülüğünde yapılan araştırma kapsamında, bir grup eşek arısına bazı eşleştirilmiş renkleri tanımaları öğretildi."
            """

    text = """.arı ABD'de yapılan bir araştırma, yaban arılarının (eşek arıları) düşünülenden çok daha zeki olduklarına ve mantık yürütebildiklerini ortaya koydu.Michigan Üniversitesi'nde yapılan araştırmaya göre, kağıt yaban .arısı adlı eşek arısı türü geçişli çıkarsama olarak bilinen bir akıl yürütme yöntemiyle, bilinen ilişkileri, bilinmeyen ilişkilerden ayırt edebiliyor. Bir başka ifadeyle bu .arılar, tıpkı insanlar gibi X, Y'den, Y de Z'den büyükse, X, Z'den büyüktür' çıkarımını yapabiliyor. Mantıksal çıkarım yapabilen ilk omurgasız hayvanlarİnsanların dışında kuş, maymun ve balık gibi omurgalı hayvanların da bu yeteneğe sahip oldukları biliniyor. Ancak araştırma ilk kez bir omurgasız hayvanın da böyle bir yeteneği olduğunu gösteriyor.arı Evrim biyoloğu Elizabeth Tibbetts öncülüğünde yapılan araştırma kapsamında, bir grup eşek arısına bazı eşleştirilmiş renkleri tanımaları öğretildi."""

    keywords3 = score_keyphrases_by_tfidf_chunk_unigram_inclusion(text, lang, stem, 
                                                                  top_n_phrases,
                                                                  tr_longer_stemming)
    print(keywords3)
    
    
    '''
    texts = ["Dünyaca ünlü pop yıldızı Madonna çocuklarına 13 yaşında cep telefonu verdikten sonra aralarındaki ilişkinin koptuğunu söyledi.Vogue Dergisi'ne konuşan Madonna Çocuklarıma 13 yaşına geldiklerinde cep telefonu vererek hata ettim. Çocuklarımla ilişkimi bitirdi. Bütünüyle değil elbette ama telefonlar hayatlarının çok çok büyük bir parçası haline geldi. Görsellere boğuldular ve kendilerini diğer insanlarla kıyaslamaya başladılar. Bu gelişimleri için çok kötü dedi.6 çocuğu olan Madonna Portekiz'in Lizbon kentinde yaşarken zamanın çoğunun çocukları okula getirip götürmek ya da küçük oğlunu futbol maçlarına taşımakla geçtiğini ve yalnız hissettiğini söyledi. Madonna evlatlık oğlu David'in telefonundan mahrum bırakıldığını ve çocukları arasında en çok David'in ona benzediğini de ekledi. En fazla ortak noktam onunla var. Beni anladığını seziyorum dedi. David Lizbon'da Benfica futbol kulübünün akademisinde yetişiyor.Buna karşın pop yıldızı 22 yaşındaki kızı Lourdes'un sosyal medya alışkanlıkları konusunda sitemkar: Dürtüleri benimki gibi değil. Sosyal medyanın ona baş belası olduğunu ve onda İnsanlar bana Madonna'nın kızı olduğum için bazı şeyler verecek hissi yarattığını hissediyorum. Ona diğer ünlü insanların çocuklarından örnekler veriyorum. Ünlü birinin kızı olabilirsin ama yapacaklarınla ciddiye alınırsın.",
" İngiltere'de, Prens Harry-Meghan Markle çifti 2 gün önce doğan erkek bebeklerinin adını kamuoyuna açıkladı. İngiltere Kraliyet Ailesi'nin son üyesinin adı Archie Harrison Mountbatten-Windsor. Bebeğin adı, çiftin Instagram hesabından kamuoyuna duyuruldu. Çiftin paylaştığı fotoğrafta, İngiltere Kraliçesi 2. Elizabeth de görülüyor. Archie Harrison Mountbatten-Windsor, İngiltere Kraliçesi 2. Elizabeth'ın 8. torun çocuğu. Prens Harry-Meghan Markle çifti bugün daha önce de erkek bebeklerini ilk kez göstermişlerdi. Meghan Markle, beyaz kundağa sarılı oğlu için Çok tatlı bir mizacı var, çok sakin. Hayal gibi dedi. Prens Harry ise gülerek Bu huyunu kimden aldı bilemiyorum dedi.Sussex Dükü Prens Harry ve eşi Düşes Meghan Markle yaklaşık 1 yıl önce evlenmişti. İngiltere Kraliyet Ailesi'nin yeni üyesi, babasının hemen arkasında tahtın yedinci varisi. Prens Harry ve Meghan Markle, iki günlük bebeklerini kameraların önüne ilk kez getirirken, anne ve baba olma deneyimlerini de gazetecilere anlattı. Markle, Sihir gibi, harika. Dünyadaki en iyi erkeklere sahibim ve gerçekten mutluyum. Ebeveyn olmak harika dedi.Bebeğin kime daha çok benzediği sorulan Prens Harry ise, Herkes bebeklerin ilk iki haftada çok değiştiklerini söylüyor. Biz de önümüzdeki ay boyunca bu değişimin nasıl olacağını izleyeceğiz. Her gün görünüşü değişiyor, kim bilir? diye konuştu. 2 günlük bebeğin prens olup olmayacağına Kraliçe karar verecek. ",
 "ABD'de yapılan bir araştırma güneş kremindeki kimyasalların yüksek oranlarda kana karıştığını ortaya koydu. Amerikan Gıda ve İlaç Dairesi'ne (FDA) bağlı İlaç Değerlendirme ve Araştırma Merkezi'nin araştırmasında güneş kremlerindeki üç kimyasalın, günlük kullanım devam ettikçe kandaki seviyesinin arttığı ve son kullanımdan sonra en az 24 saat daha kanda kaldığına da işaret ediyor. Sonuçları tıp dergisi JAMA'da yayımlanan araştırma FDA'nın üretici firmalardan inceleme yapmasını istediği 12 kimyasaldan dördünü; avobenzon, oksibenzon, ecamsule ve oktokrilen'i kapsıyor. Araştırma kapsamında 24 sağlıklı gönüllü, bir hafta boyunca güneş kremi kullandı. Günde dört kez vücutlarının yüzde 75'ine krem süren bu kişilerden belli aralıklarla kan örnekleri alındı. İlk gün sonunda ecamsule krem kullanan altı kişiden beşinin kanında istatistiksel olarak yüksek oranda bu kimyasala rastlandı. Oksibenzon, avabenzon ve oktokrilen içeren kremler kullanan kişilerin tümünün kanında yüksek oranda kimyasal çıktı. Özellikle oksibenzon seviyesinin diğer kimyasallarınkinden 50-100 kat daha yüksek olduğu görüldü. Amerikan Dermatoloji Akademisi'nden Dr. David Leffell, bu kimyasalların kana karışmasının etkileri konusunda ileri çalışmalar yapılması gerektiğini belirterek Araştırmalar tamamlanıncaya kadar güneş kremi kullanımına devam edilmeli tavsiyesinde bulundu. Güneş kremleri konusunda her yıl tavsiyeler yayımlayan Çevre Çalışma Grubu Dairesi'nin Başkan Yardımcısı Scott Faber de Asıl düşman güneş. Burada haber değeri taşıyan şey, cildimize sürdüğümüz maddelerin vücudumuz tarafından emilmesi değil. Güneş kremi üreticilerinin şimdi bu kimyasalların sağlık riski oluşturup oluşturmadığını incelemesi gerekiyor dedi.Amerikan Cilt Kanseri Vakfı'na göre ülkede her yıl cilt kanseri teşhisi konan kişilerin sayısı diğer kanser türlerine yakalanan kişilerin sayısının toplamından daha fazla.Dünya Kanser Araştırmaları Vakfı'na göre, küresel çapta melanoma hem kadınlarda hem de erkeklerde 19'uncu en yaygın kanser türü.ABD Gıda ve İlaç Dairesi, başlangıçta güneş yanıklarına karşı kullanılan kremlerin reçetesiz satılmasına onay vermişti. İki tür güneş kremi bulunuyor. Bu kremlerden birinde güneş ışınlarını filtrelemek için kimyasal bileşimler, diğerinde ise titanyum diyoksit ve çinko oksit gibi, sürüldükten sonra vücutta beyaz bir tabaka bırakan mineraller kullanılıyor. Birçok kişinin vücudunda beyaz bir tabakayla dolaşmak istemediği için kimyasal içeren güneş kremlerinin popular hale geldiği belirtiliyor.",
"Birleşmiş Milletler, insanların doğa üstündeki yıkıcı etkilerini, bugüne kadarki en kapsamlı raporlardan biri ile ortaya koydu. Rapora göre, 1 milyon hayvan ve bitki türü yok olma tedidiyle karşı karşıya. Paris'ten BBC Çevre Muhabiri Matt McGrath'ın haberi. Doğa her yerde daha önce görülmemiş bir hızda kötüleşirken, gıda ve enerji ihtiyacı da bunu körüklüyor. BM raporunda bu gidişatın değiştirebileceği belirtiliyor. Ancak raporda insanların doğayla etkileşimlerinde dönüştürücü bir değişime ihtiyaç duyulduğu vurgulanıyor.Biyo çeşitlilik ve Ekosistem Servisleri üstüne Hükümetlerarası Bilim-Politika Platformu tarafından 15 bin ayrı referans materyallerinden toplanan raporun tamamı 1800 sayfa. Bugün yayımlanan 40 sayfalık özet, insanların yaşadıkları tek yerle ilgili güçlü iddialar içeriyor. Yerkürenin her zaman insanların eylemleri nedeniyle zarar gördüğü belirtilen raporda son 50 yıldaki gelişmelerin büyük izler bıraktığı kaydediliyor.1970'ten bu yana dünya nüfusu ikiye katlandı, Küresel ekonomi dört kat büyüdü, uluslararası ticaret ise 10 kat arttı. Bu kadar hızlı gelişen dünyayı beslemek, giydirmek ve enerji sağlamak için inanılmaz süratte özellikle de tropikal bölgelerde ormanlar yok edildi. 1980 ile 2000 yılları arasında Güney Amerika'da ve Güney Doğu Asya'da 100 milyon hektar orman yok oldu. Sulak alanların ise 1700'li yıllardan bu yana sadece yüzde 13'ü 2000 yılına kadar ulaşabildi. 1992 yılından bu yana şehirler büyük bir hızda genişledi ve kentlerin kapladı alan iki katına çıktı. Bu düzeyde insan faaliyeti, hayvanları ve bitkileri daha önce hiç olmadığı kadar fazla oranda öldürüyor. Rapora göre hayvan ve bitkilerin ortalama yüzde 25'i tehdit altında. Bu da bir milyona yakın türün on yıllar içinde yok olabileceği anlamına geliyor. Bu oran son 10 milyon yılda yaşanan yıkımın onlarca ya da yüzlerce katı. Raporu kalem alan Minnesota Üniversitesi'nden Dr Kate Brauman, Biyoçeşitlilik ve doğadaki eşisz yıkımı belgeledik. Bu, oran ve sürat anlamında daha önce insanlik tarihinde gördüklerimizden çok farklı diyor.",
"Hindistan'da ordunun resmi Twitter hesabından yapılan, Yeti canavarının ayak izinin bulunduğuna yönelik paylaşım, sosyal medyada geniş katılımlı bir tartışma başlattı. 6 milyon takipçisi olan silahlı kuvvetler hesabından yapılan paylaşımda fotoğrafla birlikte Makalu Üs Bölgesinde Yeti canavarına ait, gizemli ayak izleri ifadesi yer aldı. Himalayalar'da yaşadığına inanılan efsanevi canavarla ilgili hikayeler, Güney Asya toplumları arasında yaygın şekilde anlatılıyor. Hint ordusunun resmi hesabından yapılan paylaşım ise efsaneyi yeniden canlandırmış durumda. Times of India gazetesi, 9 Nisan tarihinde kaydedilen ayak izinin, ordu yetkilileri tarafından, geçmiş teorileri doğruladığı gerekçesiyle kamuoyuna duyurulduğunu bildirdi."
]
    import pandas as pd
    df = pd.read_excel('<PATH>')
    txtcol = "Original Text"
    texts = df[txtcol].tolist()
    for text in texts:
        #keywords = score_keyphrases_by_tfidf_single(text, lang, candidate_type, stem, top_n_phrases)
        keywords = score_keyphrases_by_tfidf_chunk_unigram_inclusion(text, lang, stem, top_n_phrases, tr_alt_stemming=tr_longer_stemming)
        pprint(text[:50])       
        pprint(keywords)
        pprint(keywords, open("<PATH>"+text[:15]+".txt", "w"))
        print("*************\n")
    
    #keywords1 = score_keyphrases_by_tfidf_single(text, lang, candidate_type, stem, top_n_phrases)
    '''
    
    '''
    import pandas as pd
    df = pd.read_excel('<PATH>')
    txtcol = "Original Text"
    targetcol = "keywords_improved_dicle"
    for i in df.index.values:
        text = df.loc[i, txtcol]
        #keywords = score_keyphrases_by_tfidf_single(text, lang, candidate_type, stem, top_n_phrases)
        keywords = score_keyphrases_by_tfidf_chunk_unigram_inclusion(text, lang, stem, top_n_phrases)
        df.loc[i, targetcol] = str(keywords)
        
        pprint(text[:50])       
        pprint(keywords)
        print("*************\n")

    df.to_excel(..)
    '''

    '''
    words = extract_candidate_words(text, lang)
    chunks = extract_candidate_chunks(text, lang)
    pprint(words)
    pprint(chunks)
    print(len(chunks))
    '''
    
    
    
    '''
    keywords2 = __score_keyphrases_by_tfidf_single(text, lang, candidate_type, stem, top_n_phrases)   
    pprint(keywords2)
    '''

    
