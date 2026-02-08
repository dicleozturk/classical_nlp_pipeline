'''
Created on Oct 17, 2016

@author: dicle

Adapted for python3 from http://sujitpal.blogspot.com.tr/2013/03/implementing-rake-algorithm-with-nltk.html

this implementation is faster but less accurate than the original one.
-dicle
'''


# Adapted from: github.com/aneesha/RAKE/rake.py
from __future__ import division
import operator
import nltk
import string

from language_tools import tr_stemmer


def isPunct(word):
    return len(word) == 1 and word in string.punctuation

def isNumeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except ValueError:
        return False

class RakeKeywordExtractor:

    def __init__(self, lang, stem=False):
        self.stopword = set(nltk.corpus.stopwords.words(lang))
        self.top_fraction = 1  # consider top third candidate keywords by score
        self.stem = stem

    def _generate_candidate_keywords(self, sentences):
        phrase_list = []
        for sentence in sentences:
            #words = map(lambda x: "|" if x in self.stopword else x, nltk.word_tokenize(sentence.lower()))
            if self.stem:
                words = map(lambda x: "|" if x in self.stopword else x, tr_stemmer.stem_words_in_text(sentence.lower(), tokenize=True))
            else:
                words = map(lambda x: "|" if x in self.stopword else x, nltk.tokenize.wordpunct_tokenize(sentence.lower()))
            
        phrase = []
        for word in words:
            if word == "|" or isPunct(word):
                if len(phrase) > 0:
                    phrase_list.append(phrase)
                    phrase = []
            else:
                phrase.append(word)
        return phrase_list

    def _calculate_word_scores(self, phrase_list):
        word_freq = nltk.FreqDist()
        word_degree = nltk.FreqDist()
        for phrase in phrase_list:
            degree = len(list(filter(lambda x: not isNumeric(x), phrase))) - 1
            for word in phrase:
                word_freq[word] += 1
                word_degree[word] += degree  # other words
        for word in word_freq.keys():
            word_degree[word] = word_degree[word] + word_freq[word]  # itself
            # word score = deg(w) / freq(w)
        word_scores = {}
        for word in word_freq.keys():
            word_scores[word] = word_degree[word] / word_freq[word]
        return word_scores

    def _calculate_phrase_scores(self, phrase_list, word_scores):
        phrase_scores = {}
        for phrase in phrase_list:
            phrase_score = 0
            for word in phrase:
                phrase_score += word_scores[word]
            phrase_scores[" ".join(phrase)] = phrase_score
        return phrase_scores
    
    def extract(self, text, incl_scores=False):
        sentences = nltk.sent_tokenize(text)
        phrase_list = self._generate_candidate_keywords(sentences)
        word_scores = self._calculate_word_scores(phrase_list)
        phrase_scores = self._calculate_phrase_scores(phrase_list, word_scores)
        sorted_phrase_scores = sorted(phrase_scores.items(), key=operator.itemgetter(1), reverse=True)
        n_phrases = len(sorted_phrase_scores)
        
        print("All phrases:")
        print(sorted_phrase_scores)
        print()
        if incl_scores:
            return sorted_phrase_scores[0:int(n_phrases / self.top_fraction)]
        else:
            return map(lambda x: x[0], sorted_phrase_scores[0:int(n_phrases / self.top_fraction)])

def test():
    
    text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."
    lang = "english"
    rake = RakeKeywordExtractor(lang)

    keywords = rake.extract(text, incl_scores=True)
  
    print(keywords)
  


def test_tr():
        
    text = """Danıştay İdari Dava Daireleri Kurulu, 2010 Akademik Personel ve Lisans Üstü Eğitim
     Giriş Sınavı (ALES) sonbahar dönemi kılavuzundaki başörtüsüyle sınava girilmesine olanak 
     sağlayan düzenlemelerin yürütmesinin durdurulmasına YÖK'ün yaptığı itirazı kabul etti. 
     Kurul, yasağı doğuran davayı açan sendikanın böyle bir dava açmaya ehliyeti bulunmadığına karar verdi. 
     2010 yılı ALES Sonbahar kılavuzunda, adaylardan istenecek arasında, ""bir fotoğraf"" denilmişti. 
     Kılavuzda, önceki benzer sınavlardakinin aksine ""fotoğraf"" tanımlanırken, ""başı açık fotoğraf"" 
     ifadesi yer almamıştı. Bu nedenle ALES'e başörtülü öğrenciler de girebilmişti. 
     'DAVA AÇMA EHLİYETİ YOK' Eğitim - İş Sendikası da adaylardan istenen belgeler arasındaki fotoğrafın 
     ""başı açık"" şeklinde tanımlanmamış olması nedeniyle dava açmış, 
     Yükseköğretim Kurulu (YÖK), bu işleminin başörtülü öğrencilerin sınava girmesine olanak tanıdığı 
     gerekçesiyle iptali ve yürütmesinin durdurulmasını istemişti. Danıştay 8'inci Dairesi de kılavuzdaki 
     söz konusu düzenlemelerin yürütmesini oy birliğiyle durdurmuştu. Eğitim - İş Sendikası'nın bu davayı 
     açma ehliyetinin bulunup bulunmadığını o dönemde Daire de tartışmış ve kılavuz hükümlerinin laiklik 
     ilkesini zedelediği ve yargı kararlarına aykırılık taşıdığını kabul ederek sendikanın dava açma 
     ehliyetinin bulunduğuna karar vermişti. Ardından da sendikanın talebini kabul ederek yürütmenin 
     durdurulmasına karar vermişti. ANKARA YÖK'ün itiraz gerekçesi de aynı. YÖK, Danıştay 8'inci Dairesi'nin 
     kararına itiraz etmişti. YÖK itirazında, Eğitim - İş Sendikası'nın bu konuda dava açabilmesi için dava 
     konusu yapılan idari işlemden, yani kılavuzdaki ""bir fotoğraf"" ibaresinden dolaylı veya dolaysız 
     etkilenmesi gerektiği oysa böyle bir durumun olmadığı belirtilmişti. YÖK, Eğitim - İş'in 
     ""ehliyetsizliği"" nedeniyle, 8'inci Daire'nin verdiği yürütmenin durdurulması kararının bozulmasını 
     istemişti. İtirazı görüşen Danıştay İdari Dava Daireleri Kurulu, itirazı oy çokluğuyla kabul etti. 
     Kurul, dosyanın esasına girmeden yaptığı incelemede, davacı Eğitim-İş'in dava açma ehliyeti bulunmadığı 
     kararını verdi. Genel Kurul'un kararı Daire için bağlayıcı olduğundan yürütmenin durdurulması kararını 
     kaldıracak. Daha sonra da davayı, ""davayı açanın ehliyetsizliği"" nedeniyle reddedecek."""
    lang = "turkish"
    stemming = True
    rake = RakeKeywordExtractor(lang, stemming)

    #stem
    '''
    roots = tr_stemmer.stem_words_in_text(text, tokenize=True)
    print(roots)
    text2 = " ".join(roots)
    '''
    keywords = rake.extract(text, incl_scores=True)
  
    print(keywords)





