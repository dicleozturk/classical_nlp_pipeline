'''
Created on May 30, 2017

@author: dicle
'''


import sys
sys.path.append("..")

import re
import json
import requests

import string
from language_tools import stopword_lists
from language_tools import tr_sentence_splitter


UNIVERSAL_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
'''
# http://universaldependencies.org/u/pos/
    ADJ: adjective
    ADP: adposition
    ADV: adverb
    AUX: auxiliary
    CCONJ: coordinating conjunction
    DET: determiner
    INTJ: interjection
    NOUN: noun
    NUM: numeral
    PART: particle
    PRON: pronoun
    PROPN: proper noun
    PUNCT: punctuation
    SCONJ: subordinating conjunction
    SYM: symbol
    VERB: verb
    X: other
'''

# paramtext is a list of sentences.
def run_syntaxnet(paramtext):
    
    paramtext = [sentence+" " for sentence in paramtext if not sentence.endswith(" ")]
        
    requests.get('http://52.215.84.142:9000/api/v1/use/Turkish')
      
    post_data = {
                    'strings': paramtext,  #[paramtext],
                    'tree': False
                }
      
      
    headers = {'content-type': 'application/json'}
    response = requests.post('http://52.215.84.142:9000/api/v1/query', data=json.dumps(post_data), headers=headers)
    
    resp = response.json()
    
    return resp
    


def postag_text(text):
    
    sentences = tr_sentence_splitter.text_to_sentences(text)
    
    return postag_sentences(sentences)
    


def postag_sentences(sentences):

    #sentences = [s.strip()+" " for s in sentences_]

    sentence_results = []


    output_key = "output"
    word_key = "word"
    postag_key = "pos_tag"
    fpos_key = "fPOS"
    parse_results = run_syntaxnet(sentences)
    #print(parse_results)
    
    for sentence, parsing in zip(sentences, parse_results):
        
        
        result_pairs = []
        
        output = []
        try:
            output = parsing[output_key]
        except KeyError:
            print("Error: KeyError")
            return output
    
        '''
        print("nparse: ", len(output))
        print(output)
        '''
        
        #words = [i["word"] for i in output]
        #print("w: ", words)
      
        
        for parse in output:
            token = parse[word_key]
            postag = parse[postag_key]
            if postag not in UNIVERSAL_TAGS:
                fpos = parse[fpos_key]
                maintag = fpos.split("++")
                postag = maintag[0]
            postag = postag.upper()
            
            if postag == "VERB":
                try:
                    if parse["VerbForm"] == "Part":
                        postag = "ADJ"
                except:
                    pass
            
            # label mismarked gerunds as NOUN
            
            if postag == "VERB" and (re.search(r"m[ae]k?$", token)):
                postag = "NOUN"
                
            result_pairs.append((token, postag))
        
        sentence_results.append(result_pairs)
    
    
    
    
    return sentence_results



def chunking(text, lang):
    import nltk
    import itertools
    #grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'

    #grammar = r'KT: {(<ADJ>* <NOUN.*>+ <(ADP|SCONJ)>)? <ADJ>* <NOUN.*>+}'
    grammar = r'KT: {(<ADJ>* <(NOUN|PROPN)>+ <(ADP|SCONJ)>)? <ADJ>* <(NOUN|PROPN)>+}'
    punct = set(string.punctuation)
    stop_words = set(stopword_lists.get_stopwords(lang))
    
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = postag_text(text)
    #print("nsents: %d" % len(tagged_sents))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))

    print(tagged_sents)
    print()
    print(all_chunks)
    
    candidates = [' '.join(word for word, pos, chunk in group).lower() for key, group in itertools.groupby(all_chunks, lambda x : x[2] != 'O') if key]

    #print("ncandidates: %d" % len(candidates))

    candidates = [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]
    #print("ncandidates: %d" % len(candidates))
    
    print(candidates)
    
    #return candidates

    '''
    result = postag_text("kastederek")
    print(result)
    
    
    result = run_syntaxnet(paramtext=["lambaları yak. "])
    from pprint import pprint
    pprint(result)
    
    
    sentences = ["KPSS Öğretmenlik Alan Bilgisi Sınavı'nın 16 Temmuz tarihinde yapılacak olması, adayların aklına 'KPSS ÖABT ertelenecek mi?' sorusunu getirdi. ",
          "Bilindiği üzere 15 Temmuz tarihinde gün boyunca 15 Temmuz şehitlerini anma programlarını yapılacak ve gece darbe nöbeti tutulacak. ",
          "Peki KPSS ÖABT ertelenecek mi? KPSS Öğretmenlik Alan Bilgisi Sınavı ertelendi mi?",
          "İşte detaylar... ",
          "KPSS ÖĞRETMENLİK ALAN BİLGİSİ SINAVI (ÖABT) ERTELENECEK Mİ?",
          "13 Mart 2017 tarihinde başvuruları alınan ve 6 Temmuz 2017 tarihinde de sınav giriş belgesi yayımlanan KPSS Öğretmenlik Alan Bilgisi Sınavı'nın yapılma tarihi de 16 Temmuz olarak ÖSYM takviminde yer alıyor. ",
          "Ancak 15 Temmuz günü, 15 Temmuz 2016 tarihinde FETÖ tarafından yapılan başarısız askeri darbe girişiminde şehit olan vatandaşlar için anma programları ve demokrasi kutlamaları yapılacak. ",
          "Aynı zamanda gece de darbe nöbeti tutulacak. ",
          "Vatandaşlar, 15 Temmuz etkinliklerine katıldıktan sonra ve gece de darbe nöbeti tuttuktan sonra ertesi sabah sınava katılım sağlayacak. ",
          "Bu durumla ilgili sınava geç kalma veya sınav saatinde uyanamama gibi problemlerin doğabileceğini belirten adaylar, sınavın bir gün ertelenmesini sosyal medya üzerinden talep ediyor. ",
          "ÖSYM'ni ve ÖSYM Başkanı Ömer Demir'i mesaj yağmuruna tutan adaylar, bir yandan da 'KPSS Öğretmenlik Alan Bilgisi Sınavı ertelenecek mi?' diye heyecanla ve merakla bekliyor. ",
          "KPSS ÖABT ertelenmesiyle ilgili resmi makamlardan henüz bir açıklama yapılmadı. ",
          "Olası bir resmi açıklamada dahilinde bu haberi takip edebilirsiniz. ",
          "KPSS Öğretmenlik Alan Bilgisi Sınavı ertelenirse, bu haber de güncellenecektir. ",
          "KPSS ÖABT SINAV GİRİŞ BELGESİ NASIL ALINIR?",
          "Ölçme, Seçme ve Yerleştirme Merkezi tarafından 16 Temmuz 2017 tarihinde yapılması planlanan KPSS Öğretmenlik Alan Bilgisi Sınavı için sınav giriş yerlerini gösteren sınav giriş belgesi, 6 Temmuz'da adayların erişimine açılmıştı. ",
          "Sınav giriş belgesi almak isteyen adaylar, ÖSYM'nin Aday İşlemleri Sistemi olan 'ais.osym.gov.tr' adresine T.C. Kimlik Numarası ve Aday Şifresi ile giriş yapacaklar. ",
          "Ardından sol menüde yer alan Başvurularım sekmesinden KPSS ÖABT sınav giriş belgesi çıkarabilecekler. ",
          "SINAVDA UYULMASI GEREKEN KURALLAR",
          "Sınav süresince adayların;",
          "konuşmaları, kopya çekmeleri veya kopya vermeleri,",
          "salondaki görevlilere soru sormaları,",
          "kopya çekme veya çekilmesine yardım etme girişiminde bulunmaları,",
          "müsvedde kâğıdı kullanmaları veya soru ve cevaplarını başka bir kâğıda yazmaları,",
          "soru ve cevapları cevap kâğıdının arkasına yazmaları,",
          "birbirlerinden kalem, silgi vb. şeyler alıp vermeleri,",
          "sınavın cevaplama süresi başlamadan soruları okumaya başlamaları,",
          "sınav süresi bittiği halde soruları okumaya ve cevap kâğıdında/sınav kitapçığında işaretleme yapmaya devam etmeleri,",
          "sınav düzenini bozacak davranışlarda bulunmaları yasaktır. ",
          "Bu yasaklara uymadığı saptanan adayların sınavları geçersiz sayılacaktır."          
          ] 
    
    sentences = ["Ali, en iyi arkadaşım, nihayet geldi.", "Arabadaki adam yürümek istiyordu."]
    results = run_syntaxnet(sentences)
    pprint(results)
    '''
    
    from pprint import pprint
    
    text = "Ne kadar güzel bir gün! Bizi kısıtlayan her şeyle savaşırız! Gören göz, ilgiyle bakan el iyi arkadaştır. Bunları görmeyip kör kalan bir çuval kirazı çiğner."
    text = "ABD Jeolojik Araştırma Merkezi göre, İran'dan sonra önce Endonezya'da 5,2 ve 5 büyüklüklerinde iki, daha sonra Papua Yeni Gine'de 6,6 büyüklüğünde deprem meydana geldi."
    sents = tr_sentence_splitter.text_to_sentences(text)
    print(sents)
    results = postag_sentences(sents)
    pprint(results)
    
    
    #["ÖSYM tarafından Nur Sema Demir'e gönderilen yanıtta, itiraz için geç kaldığı belirtildi."]))

    pprint(postag_sentences(["ABD'de yapılan bir araştırma, yaban arılarının (eşek arıları) düşünülenden çok daha zeki olduklarına ve mantık yürütebildiklerini ortaya koydu."]))

    
  
    ''' 
    text = "A clause is a finite verb and all its arguments, i.e. a main verb and everything that depends on it. If you have a sentence with a single main verb, the entire sentence is one clause. Conjunctions and relative pronouns typically introduce new clauses."
    #chunking(text)
    tr_text = "Başta inşaat olmak üzere birçok sektörün canlanacağı, nakliye masraflarının düşeceği, İzmirli işadamlarının daha sık İstanbul ve Avrupa'ya ulaşacağı, kentin daha fazla yatırım alacağı, daha çok turist çekeceği belirtildi. Türkiye'nin en büyük gıda ihracatının gerçekleştirildiği Ege Bölgesi ihracatına büyük katkı sağlayacak otoyolun, zamanla yarışılan gıda ürünleri ihracatçılarına büyük avantajlar sağlayacağı, otoyolun sadece iki şehri değil, aynı zamanda Ege ve Marmara Ekonomisini birbirine bağlayacağı kaydedildi."

    print("en:")
    chunking(text, "en")

    print("tr:")
    chunking(tr_text, "tr")
    '''

    '''
    print(postag_text(tr_text))
    
    ss = tk.tr_sentence_tokenizer(tr_text)
    print(ss)

    x = postag_sentences(ss)
    print(x)
    '''