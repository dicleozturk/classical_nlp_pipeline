'''
Created on Jul 2, 2017

@author: dicle
'''


import sys
from dataset import io_utils
sys.path.append("..")


import os

import nltk.tokenize as nltktokenizer
#import nltk.data
from nltk.data import load as nltk_loader


from language_tools import tr_sentence_splitter


'''
takes a text and returns [[t,]] list of sentences split to its tokens.
'''
def get_tokenized_sentences3(content):
        
    #print("\nTokenizing the text into sentences and tokens.")
    
    #tr_tokenizer = nltk.data.load('tokenizers/punkt/turkish.pickle')
    tr_tokenizer = nltk_loader('tokenizers/punkt/turkish.pickle')
    sentences = tr_tokenizer.tokenize(content)
    sentences = [s.strip() for s in sentences]
    
    tsentences = []
    for sentence in sentences:
        tokens = nltktokenizer.word_tokenize(sentence.encode("utf-8").decode("utf-8"), language="turkish")
        tokens = [t.strip() for t in tokens]
        tsentences.append(tokens)
    
    return tsentences



def get_tokenized_sentences2(text):
    
    
    sentences = tr_sentence_splitter.text_to_sentences(text)
    
    tsentences = []
    for sentence in sentences:
        tokens = nltktokenizer.word_tokenize(sentence.encode("utf-8").decode("utf-8"), language="turkish")
        tokens = [t.strip() for t in tokens]
        tsentences.append(tokens)
    
    return tsentences



def get_tokenized_sentences(text):
    
    
    sentences = tr_sentence_splitter.text_to_sentences(text)
    sentences = [s.strip() for s in sentences]
    
    tsentences = []
    for sentence in sentences:
        tokens = nltktokenizer.word_tokenize(sentence.encode("utf-8").decode("utf-8"), language="turkish")
        tokens = [t.strip() for t in tokens]
        tsentences.append(tokens)
    
    return tsentences





    text = """
    İçtüzük değişikliği AKP'yi karıştırdı


AKP'li Yalçınbayır, partisinin yaptığı içtüzük değişikliğini hukuka ve demokrasiye aykırı buldu

     ABDULLAH KARAKUŞ Ankara

     CHP'nin şiddetle karşı çıktığı TBMM içtüzük değişikliği, AKP'yi karıştırdı. eski Başbakan Yardımcısı ve AKP Bursa Milletvekili Ertuğrul Yalçınbayır, partisinin yaptığı içtüzük değişikliğinin hukuka ve demokrasiye aykırı olduğunu söyledi. Yalçınbayır, "Bir içtüzük değişikliğinde bu yapılırsa önümüzdeki günlerde demokrasiyle bağdaşmayan birçok şeyle karşılaşabileceğiz demektir" diye konuştu. Hükümetin kanun yaparken daha ciddi davranması gerektiğini kaydeden Yalçınbayır, partisini şu sözlerle eleştirdi:
     ÇELİŞKİLİ DAVRANMAYA HAKLARI YOK: İçtüzük değişikliği hukuka, demokrasiye aykırı. Anayasa Mahkemesi'nin iptal edeceği bir içtüzük. Geçen dönemde bizim karşı çıktığımız düzenlemeden daha da kötü. Muhalefetteyken bu içtüzük değişikliğine 50 milletvekilimiz karşı çıkmıştı. Bu 50 milletvekili o imzalarını hatırlamalı. İçtüzükle muhalefetin söz hakkı sınırlandırılıyor. 
     ANAYASA DEĞİŞİKLİĞİ DE YANLIŞTI: Programda katılımcılık ve uzlaşma olacak diyoruz. Katılımcılık olmadan anayasa değişikliği yapılmaz. Ulusal uzlaşma olması gerekirdi. Getirilen anayasa değişikliğinde bunu yapamadık, yanlış oldu.
     AKP'liler, DSP - ANAP - MHP koalisyonunca TBMM içtüzüğünde yapılan aynı yöndeki değişikliğin iptali için Anayasa Mahkemesi'ne dava açmış, mahkeme de 31 Ocak 2002'de değişikliği iptal etmişti.
    
    """
    
    text = """ 2 Temmuz! İnsanları yaktılar.Kimsenin burnu kanamadı.Güllerle kaşılamışlar gelenleri,başka herkes susmuş!
    
    """
    
    text = "Irak'ta petrol pompalanmasının durdurulmasıyla ABD ham petrolünün varili 44.97 dolara yükseldi.Sonuçta isyan etmeliyiz. "
    
    text = "sjhdsj <NUM> *NUM* @<USER> <@USER> <URL> *URL* *<USER>*"
    from pprint import pprint
    sents = get_tokenized_sentences(text)
    pprint(sents)
    print(len(sents))
    
    
    
    
    '''
    txtfolder1 = "<PATH>"
    txtfolder2 = "<PATH>"
    
    
    fnames1 = io_utils.getfilenames_of_dir(txtfolder1, removeextension=False)
    
    for fname in fnames1:
        
        p1 = os.path.join(txtfolder1, fname)
        p2 = os.path.join(txtfolder2, fname)
        text = open(p1, "r").read()
        text = text.strip()
        tokenized_sentences = get_tokenized_sentences2(text)
        sentencelist2 = [" ".join(tokens) for tokens in tokenized_sentences]
        sentencelist2 = [s.strip() for s in sentencelist2]
        
        io_utils.write_lines(sentencelist2, p2, linesep="\n")
        
    
    '''
    
    '''
    text = """
        Aynen şöyle : “II. Abdülhamid’den Mustafa Kemal’e Devlet ve Millet”. Kitapta simgelere önem vermedeki devamlılık anlatıldığı için alt başlık böyle. REAYA’DAN VATANDAŞ’A Tarihsel olarak “nüfus” unsuru üç aşamadan geçerek evrildi : - Reaya dönemi : Köylü tarlasındadır , devletle pek işi yoktur. Tuğralar fermanlarda yer alır ; her yerde görülmez.- 
        Teba dönemi : Köylü şehirle ve devletle ilişkiye geçmiştir ; sürtüşmeler de başlamıştır. 
        Bizde ve Avrupa’da “hükümdar” figürü ve alametleri etrafında bir “aidiyet” duygusu ve “sadık teba” kültürü yaratılmaya çalışılır. - Vatandaş dönemi : Ahali artık aktiftir. Tanzimat sürecinde , 1869’da Vatandaşlık Kanunu çıkar. 
        Tuğra’dan ayrı olarak Abdülmecid “Devlet-i Osmani arması”nı çizdirir.
    
            """
    text = "Ben ona 'Nereden geliyorsun?' dedim ve gittim."
    print(get_tokenized_sentences(text))
    print()
    '''
    