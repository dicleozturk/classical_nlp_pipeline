'''
author: esra
Created on Jul 13, 2017

'''





from __future__ import unicode_literals


import sys
sys.path.append("..")


from nltk.data import load as nltk_loader
import re

def preprocess(text):
    #corect wrong sentence spaces.    i.e.    "Ali geldi.Gunaydin."->"ali geldi. Gunaydin."
    text = re.sub(r'\.(\w{2,})', r'. \1', text, flags = re.UNICODE)
    #To avoid sentence breaks after quantifiers. will be reverted.
    #i.e.    "II. Dunya Savasi"->"II.Dunya Savasi"    "2. Dunya Savasi"->"2.Dunya Savasi"
    #!!May cause errors if the text is math or science based. 
    text = re.sub(r'([^a-zA-Z\d]|^)(\d+|[VIX]{1,5})\. ([A-Za-z]+)', r'\1\2.\3', text, flags=re.IGNORECASE)
   
    #handles quotations and avoids sentence break inside one or recursive quotations.
    text = re.sub(r"""(^|[^a-zA-Z])(['"`])((.|\n)+?)(\2(.|\n|$))""", lambda m: 
                  m.group(1)+m.group(2)+re.sub(r"( |\n)" , r'#%%#', m.group(3), flags=re.IGNORECASE)+re.sub(r"""(["'`])([^.?!])""",r"""_\1_\2""",m.group(5), flags=re.IGNORECASE)
                  , text, flags=re.UNICODE)
    return text



def postprocess(sents):
    #corect wrong sentence spaces.    i.e.    "Ali geldi.Gunaydin."->"ali geldi. Gunaydin." 
    sents = list(map(lambda x: re.sub(r"""([A-Za-z'"`\d]+)\.([A-Za-z'"`]+)""", r'\1. \2', x, flags=re.UNICODE), sents))
    #II.Dunya Savasi"->"II. Dunya Savasi
    #revert quantifier modifications
    sents = list(map(lambda x: re.sub(r'([^a-zA-Z\d])(\d+|[VIX]{1,5})\.([A-Za-z]+)', r'\1\2. \3', x, flags=re.UNICODE), sents))
    #reverts quotation handling steps
    sents =list(map(lambda x: re.sub(r"""_([`"'])_""", r'\1', x, flags=re.UNICODE), sents))
    sents = list(map(lambda x : re.sub(r"#%%#",r' ', x, flags=re.UNICODE), sents))
    
    #split "..." sentences
    for s in sents:
        while re.search(r"\.\. \w", s, flags=re.UNICODE):
            ind = sents.index(s)
            s1 = s[:(s.index(".. ")+2)]
            s = s[(s.index(".. ")+3):]
            sents.insert(ind+1,s)
            sents[ind]=s1
    
    sents = list(map(lambda x : x.strip(), sents))
    return sents


def tokenize(text):
    tr_tokenizer = nltk_loader('tokenizers/punkt/turkish.pickle')
    return tr_tokenizer.tokenize(text)


def _text_to_sentences(text):
    text = preprocess(text)
    sents = tokenize(text)
    sents = postprocess(sents)
    return sents


def debugResult():
    text=open("input.txt",'r').read().decode('utf-8');
    sents =text_to_sentences(text)
    for i in sents:
        print(i)
        
def text_to_sentences(text):
    #text = re.sub(r'\.([A-Za-z\)\]]{2,})', r'. \1', text, flags = re.UNICODE)
    #text = re.sub(r'\.(\D+)', r'. \1', text, flags=re.UNICODE)
    text = re.sub(r'([A-Za-zÖöĞğŞşİıÜüÇç]+[\.\!\?]{1,4})([A-Za-zÖöĞğŞşİıÜüÇç0-9]+)', r'\1 \2', text, flags=re.UNICODE)          
    #text = re.sub(r'(\d+[\.\!\?]{1,4})([A-Za-zÖöĞğŞşİıÜüÇç]+)', r'\1 \2', text, flags=re.UNICODE)
    text = re.sub(r'([^a-zA-Z\d]|^)(\d+|[VIX]{1,5})\. ([A-Za-z]+)', r'\1\2.\3', text, flags=re.IGNORECASE)
   
    #text = re.sub(r'(\D+)\.(\d+)', r'\1. \2', text, flags = re.UNICODE)    
    print("\n", text)
    sents = tokenize(text)
    sents = list(map(lambda x: re.sub(r'([^a-zA-Z\d])(\d+|[VIX]{1,5})\.([A-Za-z]+)', r'\1\2. \3', x, flags=re.UNICODE), sents))
    
    sents = [s.strip() for s in sents]
    return sents



    text = """
    İçtüzük değişikliği AKP'yi karıştırdı


AKP'li Yalçınbayır, partisinin yaptığı içtüzük değişikliğini hukuka ve demokrasiye aykırı buldu

     ABDULLAH KARAKUŞ Ankara

     CHP'nin şiddetle karşı çıktığı TBMM içtüzük değişikliği, AKP'yi karıştırdı. eski Başbakan Yardımcısı ve AKP Bursa Milletvekili Ertuğrul Yalçınbayır, partisinin yaptığı içtüzük değişikliğinin hukuka ve demokrasiye aykırı olduğunu söyledi. Yalçınbayır, "Bir içtüzük değişikliğinde bu yapılırsa önümüzdeki günlerde demokrasiyle bağdaşmayan birçok şeyle karşılaşabileceğiz demektir" diye konuştu. Hükümetin kanun yaparken daha ciddi davranması gerektiğini kaydeden Yalçınbayır, partisini şu sözlerle eleştirdi:
     ÇELİŞKİLİ DAVRANMAYA HAKLARI YOK: İçtüzük değişikliği hukuka, demokrasiye aykırı. Anayasa Mahkemesi'nin iptal edeceği bir içtüzük. Geçen dönemde bizim karşı çıktığımız düzenlemeden daha da kötü. Muhalefetteyken bu içtüzük değişikliğine 50 milletvekilimiz karşı çıkmıştı. Bu 50 milletvekili o imzalarını hatırlamalı. İçtüzükle muhalefetin söz hakkı sınırlandırılıyor. 
     ANAYASA DEĞİŞİKLİĞİ DE YANLIŞTI: Programda katılımcılık ve uzlaşma olacak diyoruz. Katılımcılık olmadan anayasa değişikliği yapılmaz. Ulusal uzlaşma olması gerekirdi. Getirilen anayasa değişikliğinde bunu yapamadık, yanlış oldu.
     AKP'liler, DSP - ANAP - MHP koalisyonunca TBMM içtüzüğünde yapılan aynı yöndeki değişikliğin iptali için Anayasa Mahkemesi'ne dava açmış, mahkeme de 31 Ocak 2002'de değişikliği iptal etmişti.
    
    """
    
    text = """
    Devlet Bakanı Mehmet Keçeciler’in "PKK’yı sandıkta yeneriz" açıklaması, seçim bölgesi ve memleketi Konya’da tepkiyle karşılandı. Şehit Aileleri Derneği Konya Şube Başkanı Ali Dönmez, "Memleketimizin evladı olmasına rağmen bundan sonra o adamla ne görüşür, ne de konuşuruz. Bu saatten sonra artık Apo’yu assalar da olur, asmasalar da. Siyasilerden artık hiçbir şey beklemiyoruz" dedi. SP Konya Milletvekili Veysel Candan da, Keçeciler’in sözlerini "Yanlış bir çağrı" diye değerlendirdi.
    """
    
    text = """Irak'ta terör tehdidi ham petrolü 45 dolara çıkardı

Irak'ta petrol pompalanmasının durdurulmasıyla ABD ham petrolünün varili 44.97 dolara yükseldi. 

Uluslararası ham petrol fiyatları, kısa bir düşüşün ardından yeniden 44 doların üstüne çıkarken, Irak'ın güneyinde ülke petrolünün yüzde 90'ını üreten Basra terminaline kuyulardan petrol pompalanması "terör tehdidi'' yüzünden durduruldu. ABD'nin Teksas petrolünün varil fiyatı, varilde 44.97 dolara yükselirken, Irak Petrol Bakanlığı, ''Şii önder Mukteda Essadr'ın tehdidi yüzünden üretimin durduruluğunu, ancak iki gün yetecek stokların hazır olduğu'' bildirildi. Hazır depolama tanklarından petrol sevkıyatının Basra'da sürdüğü bildirildi. ABD'de petrol cuma günü 1983'ten bu yana en yüksek değer olan 44.77 dolardan işlem görmüştü. Rusya'da ise en büyük petrol üreticisi Yukos şirtine bağlı kurum olan ve Yukos üretiminin yüzde 60'ını karşılayan ''Yuganskneftegaz''ın hisse senetleri, icra memurlarınca donduruldu. Mahkeme, bu kararaYukos'un 3 milyar 400 milyon dolarlık borçlarını ödeyememesini gerekçe görteriyor. Cuma günü ABD ham petrolü varil başına 44.77 dolar ile 21 yılın zirvesini görmüştü. Gözler Petrol İhraç Eden Ülkeler Örgütü'ne (OPEC) çevrilirken, Londra'da işlem gören Brent ham petrolü ise varil başına 41.5 dolara yükseldi. Başta ABD ve Çin olmak üzere, dünyadaki petrol talebinin artmasıyla arzın, olası herhangi bir kesintiyi karşılayamayacak kadar sınırlarına yaklaşmış olmasının yarattığı endişeler, petrolün yılın bu dönemine kadar yüzde 30'dan fazla değer kazanmasına yol açtı."""


    
    text = """ 2 Temmuz! İnsanları yaktılar.200.000...Kimsenin burnu kanamadı.Güllerle karşılamışlar gelenleri,başka herkes susmuş!200.000 insan!
    6. kattayım.
    """
    
    text = """Şişli’de, 7 katlı boş bir binanın en son katında henüz belirlenemeyen bir nedenle yangın çıktı. Hurda toplayan bir kişi 6. katta mahsur kalınca, itfaiye ekipleri tarafından merdiven yardımıyla kurtarıldı.Yangın, saat 17.30 sıralarında Halaskargazi Caddesi 196 numaradaki binada meydana geldi. Bir süre önce boşaltıldığı öğrenilen binanın son katında henüz belirlenemeyen bir nedenle yangın çıktı. O sırada hurda topladığı iddia edilen bir kişi duman yüzünden binada mahsur kaldı. Binadaki dumanı gören çevredeki vatandaşlar durumu itfaiye ekiplerine bildirdi. Olay yerine gelen itfaiye ekipleri bir yandan yangına müdahale ederken, bir yandan da merdiven yardımıyla 6. katta mahsur kalan kişiyi kurtardı. Hayati tehlikesi bulunmadığı belirtilen bu kişi Şişli Hamidiye Etfal Eğitim ve Araştırma Hastanesi’ne kaldırıldı. İtfaiye ekipleri yangını kısa süre içinde söndürürken, olayla ilgili soruşturma başlatıldı.
    """
    
    text = """"ABD'de yapılan bir araştırma, yaban arılarının (eşek arıları) düşünülenden çok daha zeki olduklarına ve mantık yürütebildiklerini ortaya koydu.Michigan Üniversitesi'nde yapılan araştırmaya göre, kağıt yaban arısı adlı eşek arısı türü geçişli çıkarsama olarak bilinen bir akıl yürütme yöntemiyle, bilinen ilişkileri, bilinmeyen ilişkilerden ayırt edebiliyor. Bir başka ifadeyle bu arılar, tıpkı insanlar gibi X, Y'den, Y de Z'den büyükse, X, Z'den büyüktür' çıkarımını yapabiliyor. Mantıksal çıkarım yapabilen ilk omurgasız hayvanlarİnsanların dışında kuş, maymun ve balık gibi omurgalı hayvanların da bu yeteneğe sahip oldukları biliniyor. Ancak araştırma ilk kez bir omurgasız hayvanın da böyle bir yeteneği olduğunu gösteriyor. Evrim biyoloğu Elizabeth Tibbetts öncülüğünde yapılan araştırma kapsamında, bir grup eşek arısına bazı eşleştirilmiş renkleri tanımaları öğretildi."
            """
    #text = "Hello!My name is Dicle. I'm coming.200.000 propil."
    from pprint import pprint
    sents = text_to_sentences(text)
    pprint(sents)
    print(len(sents))
    
    for i,s in enumerate(sents):
        print(i, "  - ", s)
    

    