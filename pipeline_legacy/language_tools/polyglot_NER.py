'''
Created on Oct 21, 2016

@author: dicle
'''

#from polyglot.text import Text


def get_named_entities(text, lang="tr"):
    ptext = Text(text, hint_language_code=lang)  # output can be re-organised
    sentences = ptext.sentences
    entities = []
    for sentence in sentences:
        lang_sentence = Text(str(sentence), hint_language_code=lang)
        s_entities = [(" ".join(entity), entity.tag) for entity in lang_sentence.entities]
        entities.extend(s_entities)
    
    entities = list(set(entities))
    return entities



def get_sentence_entities(sentence, lang="tr", clean_tags=True):
    
    if len(sentence.strip()) == 0:
        return []
    psent = Text(sentence, hint_language_code=lang)
    pentities = psent.entities
    entities = [(" ".join(entity), entity.tag) for entity in pentities]
    
    if clean_tags:
        entities = [(name_, tag.replace("I-", "")) for name_, tag in entities]
        
    return entities


def get_sentence_entities_biomarked(sentence, lang="tr"):
    
    if len(sentence.strip()) == 0:
        return []
    psent = Text(sentence, hint_language_code=lang)
    pentities = psent.entities
    entities = [(" ".join(entity), entity.tag) for entity in pentities]
    

    entities = [(name_, tag.replace("I-", "")) for name_, tag in entities]
        
    return entities


    text = """Can Dündar'ın babası Ali Rıza Dündar, bugün öğleden sonra vefat etti.   Dündar'ın cenazesi yarın Ankara Kocatepe Camii'nde ikindi namazının ardından kılınacak cenaze namazından sonra Gölbaşı Mezarlığı'na defnedilecek.   Can Dündar'ın babası geçtiğimiz günlerde rahatsızlanmış, hastanede tedavi gördükten sonra eve çıkmıştı.   Dündar babasının rahatsızlığını Milliyet'teki köşesine "Ben babamın beşiğini tıngır mıngır sallar iken" başlıklı bir yazıyla taşımıştı.   Yazıda duygularını şu cümlelerle anlatmıştı: "Rolü değişiyor babalarla çocukların.  .  .  Size yürümeyi öğreten adamın koluna girip yürütüyorsunuz.   Bir zamanlar sizi besleyen eline destek olup kurumuş dudaklarına su veriyor, içine ekmek doğranmış çorba içiriyorsunuz.   Tıpkı rolleri değişmeden önce onun size yaptığı gibi, geceleri sessizce baş ucuna gidip nefesini dinliyorsunuz.  " Dündar'a baş sağlığı diliyoruz.  """
    
    text = """
    Taner Tüzün Nürnberg, 26 Mayıs (DHA) Nürnberg Ditip Derneği'nin 15.   kez düzenlediği ve dört gün sürecek Kültür Şöleni büyük ilgi görüyor.   Üç ayrı büyük çadırın haricinde dışarıda kurulun stantlar dolup taştı.   Şölende bir konuşma yapan, Nürnberg Belediye Başkan Yardımcısı Christian Vogel, Dokunmazlıklar ve terörle mücadeleyi kastederek, Ankara'daki gelişmelerden endişe duyuyoruz dedi.  TÜRKİYE BİR HUKUK DEVLETİDİRBaşkonsolos Yavuz Kül, birlik ve beraberliğin önemine dikkat çektikten sonra, Vogel'in sözlerine, Türkiye'de terörle mücadele de alınan tüm önlemler Anayasal çerçevede yapılmaktadır.   Milletvekillerin dokunmazlıkların kaldırılması da aynı ölçüde değerlendirilmelidir.   Alınan tüm önlemler, hukuk çerçevesindedir diye cevap verdi.  AÇILIŞ İSTİKLAL MARŞININ OKUNMASI İLE BAŞLADIKültür şöleninin açılışı İstiklal marşımızın ve Kur'an-ı Kerim'in okunması ile başladı.   Serdar Türen'in takdim ettiği programda, Ditib Nürnberg Başkanı HASAN Aslan bir konuşma yaparak, etkinliklerin birlik ve beraberliğe katkı sağladığını söyledi.  Bavyera Milletvekili Arif Taşdelen, Ditip Kuzey Bavyera Birliği Başkanı Erhan Çınar, Nürnberg Göç ve Uyum Konseyi Başkanı İlhan Postaloğlu da birer konuşma yaptılar.   Postaloğlu, Madem bizi AB'ye almıyorsunuz, içişlerimize de karışmayın dedi.  KONSERLER VERİLDİNürnberg Ditib' in kültür şöleninde, Samsun Radyosu korosu, Orhan Hakalmaz, Mustafa Cihat gibi isimler konser verdiler.   Nürnberg Ditib Kültür şöleninde, Başkonsolos Yavuz Kül, Belediye ikinci Başkanı Christian vogel, Alman siyesiler, Çalışma Ataşesi Faruk Küçük, THY Nürnberg Müdürü Osman Nuri Hasırcı, MUSİAD Nürnberg Başkanı Kadir Bozkurt, UETD Kuzey Bavyera Başkanı Yılmaz Deliduman, Ditib Kuzey Bavyera Birlik Başkanı Erhan Çınar , Nürnberg Belediyesi Meclis üyesi YASEMİN YILMAZ, Dergahspor Başkanı İbrahim Akbulut, Fürth Ditib Başkanı Refet Avcı gibi bir çok isim hazır bulundu.  
    
        """
    
    text = """ 26. 04. 2017 Tarihinde işlem yapılmıştır ancak temliğe işlenmemiştir. İsim Soyisim: Ufuk Tuncer Cihaz Modeli: Türk telekom mobil wifi modem Cihaz imei:<PHONE> Telf no: <PHONE> """
    entities = get_named_entities(text)
    print(entities)
    
    # check pos tags
    
    
    
    
    
    
    