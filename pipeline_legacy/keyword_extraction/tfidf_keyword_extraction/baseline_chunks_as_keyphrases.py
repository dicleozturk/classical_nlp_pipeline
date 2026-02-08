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



    from pprint import pprint
    
    text = "Hello, there is a green apple on the blue table which is so nice.."
    
    print(extract_candidate_chunks(text, lang="en"))
    
    tr_text = """"Mardin'in Savur ilçesinde oturan Nur Sema Demir, 31 Temmuz'da açıklanan üniversite sınavında, sayısal bölüm sorularından 253, sözelden 278 ve temel yeterlilik testinde 324 puan aldı. Aldığı puana göre tercih yapan Demir, 1 ay sonra açıklanan yerleştirme sonucunda sayısal puanının, '0' olduğunu ve 4 yıllık bir fakülte beklerken, 18'inci tercihi olan 2 yıllık Moda Tasarımı Bölümü'ne yerleştirildiğini öğrenince şok yaşadı. Demir ailesi, ilk sınav sonucunun yerleştirme sonucundan çok farklı olduğununu, 4 yıllık bir okula yerleştirilmesini sağlayan sayısal puanın ikinci sonuç belgesinde sıfır gösterildiğini ve sonuç belgesindeki doğru ve yanlış soru sayılarının farklı olduğunu görünce ÖSYM'ye itirazda bulundu. ÖSYM tarafından Nur Sema Demir'e gönderien yanıtta, itiraz için geç kaldığı belirtildi. Demir ailesi, yapılan yanlış nedeniyle kızlarının mağdur edildiğini söyledi."
    """
    
    tr_text = "ABD'de yapılan bir araştırma, yaban arılarının (eşek arıları) düşünülenden çok daha zeki olduklarına ve mantık yürütebildiklerini ortaya koydu."
    
    tr_text = """ABD'de yapılan bir araştırma, yaban arılarının (eşek arıları) düşünülenden çok daha zeki olduklarına ve mantık yürütebildiklerini ortaya koydu.Michigan Üniversitesi'nde yapılan araştırmaya göre, kağıt yaban arısı adlı eşek arısı türü geçişli çıkarsama olarak bilinen bir akıl yürütme yöntemiyle, bilinen ilişkileri, bilinmeyen ilişkilerden ayırt edebiliyor. Bir başka ifadeyle bu arılar, tıpkı insanlar gibi X, Y'den, Y de Z'den büyükse, X, Z'den büyüktür' çıkarımını yapabiliyor. Mantıksal çıkarım yapabilen ilk omurgasız hayvanlarİnsanların dışında kuş, maymun ve balık gibi omurgalı hayvanların da bu yeteneğe sahip oldukları biliniyor. Ancak araştırma ilk kez bir omurgasız hayvanın da böyle bir yeteneği olduğunu gösteriyor. Evrim biyoloğu Elizabeth Tibbetts öncülüğünde yapılan araştırma kapsamında, bir grup eşek arısına bazı eşleştirilmiş renkleri tanımaları öğretildi."""
    
    tr_keywords = extract_candidate_chunks(tr_text, lang="tr")
    pprint(tr_keywords)
    
    
    