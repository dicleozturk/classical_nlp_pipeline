'''
Created on May 23, 2019

@author: dicle
'''


from language_tools import tr_postag_syntaxnet
from language_tools import tr_sentence_splitter
import tr_chunker.TR_chunker.chunker_main2 as tr_chunk
import keyword_extraction.tfidf_keyword_extraction.baseline_chunks_as_keyphrases as bl
from keyword_extraction.tfidf_chunk_keyword_extraction.keyword_extractor import extract_candidate_words

    from pprint import pprint
    
    text = "ABD'de yapılan bir araştırma, yaban arılarının (eşek arıları) düşünülenden çok daha zeki olduklarına ve mantık yürütebildiklerini ortaya koydu.Michigan Üniversitesi'nde yapılan araştırmaya göre, kağıt yaban arısı adlı eşek arısı türü geçişli çıkarsama olarak bilinen bir akıl yürütme yöntemiyle, bilinen ilişkileri, bilinmeyen ilişkilerden ayırt edebiliyor. Bir başka ifadeyle bu arılar, tıpkı insanlar gibi X, Y'den, Y de Z'den büyükse, X, Z'den büyüktür' çıkarımını yapabiliyor. Mantıksal çıkarım yapabilen ilk omurgasız hayvanlarİnsanların dışında kuş, maymun ve balık gibi omurgalı hayvanların da bu yeteneğe sahip oldukları biliniyor. Ancak araştırma ilk kez bir omurgasız hayvanın da böyle bir yeteneği olduğunu gösteriyor. Evrim biyoloğu Elizabeth Tibbetts öncülüğünde yapılan araştırma kapsamında, bir grup eşek arısına bazı eşleştirilmiş renkleri tanımaları öğretildi."

    '''
    sents = tr_sentence_splitter.text_to_sentences(text)
    
    all_chunks = tr_chunk.chunk_text(text)
    all_chunks2 = bl.extract_candidate_chunks(text, lang="tr")

    all_chunks_s = []
    for i,s in enumerate(sents):
        
        s_chunks = tr_chunk.chunk_text(s)
        all_chunks_s.extend(s_chunks)
        pprint((i,s, s_chunks))
    all_chunks_s = list(set(all_chunks_s))
    print(len(all_chunks), len(all_chunks_s), len(all_chunks2))
    '''
    text="araştırma, çalışma, inceleme konuları çok önemli!"
    print(extract_candidate_words(text, lang="tr"))
    print(tr_postag_syntaxnet.postag_text(text))
    print(tr_chunk.chunk_text(text))
    
    