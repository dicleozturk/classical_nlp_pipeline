'''
Created on May 28, 2019

@author: dicle
'''

from time import time

from learning.language_tools.token_splitting import token_splitter
from language_tools import tr_stemmer





def stemmer_measure_time(tokens):
    
    stems1 = []
    stems2 = []
    
    t0 = time()
    for t in tokens:
        stem = tr_stemmer.stem(t)
        stems1.append(stem)
    
    t1 = time()
    
    stems2 = tr_stemmer.stem_words(tokens)
    t2 = time()
    
    print("one by one stemming", round(t1-t0, 3))
    print("bulk stemming", round(t2-t1, 3))


    print()
    
    t = """ABD'de yapılan bir araştırma, yaban arılarının (eşek arıları) düşünülenden çok daha zeki olduklarına ve mantık yürütebildiklerini ortaya koydu.Michigan Üniversitesi'nde yapılan araştırmaya göre, kağıt yaban arısı adlı eşek arısı türü geçişli çıkarsama olarak bilinen bir akıl yürütme yöntemiyle, bilinen ilişkileri, bilinmeyen ilişkilerden ayırt edebiliyor. Bir başka ifadeyle bu arılar, tıpkı insanlar gibi X, Y'den, Y de Z'den büyükse, X, Z'den büyüktür' çıkarımını yapabiliyor. Mantıksal çıkarım yapabilen ilk omurgasız hayvanlarİnsanların dışında kuş, maymun ve balık gibi omurgalı hayvanların da bu yeteneğe sahip oldukları biliniyor. Ancak araştırma ilk kez bir omurgasız hayvanın da böyle bir yeteneği olduğunu gösteriyor. Evrim biyoloğu Elizabeth Tibbetts öncülüğünde yapılan araştırma kapsamında, bir grup eşek arısına bazı eşleştirilmiş renkleri tanımaları öğretildi.
    """
    
    tokens = token_splitter(t, lang="tr")
    print(len(tokens))
    
    x = "Önce"
    y = "önce".capitalize()
    print(x == y)
    