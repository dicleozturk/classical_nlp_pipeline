'''
Created on May 28, 2019

@author: dicle
'''


import nltk.tokenize as nltktokenizer
from language_tools import str_utils


def token_splitter(text, lang="en", remove_punkt=False):
    
    if lang in ["tr", "turkish"]:
        tokens = nltktokenizer.word_tokenize(text, language="turkish")
        
   
    elif lang in ["ar", "arabic", "arab"]:
        tokens = nltktokenizer.wordpunct_tokenize(text)
    
    elif lang in ["fr", "french"]:
        tokens = nltktokenizer.word_tokenize(text, language="french")
    
    else:        
        tokens = nltktokenizer.word_tokenize(text, language="english")
    
    if remove_punkt:
        tokens = [token for token in tokens if not str_utils.is_punctuation(token)]

    return tokens

    txt = """ABD'de yapılan bir araştırma, yaban arılarının (eşek arıları) düşünülenden çok daha zeki olduklarına ve mantık yürütebildiklerini ortaya koydu.Michigan Üniversitesi'nde yapılan araştırmaya göre, kağıt yaban arısı adlı eşek arısı türü geçişli çıkarsama olarak bilinen bir akıl yürütme yöntemiyle, bilinen ilişkileri, bilinmeyen ilişkilerden ayırt edebiliyor. Bir başka ifadeyle bu arılar, tıpkı insanlar gibi X, Y'den, Y de Z'den büyükse, X, Z'den büyüktür' çıkarımını yapabiliyor. Mantıksal çıkarım yapabilen ilk omurgasız hayvanlarİnsanların dışında kuş, maymun ve balık gibi omurgalı hayvanların da bu yeteneğe sahip oldukları biliniyor. Ancak araştırma ilk kez bir omurgasız hayvanın da böyle bir yeteneği olduğunu gösteriyor. Evrim biyoloğu Elizabeth Tibbetts öncülüğünde yapılan araştırma kapsamında, bir grup eşek arısına bazı eşleştirilmiş renkleri tanımaları öğretildi.
    """
    
    #txt = '>لم أعرف أن هذا الفندق يطل على مقبر المدينة .. كنت مرهقاَ بعد رحلة طويلة شاقة ، و ليس لي إلا أن أنام . فتحت الستائر و انكشفت لي قبور متجاورة . قبور تتجمع و تتلقي ، تخرج من مكانها و تتزحزح و ترتج كالماء ، و تطفو للأعلى ، جامحة تضرب بعضها بعضاً ، و الشواهد تنهار بلونها الرمادي و الأبيض أما القبب الصغيرة فكانت ملفوفة بخرق خضراء . لم أخبر زوجتي بالأمر ، كانت مشغولة بالاستحمام بعد السفر الطويل . أسمع رشرشات الماء على البلاط و أرى الأبجورات الموضوعة بعناية .'
    txt = "Ali,Veli geliyor mu?Elmalar,armutlar kızarmış mı?"
    print(token_splitter(txt, lang="tr"))
    
    
    
    