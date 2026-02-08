'''
Created on May 30, 2017

@author: dicle
'''

import sys
sys.path.append("..")


import re
import nltk

def tr_sentence_tokenizer(text):
    
    tr_tokenizer = nltk.data.load('tokenizers/punkt/turkish.pickle')
    sentences = tr_tokenizer.tokenize(text)
    sentences = [s.strip() for s in sentences]
    return sentences



# get rid of this chain fix, make it better!

def fix_sentence_ending(text):
    
    text_ = text.replace(".", ". ").replace("?", "? ").replace("!", "! ")
    text_ = re.sub(r"(\.\s){2,}", "... ", text_)
    return text_
    

    tr_sentence_tokenizer("a")
    print()
    
    text = """ Can Dündar'ın babası Ali Rıza Dündar, bugün öğleden sonra vefat etti.Dündar'ın cenazesi yarın Ankara Kocatepe Camii'nde ikindi namazının ardından kılınacak cenaze namazından sonra Gölbaşı Mezarlığı'na defnedilecek...Can Dündar'ın babası geçtiğimiz günlerde rahatsızlanmış, hastanede tedavi gördükten sonra eve çıkmıştı.   Dündar babasının rahatsızlığını Milliyet'teki köşesine "Ben babamın beşiğini tıngır mıngır sallar iken" başlıklı bir yazıyla taşımıştı.   Yazıda duygularını şu cümlelerle anlatmıştı: "Rolü değişiyor babalarla çocukların.  .  .  Size yürümeyi öğreten adamın koluna girip yürütüyorsunuz.   Bir zamanlar sizi besleyen eline destek olup kurumuş dudaklarına su veriyor, içine ekmek doğranmış çorba içiriyorsunuz..Tıpkı rolleri değişmeden önce onun size yaptığı gibi, geceleri sessizce baş ucuna gidip nefesini dinliyorsunuz.  " Dündar'a baş sağlığı diliyoruz.  
  """
    print(fix_sentence_ending(text))
    
    
    
    