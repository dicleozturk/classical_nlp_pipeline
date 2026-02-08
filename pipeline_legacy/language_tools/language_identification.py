'''
Created on Jun 19, 2017

@author: dicle
'''



import sys
sys.path.append("..")



from langid.langid import LanguageIdentifier, model



def detect_language(text):
    
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    lang, prob = identifier.classify(text)
    
    return lang, prob




def is_in_lang(text, inputlang):
    
    lang, _ = detect_language(text)
    return lang == inputlang


    text = "ismin nedir? merhaba nasılsın"
    print(detect_language(text))
    
    print()
    
    
    
    