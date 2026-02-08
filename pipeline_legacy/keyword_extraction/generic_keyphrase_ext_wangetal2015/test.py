'''
Created on Nov 2, 2018

@author: dicle
'''



from language_tools.tokenizers import token_splitter, sentence_splitter




    lang = "tr"
    corpus = ["Başkent Diyarbakır'ın güzel duvarlarının Suriçi Caddesi'nin içinde yürüyeceğiz.."]
    all_words = []
    doc_base = []   # [ doc_id, [sentence_id : [word_i]]]
    for i,doc in enumerate(corpus):

        doc_tokens = token_splitter.text_to_words(doc, lang=lang, remove_punkt=True)
        all_words.extend(doc_tokens)
        
        doc_sentences = sentence_splitter.text_to_sentences(doc, lang=lang)
        doc_base_sentences = []
        for sent_id,sentence in enumerate(doc_sentences):
            sent_tokens = token_splitter.text_to_words(sentence, lang=lang, remove_punkt=True)
            doc_base_sentences.append((sent_id, sent_tokens))
        doc_base.append((i, doc_base_sentences))
    
    
    print(doc_base)
    
    
        
    print()