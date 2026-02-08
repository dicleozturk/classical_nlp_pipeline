'''
Created on Nov 7, 2019

@author: dicle
'''
from dataset import io_utils
from keyword_extraction import detect_topics

    dpath = "<PATH>"
    df = io_utils.read_excel(dpath)
    texts = df["Text"].tolist()
    print(len(texts))
    
    #texts = ['Puis-je mettre mon forfait Internet résidentiel Fizz sur pause?', 'Comment puis-je mettre mon forfait Internet résidentiel Fizz sur pause?', 'Comment puis-je mettre mon forfait Internet résidentiel Fizz sur pause?', 'Comment puis-je mettre mon forfait Internet résidentiel Fizz sur pause?', 'Comment puis-je mettre mon forfait Internet résidentiel Fizz sur pause?', 'Comment puis-je mettre mon forfait Internet résidentiel Fizz sur pause?', 'Comment puis-je mettre mon forfait Internet résidentiel Fizz sur pause?']
    
    topical_words = detect_topics.detect_topics_nmf(texts, lang="fr",
                                                    N=20, stopword=True, stemming=False, remove_numbers=True, deasciify=True, remove_punkt=True, lowercase=True, wordngramrange=(1,1))
    detect_topics.detect_topic_svd(texts, [], lang="fr",
                                   N=20, stopword=True, stemming=False, remove_numbers=True, deasciify=True, remove_punkt=True, lowercase=True, wordngramrange=(1,1))
    print(topical_words)