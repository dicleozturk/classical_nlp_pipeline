'''
Created on Jun 6, 2017

@author: dicle
'''


import sys
sys.path.append("..")




#####  for decomposer models   #####################
def print_topic_words(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names()
    topics_keywords = _get_top_words(model, feature_names, n_top_words)
    for topic, keywords in topics_keywords.items():
        print(topic, " : ", ", ".join(keywords))


def get_topic_words(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names()
    topics_keywords = _get_top_words(model, feature_names, n_top_words)
    return topics_keywords



def get_topic_words_with_weights(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names()
    
    topics_keywords = {}
    for topic_idx, topicvect in enumerate(model.components_):
        indices = topicvect.argsort()[:-n_top_words - 1:-1]
        weights = sorted(topicvect)[:-n_top_words - 1:-1]
        topical_terms_weights = []
        for i, weight in zip(indices, weights):
            topical_terms_weights.append((feature_names[i], weight))
        
        topics_keywords[topic_idx] = topical_terms_weights
        
        
    return topics_keywords



def _get_top_words(model, feature_names, n_top_words):
    topics_keywords = {}
    for topic_idx, topic in enumerate(model.components_):
        topical_terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]       
        topics_keywords[topic_idx] = topical_terms
    return topics_keywords

###########################################################



def get_top_N_words_with_weights(clusterer, vectorizer, nclusters, top_N_words):
    
    clwords = dict.fromkeys(list(range(nclusters)), [])
    order_centroids = clusterer.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    #print(clusterer.cluster_centers_)
    #print(clusterer.cluster_centers_.shape)
    for i in range(nclusters):
        
        words = [(terms[ind], clusterer.cluster_centers_[i, ind]) 
                        for ind in order_centroids[i, : top_N_words]]
        clwords[i] = words
    
    return clwords










