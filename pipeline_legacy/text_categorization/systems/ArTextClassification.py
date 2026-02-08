'''
Created on Jul 5, 2019

@author: dicle
'''


import sklearn.linear_model as sklinear


import text_categorization.prototypes.tasks.general_text_classifier as txt_classifier
import text_categorization.prototypes.classification.classification_system as clsf_sys
import text_categorization.prototypes.classification.prep_config as prepconfig
from dataset import corpus_io, corpus_analysis

ar_txt_clf_config_obj = prepconfig.FeatureChoice(lang="ar", weights={"word_tfidf" : 1,
                                                                   "polyglot_value" : 0,
                                                                   "polyglot_count" : 0,
                                                                   "lexicon_count" : 0,
                                                                   "char_tfidf" : 1}, 
                                              stopword=True, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=False,
                                              remove_numbers=True, deasciify=False, remove_punkt=True, lowercase=False,
                                              wordngramrange=(1,2), charngramrange=(2,2),  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              #classifier=nb.MultinomialNB()
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=42)
                                              )
    
    