'''
Created on May 11, 2017

@author: dicle
'''
import sys
sys.path.append("..")


import sklearn.naive_bayes as nb
import sklearn.linear_model as sklinear


import text_categorization.prototypes.classification.classification_system as clsf_sys
import text_categorization.prototypes.classification.prep_config as prepconfig
import text_categorization.prototypes.tasks.general_text_classifier as gen_txt
from dataset import corpus_io

tr_spam_detect_config_obj = prepconfig.FeatureChoice(lang="tr", weights=dict(text_based=1,
                                                                        token_based=1),
                                              stopword=True, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=False,
                                              remove_numbers=False, deasciify=True, remove_punkt=True, lowercase=True,
                                              wordngramrange=(1,2), charngramrange=None,  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42),
                                              )


    
    
