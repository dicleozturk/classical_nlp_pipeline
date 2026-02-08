'''
Created on Apr 10, 2018

@author: dicle
'''

import sklearn.linear_model as sklinear


import text_categorization.prototypes.classification.classification_system as clsf_sys
import text_categorization.prototypes.classification.prep_config as prepconfig
import text_categorization.prototypes.tasks.general_text_classifier as gen_txt
from dataset import corpus_io
from text_categorization.utils import tc_utils



gen_txt_clsf_config_obj = prepconfig.FeatureChoice(lang="fr", weights=dict(text_based=1,
                                                                        token_based=1),
                                              stopword=True, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=False,
                                              remove_numbers=False, deasciify=False, remove_punkt=True, lowercase=True,
                                              wordngramrange=(1,2), charngramrange=None,  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42),
                                              )





def main(clsf_system, 
         modelrootfolder, modelname,
         test_instances, test_labels):
    
 
        
    # 2- APPLY CROSS-VALIDATED CLASSIFICATION and GET PERFORMANCE
    accuracy, fscore, duration = clsf_system.get_cross_validated_clsf_performance(texts, labels, nfolds=3)
    
    
    # 3- TRAIN THE MODEL WITH THE ABOVE PARAMETERS; SAVE IT ON THE FOLDER modelrootfolder/modelname
    model, modelfolder = clsf_system.train_and_save_model(texts, labels, modelrootfolder, modelname)
    '''
    import os
    modelfolder = os.path.join(modelrootfolder, modelname)
    '''
    
    # 4.a- PREDICT ONLINE (the model is in the memory)
    predicted_labels, prediction_map = clsf_system.predict_online(model, test_instances)
    
    # 4.b- PREDICT OFFLINE (the model is loaded from the disc)
    predicted_labels, prediction_map = clsf_system.predict_offline(modelfolder, test_instances)
    
    print(prediction_map)
    
    # 4.c- GET PREDICTION PERFORMANCE IF TRUE LABELS ARE AVAILABLE
    if test_labels:
        test_acc, test_fscore = tc_utils.get_performance(test_labels, predicted_labels, verbose=True)
    
    # -- if paramaters are modified (tr_sent_config_obj), create a new ClassificationSystem() and run necessary methods.
    
    
    