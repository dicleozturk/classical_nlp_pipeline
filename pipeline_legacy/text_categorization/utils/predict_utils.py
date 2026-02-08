'''
Created on Jun 16, 2017

@author: dicle
'''

import os
import pandas as pd


import joblib
import sklearn.model_selection as cv

from dataset import io_utils
from text_categorization.prototypes.classification.classification_system import ClassificationSystem
from text_categorization.utils import tc_utils
from text_categorization.prototypes.classification import CLSF_CONSTANTS



# offline prediction


def _dump_classification_system(model,
                                 task_obj,
                                 picklefolder,
                                 modelname):
    
    
    recordfolder = io_utils.ensure_dir(os.path.join(picklefolder, modelname))
    
    modelpath = os.path.join(recordfolder, CLSF_CONSTANTS.MODEL_FILE_NAME)
    classifierpath = os.path.join(recordfolder, CLSF_CONSTANTS.CLASSIFIER_FILE_NAME)

    joblib.dump(model, modelpath)
    joblib.dump(task_obj, classifierpath)
    
    return recordfolder
    
    
    
def _load_classification_system( picklefolder):

    modelpath = os.path.join(picklefolder, CLSF_CONSTANTS.MODEL_FILE_NAME)
    classifierpath = os.path.join(picklefolder, CLSF_CONSTANTS.CLASSIFIER_FILE_NAME)
    
    model = joblib.load(modelpath)
    task_obj = joblib.load(classifierpath)
    
    return model, task_obj


def predict_offline(modelfolder, test_instances):
        
    model, task_obj = _load_classification_system(modelfolder)
    
    predicted_labels, prediction_map = task_obj.predict(model, test_instances)
    return predicted_labels, prediction_map
    

def measure_test_performance(model, task_obj, test_instances, test_labels):
    
    predicted_labels, prediction_map = task_obj.predict(model, test_instances)
    
    acc, fscore = tc_utils.get_performance(test_labels, predicted_labels, verbose=False)
    return acc, fscore



def single_train_performance(task_obj, texts, labels):
    
    train_texts, test_texts, train_labels, test_labels = cv.train_test_split(texts, labels, test_size=0.3)
    
    model, _ = task_obj.train_and_save_model2(train_texts, train_labels)
    
    predicted_labels, _ = task_obj.predict_online(model, test_texts)
    acc, fscore = tc_utils.get_performance(test_labels, predicted_labels, verbose=False)
    
    
    print("ntrain: ", len(train_texts), " - ntest: ", len(test_texts))
    print("Performance: ", acc, fscore)
    
    return acc, fscore



def predict_df(modelfolder,
               testdf,
               text_col,
               truecat_col):
    
    
    test_instances = testdf[text_col].tolist()
    test_labels = None
    try:
        test_labels = testdf[truecat_col].tolist()
    except:
        pass
    
    predicted_labels, prediction_map = predict_offline(modelfolder, test_instances)
    
    predcat_col = "predicted_label"
    predmap_col = "prediction_probabilities"
    testdf[predcat_col] = predicted_labels
    testdf[predmap_col] = prediction_map
    
    if test_labels:
        test_acc, test_fscore = tc_utils.get_performance(test_labels, predicted_labels, verbose=True)
    
        print("test scores: ", test_acc, test_fscore)
    
    return testdf, prediction_map


'''
def predict_instances(modelfolder, 
                      test_instances,
                      test_labels=None):
    
    
    predicted_labels, prediction_map = predict_offline(modelfolder, test_instances)
    
    
    if test_labels:
        test_acc, test_fscore = tc_utils.get_performance(test_labels, predicted_labels, verbose=True)
    
        print("test scores: ", test_acc, test_fscore)
    
    return predicted_labels, prediction_map
'''


