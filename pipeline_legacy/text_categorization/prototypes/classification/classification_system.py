'''
Created on May 11, 2017

@author: dicle
'''

import sys
sys.path.append("..")



import os
from abc import ABC, abstractmethod
import random

import time
import joblib
import sklearn.model_selection as cv
import sklearn.metrics as mtr

from text_categorization.utils import tc_utils
from dataset import corpus_io, io_utils
from text_categorization.prototypes.classification import CLSF_CONSTANTS 


class ClassificationSystem(ABC):
    
    clsf_task = None   # _ClassificationTask()
    folder_ = ""
    
    
    def __init__(self, Task,
                       task_name,
                       config_obj,
                       folder_=None):
        
        self.clsf_task = Task(feature_config=config_obj,
                              classifier=config_obj.classifier,   # here or outside??
                              task_name=task_name
                              ) 
        self.folder_ = folder_
    
    '''
    def read_dataset(self,   train_data_folder,
                             train_data_fname,
                             text_col,
                             cat_col,
                             csvsep,
                             shuffle_dataset):
        texts, labels = corpus_io.read_dataset_csv(train_data_folder, train_data_fname, 
                                          csvsep, text_col, cat_col, shuffle_dataset) 
        return texts, labels
    '''
    
    
    def get_task(self):
        return self.clsf_task
    
    
    def get_cross_validated_clsf_performance(self, instances, labels, nfolds=3):
        '''
        returns acc, fscore, duration as the results of nfolds-fold cross-validated classification.
        '''
        #print("----", self.clsf_task.txt_column_name) 
        accuracy, fscore, duration, clsf_report = self.clsf_task.cross_validated_classify(instances, labels, nfolds)
        return accuracy, fscore, duration, clsf_report


    def get_cross_validated_clsf_performance3(self, instances, labels, nfolds=3):
        '''
        returns acc, fscore, duration as the results of nfolds-fold cross-validated classification.
        '''
        #print("----", self.clsf_task.txt_column_name) 
        ypred = self.clsf_task.cross_validated_classify3(instances, labels, nfolds)
        ypred = list(ypred)
        accuracy = mtr.accuracy_score(labels, ypred)
        fscore = mtr.f1_score(labels, ypred, average="macro", pos_label=None)
        result_list = mtr.precision_recall_fscore_support(labels, ypred)
        clsf_report = mtr.classification_report(labels, ypred)
        confusion_matrix = mtr.confusion_matrix(labels, ypred)
        
        print("actual: ", labels)
        print("predicted: ", ypred)
        
        return accuracy, fscore, result_list, clsf_report, confusion_matrix  
    
    def get_cross_validated_clsf_probabilities(self, instances, labels, nfolds=3):
        '''
        returns acc, fscore, duration as the results of nfolds-fold cross-validated classification.
        '''
        #print("----", self.clsf_task.txt_column_name) 
        
        ypred, yscore = self.clsf_task.cross_validated_classify_probabilities(instances, labels, nfolds)
        
        return ypred, yscore
    
    
    
    '''
      ytrue: true labels, of size n_instances
      ypred: predicted labels, of size n_instances
      yscore: prediction scores, of size n_instances X n_labels
    '''
    def get_cross_validated_clsf_performance_detailed(self, ytrue, ypred, yscore,
                                                      modelmainfolder):
        '''
        finds accuracy, fscore and other evaluations metrics values as well as
        roc curves and average precision scores. records the values in a dict
        and the figures in the disc.
        '''
        print("Computing performance evaluation metrics.")
        results_dict = tc_utils.get_performance_results(ytrue, ypred, yscore, modelmainfolder)
        resultspath = os.path.join(modelmainfolder, "results.dict")
        from pprint import pprint
        pprint(results_dict, open(resultspath, "w"))
        print(" Scores recorded in ", modelmainfolder)
        
        # write objects
        objectspath = io_utils.ensure_dir(os.path.join(modelmainfolder, "objects"))
        pprint(ytrue, open(os.path.join(objectspath, "ytrue.object"), "w"))
        pprint(ypred, open(os.path.join(objectspath, "ypred.object"), "w"))
        pprint(yscore, open(os.path.join(objectspath, "yscore.object"), "w"))
        
        return results_dict
    
    
    
    
    def get_random_baseline_performance(self, instances, labels):
        
        labelset = list(set(labels))
        random_labels = [random.choice(labelset) for _ in range(len(labels))]
    
        return self.get_cross_validated_clsf_performance(instances, random_labels)
    
    
    def get_majority_vote_baseline_performance(self, instances, labels):
        
        from collections import Counter
        c = Counter(labels)
        most_common_label = c.most_common(1)[0][0]
        majority_voted_labels = [most_common_label]*len(labels)

        acc, fscore = tc_utils.get_performance(labels, majority_voted_labels)
        return acc, fscore
    
    
    def single_train_performance(self, texts, labels, shuffle=False):
    
        train_texts, test_texts, train_labels, test_labels = cv.train_test_split(texts, labels, 
                                                                                 test_size=0.3,
                                                                                 shuffle=shuffle)
        
        model, _ = self.train_and_save_model2(train_texts, train_labels)
        
        predicted_labels, _ = self.predict_online(model, test_texts)
        acc, fscore = tc_utils.get_performance(test_labels, predicted_labels, verbose=False)
        
        
        print("ntrain: ", len(train_texts), " - ntest: ", len(test_texts))
        print("Performance: ", acc, fscore)
        
        return acc, fscore
    
    
    def build_system(self, 
                     Task,
                     task_name,
                     config_obj,
                     #classifier,
                     train_data_folder,
                     train_data_fname,
                     text_col,
                     cat_col,
                     csvsep,
                     shuffle_dataset,
                     cross_val_performance,
                     modelfolder,
                     modelname,
                     N=None):
         
        clsf_task = Task(feature_config=config_obj,
                                      classifier=config_obj.classifier,   # here or outside??
                                      task_name=task_name
                                      )
        texts, labels = corpus_io.read_dataset_csv(train_data_folder, train_data_fname, 
                                          csvsep, text_col, cat_col, shuffle_dataset)  # make this a member

        if N:
            texts = texts[:N]
            labels = labels[:N]
            modelname = modelname + "_" + str(N)
         
        model, modelpath = self.run_classification_system(clsf_task, texts, labels, modelfolder, modelname, cross_val_performance)
        
        return model, modelpath 
    
    def _dump_classification_system(self, model,
                                 task_obj,
                                 picklefolder,
                                 modelname):
    
    
        recordfolder = io_utils.ensure_dir(os.path.join(picklefolder, modelname))
        
        modelpath = os.path.join(recordfolder, CLSF_CONSTANTS.MODEL_FILE_NAME)
        classifierpath = os.path.join(recordfolder, CLSF_CONSTANTS.CLASSIFIER_FILE_NAME)
    
        joblib.dump(model, modelpath)
        joblib.dump(task_obj, classifierpath)
        
        return recordfolder
        
    def _load_classification_system(self, picklefolder):
        
    
        
        
        modelpath = os.path.join(picklefolder, CLSF_CONSTANTS.MODEL_FILE_NAME)
        classifierpath = os.path.join(picklefolder, CLSF_CONSTANTS.CLASSIFIER_FILE_NAME)
        
        model = joblib.load(modelpath)
        task_obj = joblib.load(classifierpath)
        
        return model, task_obj

    def train_and_save_model(self, texts, labels, 
                              picklefolder,
                              modelname):
        
        learning_model = self.clsf_task.train(texts, labels)
        
        print("Training finished.")
        
        modelname_unique = modelname + "_" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        modelfolder = self._dump_classification_system(learning_model, self.clsf_task, 
                                                       picklefolder, modelname_unique)
        print("Model written on the disc.")
        #clsf_system.cross_validated_classify(texts, labels)
    
        return learning_model, modelfolder
    
    
    def train_and_save_model2(self, texts, labels, 
                              picklefolder=None,
                              modelname=None):
        
        learning_model = self.clsf_task.train(texts, labels)
        
        print("Training finished.")
        
        
        modelfolder = ""
        if picklefolder and modelname:       
            modelfolder = self._dump_classification_system(learning_model, self.clsf_task, picklefolder, modelname)
            print("Model written on the disc.")
    
        return learning_model, modelfolder
    
    
    def predict_offline(self, modelfolder, test_instances):
        
        model, task_obj = self._load_classification_system(modelfolder)
        
        predicted_labels, prediction_map = task_obj.predict(model, test_instances)
        return predicted_labels, prediction_map
    
    
    def predict_online2(self, model, task_obj, test_instances):
        
        return task_obj.predict(model, test_instances)
    
    
    def predict_online(self, model, test_instances):
        
        return self.clsf_task.predict(model, test_instances)
    
    
    def run_classification_system(self,  clsf_task,
                                     texts,
                                     labels,
                                     modelfolder,
                                     modelname,
                                     cross_val_performance=True):
            
        if cross_val_performance:
            clsf_task.cross_validated_classify(texts, labels)
            #clsf_task.measure_performance(texts, labels)
            
        model, modelpath = self.train_and_save_model(texts, labels, clsf_task, modelfolder, modelname)
        
        return model, modelpath
    
    
    print(ClassificationSystem())
    
    
        
    