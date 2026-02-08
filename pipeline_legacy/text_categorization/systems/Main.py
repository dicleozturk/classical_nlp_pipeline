'''
Created on May 12, 2017

@author: dicle
'''

import sys
sys.path.append("..")

import os 


from text_categorization.utils import tc_utils


def main(clsf_system, 
         texts, labels,
         modelrootfolder, modelname,
         test_instances, test_labels):
    
 
    nfolds = 3
    # 2- APPLY CROSS-VALIDATED CLASSIFICATION and GET PERFORMANCE
    accuracy, fscore, duration, clsf_report = clsf_system.get_cross_validated_clsf_performance(texts, labels, nfolds=nfolds)
    print(accuracy, fscore, duration, clsf_report)
    
    # 3- TRAIN THE MODEL WITH THE ABOVE PARAMETERS; SAVE IT ON THE FOLDER modelrootfolder/modelname
    model, modelfolder = clsf_system.train_and_save_model(texts, labels, modelrootfolder, modelname)
    
    # 3.a- record performance scores
    score_report = str(nfolds) + "- fold cross validated learning results:"
    score_report += "\nduration (sec.): " + str(duration) + "\n" + clsf_report
    open(os.path.join(modelfolder, "score_report.txt"), "w").write(score_report)

    #modelfolder = io_utils.ensure_dir(os.path.join(modelrootfolder, modelname))
    #modelfolder = os.path.join(modelrootfolder, modelname)
    
    # 4.a- PREDICT ONLINE (the model is in the memory)
    predicted_labels, prediction_map = clsf_system.predict_online(model, test_instances)
    
    # 4.b- PREDICT OFFLINE (the model is loaded from the disc)
    #predicted_labels, prediction_map = clsf_system.predict_offline(modelfolder, test_instances)
    
    print("predicted_labels:", predicted_labels)
    print("prediction_map:", prediction_map)
    
    print("******")
    print("text\ttrue\tpred")
    for text,true,pred in zip(test_instances, test_labels, predicted_labels):
        print(text, "\t", true, "\t", pred)
    print("*******")
    
    # 4.c- GET PREDICTION PERFORMANCE IF TRUE LABELS ARE AVAILABLE
    if test_labels:
        test_acc, test_fscore, _ = tc_utils.get_performance(test_labels, predicted_labels, verbose=True)
    
        print("test scores: ", test_acc, test_fscore)
    # -- if paramaters are modified (tr_sent_config_obj), create a new ClassificationSystem() and run necessary methods.
    
    