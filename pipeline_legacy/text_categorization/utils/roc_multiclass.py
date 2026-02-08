'''
Created on Sep 5, 2019

@author: dicle
'''


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import random

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as mtr


def binarize_labels(ytrue):
    classes_ = sorted(list(set(ytrue)))
    
    if len(classes_) < 3:
        bin_labels1 =  label_binarize(ytrue, classes=classes_)
        bin_labels = np.zeros((len(ytrue), 2), dtype=int)
        for i,label in enumerate(bin_labels1):
            bin_labels[i, label] = 1   # assuming label_binarize applies 0-1 encoding
        
    else:
        bin_labels =  label_binarize(ytrue, classes=classes_)
    bin_labels = np.array(bin_labels, dtype=int)
    return bin_labels

'''
 y values (both ytrue and yscore) are label binarized; i.e., these parameters are
 matrices of size n_instances X n_classes. 
 ytrue[i,j]=1 if ith instance is in class j; and the rest of the columns for the ith row
 is 0.
 yscore[i,j] = max(y[i,:]) if the ith instance is predicted to be in class j.
 for all i, y[i,:] are real numbers.  
 obtaining such y values requires using OneVsRestClassifier upon the chosen learning algorithm
 for the model.
'''
def find_roc_values(ytrue, yscore):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    
    n_classes = ytrue.shape[1]
        
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(ytrue[:, i], yscore[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(ytrue.ravel(), yscore.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    return fpr, tpr, thresholds, roc_auc


def find_average_precision_values(ytrue, yscore, avg):
    
    avg_pr_dict = dict()
    n_classes = ytrue.shape[1]
    
    for i in range(n_classes):
        avg_pr_dict[i] = mtr.average_precision_score(ytrue[:, i], yscore[:, i], average=avg)
    
    avg_pr_dict["micro"] = mtr.average_precision_score(ytrue.ravel(), yscore.ravel(), average=avg)
    return avg_pr_dict



''' gives the same result with auc(fpr, tpr) calculated with the results of roc_curve 
'''
def find_roc_auc_scores(ytrue, yscore, avg):
    
    roc_auc_dict = dict()
    n_classes = ytrue.shape[1]
    
    for i in range(n_classes):
        roc_auc_dict[i] = mtr.roc_auc_score(ytrue[:, i], yscore[:, i], average=avg)
    
    roc_auc_dict["micro"] = mtr.roc_auc_score(ytrue.ravel(), yscore.ravel(), average=avg)
    return roc_auc_dict


def find_precision_recall_curves(ytrue, yscore):
    pr_rec_curve_dict = dict()
    n_classes = ytrue.shape[1] 
    
    for i in range(n_classes):
        pr_rec_curve_dict[i] = mtr.precision_recall_curve(ytrue[:, i], yscore[:, i])
    
    pr_rec_curve_dict["micro"] = mtr.precision_recall_curve(ytrue.ravel(), yscore.ravel())
    return pr_rec_curve_dict
    
    
    
# valindex is either class index to draw the roc curve for the specific class
#  or micro to draw the roc curve (micro-avg'd) for the model
def draw_roc_curve(fpr, tpr, roc_auc, valindex, figpath=None):

    plt.figure()
    lw = 2
    plt.plot(fpr[valindex], tpr[valindex], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[valindex])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristics Curve')
    plt.legend(loc="lower right")
    if figpath:
        plt.savefig(figpath)
    #plt.show()



# valindex is either class index to draw the curve for the specific class
#  or micro to draw the curve (micro-avg'd) for the model
def draw_precision_recall_curve(precisions, recalls, figpath=None):

    plt.figure()
    lw = 2
    plt.plot(precisions, recalls, color='darkorange',
             lw=lw, label='Precision - Recall curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision values over thresholds')
    plt.ylabel('Recall values over thresholds')
    plt.title('Precision - Recall Curve')
    plt.legend(loc="lower right")
    if figpath:
        plt.savefig(figpath)
    #plt.show()

