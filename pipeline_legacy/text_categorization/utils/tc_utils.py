'''
Created on Nov 2, 2016

@author: dicle
'''

import numpy as np
import os
import sklearn.metrics as mtr
from text_categorization.utils import roc_multiclass


def get_performance(ytrue, ypred, verbose=True):
    
    acc = mtr.accuracy_score(ytrue, ypred)
    # fscore = mtr.f1_score(ytrue, ypred, average="weighted", pos_label=None) #, labels=list(set(ytrue)))
    fscore = mtr.f1_score(ytrue, ypred, average="macro", pos_label=None)  # , labels=list(set(ytrue)))

    # fscore=0
    
    confmatrix = mtr.confusion_matrix(ytrue, ypred)
    report = mtr.classification_report(ytrue, ypred)

    outstr = ""
    outstr += "\n" + "accuracy: " + str(acc)
    outstr += "\n" + "f1 score: " + str(fscore)
    outstr += "\n" + "confusion matrix\n" + str(confmatrix)
    outstr += "\n" + "classification report\n" + report
    outstr += "\n" + str(mtr.precision_recall_fscore_support(ytrue, ypred))
    
    if verbose:   
        print(outstr)
    
    
    '''
    c = list(set(ytrue))[1]
    print("conf matrix for class ", c)
    print(mtr.confusion_matrix(ytrue, ypred, labels=[c]))
    #tn, fp, fn, tp = mtr.confusion_matrix(ytrue, ypred, labels=[c]).ravel()
    #print(tn, fp, fn, tp)
    
    labels = list(set(ytrue))
    
    for i, label in enumerate(labels):
        print(i, " counts for label ", label)
        tp, fp, fn, tn, ntrue = get_item_counts(ytrue, ypred, i)
        print("tp, fp, fn, tn, ntrue : ", tp, fp, fn, tn, ntrue)
        print()
    '''
    return acc, fscore, outstr



def get_item_counts(ytrue, ypred, labelindex):
    
        
    confmat = mtr.confusion_matrix(ytrue, ypred)
    
    n_all_instances = np.sum(confmat)
    n_true = sum(confmat[labelindex, :])
    n_predicted = sum(confmat[:, labelindex])
    
    tp = confmat[labelindex, labelindex]
    fn = n_true - tp
    fp = n_predicted - tp
    tn = n_all_instances - (tp + fp + fn)
    
    return tp, fp, fn, tn, n_true




# /** ********************************* *** #
####  metrics giving average values over the whole classification, one score 
# avg should be one of {micro, macro, weighted}

def get_accuracy(ytrue, ypred):
    
    acc = mtr.accuracy_score(ytrue, ypred)
    return acc

def get_f1score(ytrue, ypred, avg="macro"):
    # fscore = mtr.f1_score(ytrue, ypred, average="weighted", pos_label=None) #, labels=list(set(ytrue)))
    fscore = mtr.f1_score(ytrue, ypred, average=avg, pos_label=None)  # , labels=list(set(ytrue)))
    return fscore


def get_precision(ytrue, ypred, avg="macro"):

    return mtr.precision_score(ytrue, ypred, average=avg)

def get_recall(ytrue, ypred, avg="macro"):
    return mtr.recall_score(ytrue, ypred, average=avg)


def get_gmean(ytrue, ypred, avg="macro"):
    pr = get_precision(ytrue, ypred, avg)
    rec = get_recall(ytrue, ypred, avg)
    import scipy.stats.mstats as scipystats
    return scipystats.gmean([pr, rec])

def get_gmean2(precision, recall):
    import scipy.stats.mstats as scipystats
    return scipystats.gmean([precision, recall])

def get_confusion_matrix(ytrue, ypred):
    return mtr.confusion_matrix(ytrue, ypred)



def get_FP_overall(ytrue, ypred):
    return sum(get_FP_perclass(ytrue, ypred))


def get_TN_overall(ytrue, ypred):
    cm = mtr.confusion_matrix(ytrue, ypred)
    return np.sum(cm) - np.sum(np.diag(cm))


def get_FN_overall(ytrue, ypred):
    return sum(get_FN_perclass(ytrue, ypred))


def get_TP_overall(ytrue, ypred):
    return sum(get_TP_perclass(ytrue, ypred))

###  metrics giving average values ******/ ############
#################################################


'''
# the classification should be binary; there should be only two classes.
def get_aucpr(ytrue, yscore=None):
    classes = list(set(ytrue))
    if len(classes) != 2:
        print("Average precision cannot be calculated in non-binary classification.")
        return None

    if yscore == None:
        # assign probability estimates for class membership for each instance
        yscore = [0.5] * len(ytrue)
        yscore = np.array(yscore)
    
    return mtr.average_precision_score(ytrue, yscore)

def get_roc_values(ytrue, yscore=None):
    classes = list(set(ytrue))
    if len(classes) != 2:
        print("Average precision cannot be calculated in non-binary classification.")
        return None

    if yscore == None:
        # assign probability estimates for class membership for each instance
        yscore = [0.5] * len(ytrue)
        yscore = np.array(yscore)
    
    true_positive_rate, false_positive_rate, thresholds = mtr.roc_curve(ytrue, yscore)
    return true_positive_rate, false_positive_rate, thresholds
'''




########## /**********************
#### metrics giving values for each class  ###
# return the number of TP (true positive) instances per class


def get_accuracy_perclass(ytrue, ypred):
    
    TPs = np.array(get_TP_perclass(ytrue, ypred))
    FPs = np.array(get_FP_perclass(ytrue, ypred))
    TNs = np.array(get_TN_perclass(ytrue, ypred))
    FNs = np.array(get_FN_perclass(ytrue, ypred))
    
    acc_perclass = (TPs + TNs) / (TPs + TNs + FPs + FNs)
    return acc_perclass.tolist()


def get_TP_perclass(ytrue, ypred):
    
    cm = mtr.confusion_matrix(ytrue, ypred)
    classes = sorted(list(set(ytrue)))
    
    nTPs = [cm[i,i] for i in range(len(classes))]
    return nTPs

# return the number of TN (true negative) instances per class
def get_TN_perclass(ytrue, ypred):
    
    cm = mtr.confusion_matrix(ytrue, ypred)
    classes = sorted(list(set(ytrue)))

    nTNs = []
    n_instances = np.sum(cm)
    
    for i,_ in enumerate(classes):
        
        tn_i = n_instances - (sum(cm[i,:]) + sum(cm[:,i]) - cm[i,i])
        nTNs.append(tn_i)
    return nTNs

# return the number of FP (false positive) instances per class
def get_FP_perclass(ytrue, ypred):
    
    cm = mtr.confusion_matrix(ytrue, ypred)
    classes = sorted(list(set(ytrue)))

    nFPs = []
    for i,c in enumerate(classes):
        fp_i = sum(cm[:,i]) - cm[i,i]
        nFPs.append(fp_i)
    
    return nFPs

# return the number of FN (false negative) instances per class
def get_FN_perclass(ytrue, ypred):
    
    cm = mtr.confusion_matrix(ytrue, ypred)
    classes = sorted(list(set(ytrue)))

    nFNs = []
    for i,c in enumerate(classes):
        tn_i = sum(cm[i,:]) - cm[i,i]
        nFNs.append(tn_i)
    return nFNs




# returns a list of 2X2 confusion matrices. the list of length the number classes.
# each matrix contains [[TN, FP], [FN, TP]] values, in the given order, for each class.
# in multiclass classification, each matrix is for the target class vs other classes.
def get_confusion_matrices_perclass(ytrue, ypred):
    classes = sorted(list(set(ytrue)))
    TNs = get_TN_perclass(ytrue, ypred)
    FPs = get_FP_perclass(ytrue, ypred)
    FNs = get_FN_perclass(ytrue, ypred)
    TPs = get_TP_perclass(ytrue, ypred)
    
    conf_matrices = []
    
    # confusion matrix is of the form [[TN, FP], [FN, TP]]
    # index 0 is for other classes and index 1 is for the target class.
    for i,_ in enumerate(classes):
        cm = np.zeros((2,2), dtype=int)
        cm[0,0] = TNs[i]
        cm[1,0] = FNs[i]
        cm[0,1] = FPs[i]
        cm[1,1] = TPs[i]
        conf_matrices.append(cm)
    
    return conf_matrices



def get_fscore_perclass(ytrue, ypred):
    
    fscore_vals = mtr.f1_score(ytrue, ypred, average=None)
    return fscore_vals


def get_precision_perclass(ytrue, ypred):
    return mtr.precision_score(ytrue, ypred, average=None)

def get_recall_perclass(ytrue, ypred):
    return mtr.recall_score(ytrue, ypred, average=None)



''' as defined in Sokolova and Lapalme, 2009, A systematic 
analysis of performance measures for classification tasks '''
def get_auc_perclass(ytrue, ypred):
    TPs = np.array(get_TP_perclass(ytrue, ypred))
    FPs = np.array(get_FP_perclass(ytrue, ypred))
    TNs = np.array(get_TN_perclass(ytrue, ypred))
    FNs = np.array(get_FN_perclass(ytrue, ypred))
    
    auc_perclass = 0.5 * ((TPs / (TPs + FNs)) + (TNs / (TNs + FPs)))
    return auc_perclass.tolist()
    

########################################################################
######## *****************************/



''' as defined in Sokolova and Lapalme, 2009, A systematic 
analysis of performance measures for classification tasks '''
def get_auc_avg(ytrue, ypred):
    
    AUCs = get_auc_perclass(ytrue, ypred)
    return sum(AUCs) / len(AUCs)



##################################################################################
#### per class metrics **************/ ###



'''
    ytrue: true labels, of size n_instances
    ypred: predicted labels, of size n_instances
    yscore: prediction scores, of size n_instances X n_labels
'''
def get_performance_results(ytrue_, ypred_, yscore_, figures_folder=None):

    ytrue = np.array(ytrue_)
    ypred = np.array(ypred_)
    yscore = np.array(yscore_)

    result_dict = dict()
    
    result_dict["classification_report"] = mtr.classification_report(ytrue, ypred)

    classes = sorted(list(set(ytrue)))
    # *** Data summary
    result_dict["number_of_instances"] = len(ytrue)
    result_dict["number_of_classes"] = len(classes)
    result_dict["class_names"] = classes
    
    
    result_dict["accuracy"] = get_accuracy(ytrue, ypred)
    # *** Average values:
    avgs = ["macro", "micro", "weighted"]
    for avg in avgs:

        result_dict["f1score_"+avg] = get_f1score(ytrue, ypred, avg=avg)
        result_dict["precision_"+avg] = get_precision(ytrue, ypred, avg=avg)
        result_dict["recall_"+avg] = get_recall(ytrue, ypred, avg=avg)
        result_dict["gmean_pr-rec_"+avg] = get_gmean(ytrue, ypred, avg=avg)

    #  ** Confusion matrix (rows are for true labels, columns are for predicted labels)
    confusion_matrix = get_confusion_matrix(ytrue, ypred)
    result_dict["confusion_matrix"] = confusion_matrix
    
    result_dict["number_of_True_Positives-(sum_of_diagonals)"] = get_TP_overall(ytrue, ypred)
    result_dict["number_of_False_Positives-(sum_of_FPs per class)"] = get_FP_overall(ytrue, ypred)
    result_dict["number_of_False_Negatives-(sum_of_FNs per class)"] = get_FN_overall(ytrue, ypred)
    result_dict["number_of_True_Negatives-(sum_of_nondiagonals)"] = get_TN_overall(ytrue, ypred)
        
    # *** Area under the Receiver Operating Characteristics (ROC) Curve
    
    ytruebin = roc_multiclass.binarize_labels(ytrue)
    
    result_dict["Average_AUC"] = get_auc_avg(ytrue, ypred)
    fpr, tpr, thresholds, roc_auc = roc_multiclass.find_roc_values(ytruebin, yscore)
    
    result_dict["fpr-(false_positive_rate)"] = fpr
    result_dict["tpr-(true-positive-rate)"] = tpr
    result_dict["thresholds"] = thresholds
    result_dict["roc_auc"] = roc_auc
    
    # draw and record the figures
    #  for the classes
    for i,c in enumerate(classes):
        figpath = os.path.join(figures_folder, "roc_curve-class_"+str(c)+".png")
        roc_multiclass.draw_roc_curve(fpr, tpr, roc_auc, i, figpath)
    #  for the model
    figpath = os.path.join(figures_folder, "roc_curve_micro-avg.png")
    roc_multiclass.draw_roc_curve(fpr, tpr, roc_auc, "micro", figpath)
    
    # *** roc auc values, same as above
    averages = ["micro", "macro", "weighted", "samples"]
    for avg in averages:
        result_dict["roc_auc_scores_"+avg+"-avgd"] = roc_multiclass.find_roc_auc_scores(ytruebin, yscore, avg)
    
    
    # *** Average precision scores
    for avg in averages:
        result_dict["average_precision_scores"] = roc_multiclass.find_average_precision_values(ytruebin, yscore, avg)
    
    
    # *** Precision - Recall Curve values
    pr_rec_curve_vals = roc_multiclass.find_precision_recall_curves(ytruebin, yscore)
    result_dict["precision-recall_curve_values"] = pr_rec_curve_vals
    # draw precision - recall curves
    #  for each class
    for i,c in enumerate(classes):
        figpath = os.path.join(figures_folder, "pr-rec-curve_class-"+str(c)+".png")
        prs = pr_rec_curve_vals[i][0]
        recs = pr_rec_curve_vals[i][1]
        roc_multiclass.draw_precision_recall_curve(prs, recs, figpath)
    #  for the model
    figpath = os.path.join(figures_folder, "pr-rec-curve.png")
    prs = pr_rec_curve_vals["micro"][0]
    recs = pr_rec_curve_vals["micro"][1]
    roc_multiclass.draw_precision_recall_curve(prs, recs, figpath)
    
    # *** Per class values
    
    precisions_per_class = get_precision_perclass(ytrue, ypred)
    recalls_per_class = get_recall_perclass(ytrue, ypred)
    gmeans_per_class = np.array([get_gmean2(pr, rec) for pr,rec in zip(precisions_per_class,
                                                                       recalls_per_class)])
    result_dict["accuracies_per_class"] = get_accuracy_perclass(ytrue, ypred)
    result_dict["f1scores_per_class"] = get_fscore_perclass(ytrue, ypred)
    result_dict["precisions_per_class"] = precisions_per_class
    result_dict["recalls_per_class"] = recalls_per_class
    result_dict["gmeans_per_class"] = gmeans_per_class
    result_dict["avg_aucs_per_class"] = get_auc_perclass(ytrue, ypred)
    result_dict["confusion_matrices_per_class"] = get_confusion_matrices_perclass(ytrue, ypred)
    result_dict["TPs_per_class"] = get_TP_perclass(ytrue, ypred)
    result_dict["FPs_per_class"] = get_FP_perclass(ytrue, ypred)
    result_dict["FNs_per_class"] = get_FN_perclass(ytrue, ypred)
    result_dict["TNs_per_class"] = get_TN_perclass(ytrue, ypred)
    n_actuals = [sum(confusion_matrix[i,:]) for i,_ in enumerate(classes)]
    n_predicteds = [sum(confusion_matrix[:,i]) for i,_ in enumerate(classes)]
    result_dict["n_actuals_per_class"] = n_actuals
    result_dict["n_predicteds_per_class"] = n_predicteds
    
    return result_dict





# full report
def print_classification_report(ytrue, ypred):
    print("\n******* CLASSIFICATION REPORT **************")
    print(mtr.classification_report(ytrue, ypred))
    print("********************************************")


    
