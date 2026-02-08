'''
Created on Mar 23, 2018

@author: dicle
'''


import os
import pandas as pd

import sklearn.linear_model as sklinear
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
import sklearn.pipeline as skpipeline
import sklearn.feature_extraction.dict_vectorizer as dv
import sklearn.metrics as skmetrics
import sklearn.model_selection as cv
from sklearn.preprocessing.label import LabelEncoder

#############################################
class TextualStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return [{'length': len(text),
                 }
                for text in texts]

def get_textual_stats_pipeline():
    
    transformer = TextualStats()
    dictvectorizer = dv.DictVectorizer()
    pipeline = skpipeline.Pipeline([('tstats_transformer', transformer),
                                    ('tstats_vect', dictvectorizer)])
    return pipeline
###############################################





#########################################
# Column selection

'''
def get_text_column(df):
    return df["DESCRIPTION"].tolist()

def get_extra_column(df):
    #return df[["CAUSE_CODE"]]
    return df["CAUSE_CODE"].tolist()

def get_text_col_transformer_():
    
    pipeline = skpipeline.Pipeline([('text_col_selector', FunctionTransformer(get_text_column, validate=False)),
                                    #('label_encoder', LabelEncoder())
                                    ])
    return pipeline
'''



def get_text_column(df, txtcolname):
    return df[txtcolname].tolist()

def get_extra_column(df):
    #return df[["CAUSE_CODE"]]
    return df["CAUSE_CODE"].tolist()

def get_text_col_transformer_(txtcolname):
    
    pipeline = skpipeline.Pipeline([('text_col_selector', FunctionTransformer(get_text_column, validate=False)),
                                    #('label_encoder', LabelEncoder())
                                    ])
    return pipeline


def get_extra_col_transformer1():
    
    pipeline = skpipeline.Pipeline([('extra_col_selector', FunctionTransformer(get_extra_column, validate=False)),
                                    ('label_encoder', LabelEncoder())
                                    ])
    return pipeline


class ExtraColumnTransformer(BaseEstimator, TransformerMixin):
    colname = ""
    def __init__(self, colname):
        self.colname = colname
    def fit(self, X, y=None):
        return self
    def transform(self, df):
        col_values = df[self.colname].tolist()
        return [{self.colname+"_feature" : col_value} for col_value in col_values]

def get_extra_col_transformer(colname):
    transformer = ExtraColumnTransformer(colname)
    dictvectorizer = dv.DictVectorizer()
    pipeline = skpipeline.Pipeline([('etransformer', transformer),
                                    ('evectorizer', dictvectorizer)])
    return pipeline

########################################


class TextColumnSelector(BaseEstimator, TransformerMixin):
    colname = ""
    def __init__(self, txtcolname):
        self.txtcolname = txtcolname
    def fit(self, X, y=None):
        return self
    def transform(self, df):
        
        col_values = df[self.txtcolname].tolist()
        #return [{"text" : text} for text in col_values]
        return col_values
'''
def get_text_col_transformer(txtcolname):
    transformer = TextColumnSelector(txtcolname)
    dictvectorizer = dv.DictVectorizer()
    pipeline = skpipeline.Pipeline([('ttransformer', transformer),
                                    ('tvectorizer', dictvectorizer)])
    return pipeline
'''
##################################################

def get_tfidf_pipeline():
    tfidf_pipeline = skpipeline.Pipeline([('tfidf', TfidfVectorizer(use_idf=True))])
    return tfidf_pipeline

def extract_features1():    
    
    final_features = [('col_selection', get_text_col_transformer()),
                      ('txt_length', get_textual_stats_pipeline()),
                      ('tfidf_vect', get_tfidf_pipeline()),
                      ]
    
    text_features = skpipeline.FeatureUnion(transformer_list=final_features)
                                        
                                             
        
    return text_features

def extract_features():    
    
    final_features = [
                      ('txt_length', get_textual_stats_pipeline()),
                      ('tfidf_vect', get_tfidf_pipeline()),
                      ]
    
    text_features = skpipeline.FeatureUnion(transformer_list=final_features)
                                        
                                
    pipeline1 = skpipeline.Pipeline([('col_selection', TextColumnSelector(txtcolname="DESCRIPTION")),
                                #('col_selection', get_text_col_transformer(txtcolname="DESCRIPTION")),
                                #('col_selection', get_text_col_transformer_(txtcolname="DESCRIPTION")),
                                ('features', text_features)
                                ])
    
    pipeline2 = get_extra_col_transformer(colname="CAUSE_CODE")
    
    union = skpipeline.FeatureUnion([('pipe1', pipeline1),
                                     ('pipe2', pipeline2)])
    return union
                        

def get_classifier():
    
    return sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
    

def extract_features_():    
    
    final_features = [('txt_length', get_textual_stats_pipeline()),
                      'tfidf_vect', TfidfVectorizer(use_idf=True)]
    
    text_features = skpipeline.Pipeline([('text_column_selector', FunctionTransformer_),
                                             ('text_features', 
                                                skpipeline.FeatureUnion(transformer_list=final_features
                                                                        ))
                                             ])
                                             
        
        
    column_features = skpipeline.Pipeline([('extra_column_selector', FunctionTransformer_),
                                           ('extra_column_features', ExtraColumnFeatures)])
    
    features = skpipeline.FeatureUnion(transformer_list=[text_features, column_features])  
    

    folder = "<PATH>"
    train_file = "data_20180314_2_train.csv"
    test_file = "data_20180314_2_test.csv"
    sep = "\t"
    
    trainpath = os.path.join(folder, train_file)
    traindf = pd.read_csv(trainpath, sep=sep)
    
    #texts = traindf["DESCRIPTION"].tolist()
    labels = traindf["CAUSE_CODE"].tolist()
 
    
    model = skpipeline.Pipeline([('features', extract_features()),
                                 ('classifier', get_classifier())])

    ypred = cv.cross_val_predict(model, traindf, labels)
    print(skmetrics.classification_report(labels, ypred))
    
    
    
    
    
    
    
    
    
    
    
    
    

