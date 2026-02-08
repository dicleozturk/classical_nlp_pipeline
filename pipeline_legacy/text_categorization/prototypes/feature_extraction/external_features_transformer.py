'''
Created on Mar 26, 2018

@author: dicle
'''

import sys
sys.path.append("..")


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





### Add external features, non-text columns as features to the feature extraction steps.






#############################
# initialized with a pandas df.

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
    txtcolname = ""
    def __init__(self, txtcolname):
        self.txtcolname = txtcolname
    def fit(self, X, y=None):
        return self
    def transform(self, df):
        col_values = df[self.txtcolname].tolist()
        #return [{"text" : text} for text in col_values]
        return col_values
    

