'''
Created on Mar 1, 2017

@author: dicle
'''


import sys
sys.path.append("..")


from sklearn.base import BaseEstimator, TransformerMixin

import sklearn.feature_extraction.dict_vectorizer as dv
import sklearn.pipeline as skpipeline


import text_categorization.prototypes.feature_extraction.sentiment_feature_extractors as sf

#########   sentiment lexicon counting - single type  ###################




###### call lexicon types (pos / neg) separately ###################3



# sentiment is pos or neg
def lexicon_count_feature_extraction_dict2(tokens, sentiment_type, lexicon_type):
        
    #keys = ["pos_lex", "neg_lex"]
    keys = ["sent_lex"]

    nwords = len(tokens)
    # t = Text(text, hint_language_code=lang)
    # nentities = len(t.entities)
    
    if nwords == 0:
        d = {key : 0.0 for key in keys}
        return d
    
    #print("lexicon type", lexicon_type)
    nsent_lex = 0
    if lexicon_type in ["tr", "turkish"]:
        # boun lexicon
        nsent_lex = sf.get_boun_polarity_count(tokens, sentiment_type)
    
    if lexicon_type in ["en", "english", "eng"]:
        nsent_lex = sf.get_english_polarity_count(tokens, sentiment_type)
            
    if lexicon_type in ["emoticon", "emot"]:
        nsent_lex = sf.get_emoticon_polarity_count(tokens, sentiment_type)    
    '''
    if lexicon_type in ["ar", "arab", "arabic"]:
        npos_lex, nneg_lex = sf.get_arabic_polarity_count1(tokens)
    ''' 
    
    
    values = [nsent_lex]  # , nentities]
    f = lambda x : round(float(x) / nwords, 5)
    # f = lambda x : round(float(x), 3)
    values = list(map(f, values))    
  
    value_dict = dict()
    for key, value in zip(keys, values):
        value_dict[key] = value

    #print("value_dict ", value_dict, tokens)

    return value_dict




# # should be called after a tokenizer/preprocessor
class CountBasedTransformer2(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
      
    tokenizer = None
    lexicon_choice = ""   # en, tr or emoticon
    def __init__(self, sentiment_type, lexicon_choice, tokenizer=None):  # tokenize can be a callable as in sklearn.tfidfvectorizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = lambda text : text.split()
        
        self.sentiment_type = sentiment_type
        
        self.lexicon_choice = lexicon_choice
        if len(self.lexicon_choice.strip()) == 0:
            self.language = "en"
        
    def fit(self, X, y=None): 
        
              
        return self
    
    # tokens_list = [[tokens_of_doc_i]]
    def transform(self, X):   
        
        tokens_list = [self.tokenizer(doc) for doc in X]  # [[tokens_of_doc_i]]
        
        return [lexicon_count_feature_extraction_dict2(tokens, self.sentiment_type, self.lexicon_choice) for tokens in tokens_list]
   


###################################################



# @TODO call this in eng and tr feature pipeline builders

def get_lexicon_count_pipeline2(sentiment_type, lexicontype, tokenizer):
    lexpipe = skpipeline.Pipeline([('lexfeatures', CountBasedTransformer2(sentiment_type, lexicontype, tokenizer)),
                                   ('lexvect', dv.DictVectorizer()),
                                   ])
    return lexpipe



'''  when will we use these?  ### 
def get_TR_lexicon_count_pipeline(tokenizer):
    
    lexpipe = skpipeline.Pipeline([('lexfeatures', CountBasedTransformer("tr", tokenizer)),
                                   ('lexvect', dv.DictVectorizer()),
                                   ])
    return lexpipe


def get_EN_lexicon_count_pipeline(tokenizer):
    
    lexpipe = skpipeline.Pipeline([('lexfeatures', CountBasedTransformer("en", tokenizer)),
                                   ('lexvect', dv.DictVectorizer()),
                                   ])
    return lexpipe

def get_EMOT_lexicon_count_pipeline(tokenizer):
    
    lexpipe = skpipeline.Pipeline([('lexfeatures', CountBasedTransformer("emoticon", tokenizer)),
                                   ('lexvect', dv.DictVectorizer()),
                                   ])
    return lexpipe

'''

#################################################


def single_word_polarity(word, lang="tr"):
    
    label = ""
    lex_results = lexicon_count_feature_extraction_dict([word], lexicon_type=lang)
    for k,v in lex_results.items():
        if v == 1:
            label = k[:3]
            return label
            
    # lexicon_count_feature_extraction_dict() returns {pos_lex: ?, neg_lex= ?}. if the key having value 1 is the label.
    emoticon_results = lexicon_count_feature_extraction_dict([word], lexicon_type="emot")
    for k,v in emoticon_results.items():
        if v == 1:
            label = k[:3]
            return label

    from text_categorization.prototypes.classification import CLSF_CONSTANTS
    label = CLSF_CONSTANTS.NEUTRAL_LABEL
    return label



