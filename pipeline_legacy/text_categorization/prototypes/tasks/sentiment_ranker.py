'''
Created on May 8, 2017

@author: dicle
'''


import sys
sys.path.append("..")


import os
from time import time 

import sklearn.pipeline as skpipeline
import sklearn.feature_extraction.text as sktext
import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb

from abc import abstractmethod


import text_categorization.prototypes.classification.classification_task as clsf_task
import text_categorization.prototypes.feature_extraction.text_preprocessor as prep
import text_categorization.prototypes.feature_extraction.token_based_transformers2 as toktrans
import text_categorization.prototypes.feature_extraction.text_based_transformers as txtrans

from misc import num_utils



SINGLE_WORD_PRED_PROB = 0.99

class SentimentRanking(clsf_task._ClassificationTask):
    
    sentiment_type = ""
    
    def __init__(self, feature_config, classifier, task_name="Sentiment Analysis"):

        self.task_name = task_name
        super().__init__(feature_config, classifier, self.task_name,)
        # default values
        
    


        
    
    @abstractmethod
    def _generate_feature_extraction_pipeline(self):
        

        lang = self.feature_config.lang
        feature_weights = self.feature_config.weights
        prep_params = self.feature_config.prepchoice
        
        self.sentiment_type = self.feature_config.sentiment_type
        
            # features found in the processed tokens
        preprocessor = prep.Preprocessor(lang=lang,
                                         stopword=prep_params.stopword, more_stopwords=prep_params.more_stopwords,
                                         spellcheck=prep_params.spellcheck,
                                         stemming=prep_params.stemming,
                                         remove_numbers=prep_params.remove_numbers,
                                         deasciify=prep_params.deasciify,
                                         remove_punkt=prep_params.remove_punkt,
                                         lowercase=prep_params.lowercase
                                    )
        
        tfidfvect = sktext.TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                    use_idf=prep_params.use_idf,
                                    ngram_range=prep_params.wordngramrange,
                                    max_features=prep_params.nmaxfeature)
    
        
        polpipe3 = toktrans.get_lexicon_count_pipeline2(sentiment_type=self.sentiment_type, 
                                                        lexicontype=lang, tokenizer=prep.identity)
        
        
        token_weights = dict(tfidfvect=feature_weights["word_tfidf"],
                             polpipe3=feature_weights["lexicon_count"])
        token_transformers_dict = dict(tfidfvect=tfidfvect,  # not to lose above integrity if we change variable names
                                       polpipe3=polpipe3)
        token_transformers = [(k, v) for k, v in token_transformers_dict.items()]
        
        tokenbasedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
                                              # ('nadropper', tbt.DropNATransformer()),                                       
                                              ('union1', skpipeline.FeatureUnion(
                                                    transformer_list=token_transformers,
                                                    transformer_weights=token_weights                                            
                                            )), ]
                                            )
        
        
        
        
        charngramvect = sktext.TfidfVectorizer(analyzer='char_wb', 
                                        ngram_range=prep_params.charngramrange, 
                                        lowercase=False)
        
        polpipe1 = txtrans.get_polylglot_polarity_count_pipe(lang)
        polpipe2 = txtrans.get_polylglot_polarity_value_pipe(lang)
        
        
        text_weights = dict(charngramvect=feature_weights["char_tfidf"],   # @TODO hardcoded
                                 polpipe1=feature_weights["polyglot_count"],
                                 polpipe2=feature_weights["polyglot_value"],
                                 )
        text_transformers_dict = dict(charngramvect=charngramvect,
                                 polpipe1=polpipe1,
                                 polpipe2=polpipe2,
                                 )
        text_transformers = [(k, v) for k, v in text_transformers_dict.items()]
        '''
        textpipes = [('charngramvect', charngramvect),]
        textpweights = {'charngramvect' : 1.5}
        textpweights = dict(charngramvect = 1 if charngramvect else 0)
        '''
        textbasedpipe = skpipeline.Pipeline([('union2', skpipeline.FeatureUnion(transformer_list=text_transformers,
                                                                                transformer_weights=text_weights),)])
        
        print(text_weights)
        
        final_transformers_dict = dict(tokenbasedpipe=tokenbasedpipe,
                                       textbasedpipe=textbasedpipe)
        final_transformers = [(k, v) for k, v in final_transformers_dict.items()]
        
        
        '''        
        #tweights = {k : 1 if v else 0 for k,v in final_transformers.items()}
        check_zero = lambda x : 1 if sum(x) > 0 else 0
        x = list(tokenbasedpipe.get_params(False).values())
        print(len(x), x[0])
        print(x[0][1])   # convert x[0] tuple to dict, then get transformer weights
        print("**")
        print(x,"\n--")
        print(list(textbasedpipe.get_params(False).values()))
        tweights = {k : check_zero(list(k.get_params(False).values())[0][0][1].get_params(False)["transformer_weights"].values())
                          for _, k in final_transformers_dict.items()}
        '''
    
        feature_union = skpipeline.FeatureUnion(transformer_list=final_transformers,
                                           # transformer_weights=tweights   # weight assignment is not necessary as the number of features is small
                                           )
    
        return feature_union
    
    
    
    
    
    def predict(self, model, test_instances):

        predicted_labels = []
        prediction_map = []
        
        for text in test_instances:               
            pred_labels, pred_map = super().predict(model, [text])
            predicted_labels.extend(pred_labels)
            prediction_map.extend(pred_map)
                
        
        return predicted_labels, prediction_map
    
    
    '''  
    def predict(self, model, test_instances):

        SINGLE_WORD_PRED_PROB = 99
        REMAINING_PROBS = 10

        predicted_labels = []
        prediction_map = []
        
        cats = model.classes_
        print("cats: ", cats)
        for text in test_instances:
            
            if len(text.split()) == 1:  # single_word
                pred_label = toktrans.single_word_polarity(text, lang=self.feature_config.lang)
                predicted_labels.append(pred_label)
                
                pred_probs = {}
                pred_probs[pred_label] = SINGLE_WORD_PRED_PROB / 100.0
                              
                
                other_cats = [c for c in cats if c != pred_label]
                
                neut_labels = ["neutral", "nötr", "neut"]
                if pred_label in neut_labels:
                    other_cats = [c for c in cats if c not in neut_labels]
                
                               
                
                n = len(other_cats)
                
                if n == 1:
                    pred_probs[other_cats[0]] = REMAINING_PROBS / 100.0
                    
                
                else:
                    other_probs = []
                    num_utils.generate_random_numbers_with_sum(n, REMAINING_PROBS, other_probs)
                    other_probs = list(map(lambda x : x / 100.0, other_probs))
                    for c,p in zip(other_cats, other_probs):
                        pred_probs[c] = p
                
                
                prediction_map.append({"predicted_label" : pred_label,
                                       "prediction_probability" : pred_probs})
            
            
            
            
            else:                   
                pred_labels, pred_map = super().predict(model, [text])
                predicted_labels.extend(pred_labels)
                prediction_map.extend(pred_map)
                
        
        return predicted_labels, prediction_map
    '''    
 
    
    '''   
    def predict(self, model, test_instances):

        SINGLE_WORD_PRED_PROB = 99
        REMAINING_PROBS = 10

        predicted_labels = []
        prediction_map = []
        
        cats = model.classes_
        print("cats: ", cats)
        for text in test_instances:
            
            if len(text.split()) == 1:  # single_word
                pred_label = toktrans.single_word_polarity(text, lang=self.feature_config.lang)
                predicted_labels.append(pred_label)
                
                pred_probs = {}
                pred_probs[pred_label] = SINGLE_WORD_PRED_PROB / 100.0
                              
                
                other_cats = [c for c in cats if c != pred_label]
                
                neut_labels = ["neutral", "nötr", "neut"]
                if pred_label in neut_labels:
                    other_cats = [c for c in cats if c not in neut_labels]
                
                               
                
                n = len(other_cats)
                
                if n == 1:
                    pred_probs[other_cats[0]] = REMAINING_PROBS / 100.0
                    
                
                else:
                    other_probs = []
                    num_utils.generate_random_numbers_with_sum(n, REMAINING_PROBS, other_probs)
                    other_probs = list(map(lambda x : x / 100.0, other_probs))
                    for c,p in zip(other_cats, other_probs):
                        pred_probs[c] = p
                
                
                prediction_map.append({"predicted_label" : pred_label,
                                       "prediction_probability" : pred_probs})
            
            
            
            
            else:                   
                pred_labels, pred_map = super().predict(model, [text])
                predicted_labels.extend(pred_labels)
                prediction_map.extend(pred_map)
                
        
        return predicted_labels, prediction_map
        
    '''
    
    
'''
4 tasks:
1- performance test : cross_val
2- train & save
3- predict offline
4- predict online
'''    





    
    print()