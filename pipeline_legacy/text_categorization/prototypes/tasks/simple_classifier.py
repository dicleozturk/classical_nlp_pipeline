'''
Created on Oct 30, 2017

@author: dicle
'''


import sys
sys.path.append("..")



from sklearn.feature_extraction.text import TfidfVectorizer

from abc import abstractmethod




import sklearn.pipeline as skpipeline

import text_categorization.prototypes.classification.classification_task as clsf_task
import text_categorization.prototypes.feature_extraction.text_preprocessor as prep
import text_categorization.prototypes.feature_extraction.text_based_transformers as txtrans



class SimpleTextClassification(clsf_task._ClassificationTask):
    
    
    def __init__(self, feature_config, classifier, task_name="Simple Categorization"):
        self.task_name = task_name
        super().__init__(feature_config, classifier, self.task_name,)
        
        
    
    @abstractmethod
    def _generate_feature_extraction_pipeline(self):
        
        lang = self.feature_config.lang
        final_weights = self.feature_config.weights
        prep_params = self.feature_config.prepchoice
    
                   
            # features found in the processed tokens
        token_features = []
        token_weights = {}
    
        preprocessor = prep.Preprocessor(lang=lang,
                                         stopword=prep_params.stopword, 
                                         more_stopwords=prep_params.more_stopwords,
                                         spellcheck=prep_params.spellcheck,
                                         stemming=prep_params.stemming,
                                         remove_numbers=prep_params.remove_numbers,
                                         deasciify=prep_params.deasciify,
                                         remove_punkt=prep_params.remove_punkt,
                                         lowercase=prep_params.lowercase
                                    )
        
        tfidfvect = TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                    use_idf=prep_params.use_idf,
                                    ngram_range=prep_params.wordngramrange,
                                    max_features=prep_params.nmaxfeature)
        
        tfidf_feat_name = "tfidf_feature"
        token_features.append((tfidf_feat_name, tfidfvect))
        token_weights[tfidf_feat_name] = 1
        
        
        text_based_features = []
        text_weights = {}
        
        avg_word_length = txtrans.get_avg_word_pipeline()
        avg_wrd_feat_name = "avg_word_length"
        text_based_features.append((avg_wrd_feat_name, avg_word_length))
        text_based_features.append((new_feat_name, new_feat_obj))


        text_weights[avg_wrd_feat_name] = 1
        text_weights[new_feat_name] = 1
        
        tokenbasedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
                                              # ('nadropper', tbt.DropNATransformer()),                                       
                                              ('union1', skpipeline.FeatureUnion(
                                                    transformer_list=token_features ,
                                                    transformer_weights=token_weights                                                
                                                    )),
                                            ])
    
        textbasedpipe = skpipeline.Pipeline([('union2', skpipeline.FeatureUnion(
                                                transformer_list=text_based_features,
                                                transformer_weights=text_weights
                                                ),
                                              )
                                            ])
        
        
        final_transformers = []
        final_transformers.append(("token_pipe", tokenbasedpipe),
                                   "text_pipe", textbasedpipe)
        
        
        feature_union = skpipeline.FeatureUnion(transformer_list=final_transformers)
        
        return feature_union
        
    
    