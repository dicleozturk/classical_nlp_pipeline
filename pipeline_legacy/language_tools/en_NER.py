'''
Created on Oct 17, 2016

@author: dicle
'''

# source: http://stackoverflow.com/questions/31836058/nltk-named-entity-recognition-to-a-python-list
'''
 (english-only)
 recoginition of PERSON, ORGANIZATION, LOCATION  by the NERs of NLTK and PolyGlot;
                 PERSON, ORGANIZATION, LOCATION, MONEY, PERCENT, DATE by Stanford NER.
'''

import os

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from polyglot.text import Text

import nltk.tag.stanford as st


class StanfordNER():
    ner_tagger = None
    
    # classifier_choice = {0:3class, 1:4classs, 2:7class, 3:non-distsim}
    def __init__(self, classifier_choice=2):
        
        nerfolderpath = "<PATH>"
        ext = ".ser.gz"
        classifiers = ["english.all.3class.distsim.crf",
                       "english.conll.4class.distsim.crf",
                       "english.muc.7class.distsim.crf"
                       ]
        nerclassifierpath = os.path.join(nerfolderpath, "classifiers", classifiers[classifier_choice] + ext)
        
        nerjarname = "stanford-ner-3.6.0.jar"
        nerjarpath = os.path.join(nerfolderpath, nerjarname)
        self.ner_tagger = st.StanfordNERTagger(nerclassifierpath, nerjarpath)

    
    def get_tagged_tokens(self, text):
        
        tokens = word_tokenize(text)
        taggedtokens = self.ner_tagger.tag(tokens)
   
        
        from itertools import groupby
        entities = []
        for tag, chunk in groupby(taggedtokens, lambda x:x[1]):
            if tag != "O":
                entities.append((" ".join(w for w, _ in chunk), tag))
        
        
        return entities
    


class NltkNER():
    
    def __init__(self):
        return
    
    # @TODO improve to get tag labels

    def get_tagged_tokens(self, text):
        
        chunked = ne_chunk(pos_tag(word_tokenize(text)))
        
        prev = None
        continuous_chunk = []
        current_chunk = []
        entity_names_types = []
    
        for chunk in chunked:
            if type(chunk) == Tree:
                entity_name = " ".join([token for token, pos in chunk.leaves()])
                current_chunk.append(entity_name) 
                entity_names_types.append((entity_name, chunk.label()))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue
    
        # return continuous_chunk, entity_names_types
        return entity_names_types

        
class PolyglotNER():
    
    def __init__(self):
        return
    
    def get_tagged_tokens(self, text):
        ptext = Text(text)  # output can be re-organised
        # @TODO # do this per sentence
        entities = [(" ".join(entity), entity.tag) for entity in ptext.entities]
        return entities



    nltk_entities = NltkNER().get_tagged_tokens(text)
    print("nlt<PATH>", nltk_entities)
    
    stanford_entities = StanfordNER().get_tagged_tokens(text)
    print("stanfor<PATH>", stanford_entities)
    
    pg_entities = PolyglotNER().get_tagged_tokens(text)
    print("polyglo<PATH>", pg_entities)

  
    
    
