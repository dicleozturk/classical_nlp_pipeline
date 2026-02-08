'''
Created on Jan 21, 2019

@author: dicle
'''


import os
import re

from pprint import pprint

'''
 convert Tur2003's annotations to conll. given annotations are in the form of
 "text text <\s[LABEL text*\s]\s> text". 

'''




entsep = "_"
tokensep = "##"
labels = ["LOC", "PER", "ORG"]

'''
 append missing dots to each line.
'''
def append_dot(lines):
    #lines = open(fpath).readlines()
    
    lines = [l.strip() for l in lines]
    lines = [l+" . " for l in lines]
    
    return lines


def join_lines(lines):
    
    text = " ".join(lines)
    return text


'''
 entity: "[LBL\stext*\s]"
 get rid of spaces, join multi word texts with "-".
'''
def unify_entity_markers(text):
    
    pattern = r"\[[A-Z]{3} [\w\s\-]+ \]"
    entities = re.findall(pattern, text)
    
    pprint(entities)
    
    entities_replacement = []
    for entity in entities:
        pieces = entity[1:-1].split()
        print(" ", entity)
        print(pieces)
        pieces = [i.strip() for i in pieces]
        repl = pieces[0] + entsep + tokensep.join(pieces[1:])
        print(repl)
        print()
        entities_replacement.append((entity, repl))
    
    for ent, repl in entities_replacement:
        text = text.replace(ent, repl)
    
    return text



'''
 split by space; mark the entity tokens.
 convert to conll token list.
'''
def tokenize_for_conll(text, docid, pos2=-1):
    
    tokens = text.split()
    tokens = [t.strip() for t in tokens]
    
    conll_tokens = []
    
    pos1 = 0
    #pos2 = -1
    for token in tokens:
        
        is_entity = False
        for label in labels:
            if token.startswith(label+entsep):
                is_entity = True
                break
        
        if is_entity:
            pieces1 = token.split(entsep)
            entity_label = pieces1[0]
            entity_tokens = pieces1[1].split(tokensep)
            
            for i,et in enumerate(entity_tokens):
                if i == 0:
                    token_label = "B-" + entity_label
                else:
                    token_label = "I-" + entity_label
                
                pos1 = pos2 + 1
                pos2 = pos1 + len(et)
                
                conll_tokens.append((et, docid, pos1, pos2, token_label))
        
        else:
            pos1 = pos2 + 1
            pos2 = pos1 + len(token)
            token_label = "O"
            conll_tokens.append((token, docid, pos1, pos2, token_label))
    
    return conll_tokens
            
        

def write_conll_file(fpath, conll_tokens):    
    
    f = open(fpath, "w")
    
    header = "-DOCSTART- -" + conll_tokens[1][1] +"- -X- O"
    f.write(header+"\n\n")
    
    for pieces in conll_tokens:
        
        str_pieces = [str(i) for i in pieces]
        line = " ".join(str_pieces) + "\n"
        f.write(line)
    
    f.write("\n\n")
    f.close()


def conversion_pipeline(inpath, docid, outpath):
    
    lines = open(inpath, "r").readlines()

    lines = append_dot(lines)
    lines = [unify_entity_markers(text) for text in lines]
    text = join_lines(lines)
    conll_tokens = tokenize_for_conll(text, docid, pos2=-1)
    write_conll_file(outpath, conll_tokens)
    
    print("Conversion complete for ", docid)
    
    


    
    