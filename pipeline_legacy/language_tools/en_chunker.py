'''
Created on Sep 11, 2017

@author: dicle
'''

import sys
sys.path.append("..")

import bllipparser as bparser

#rrp = bparser.RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
parser = bparser.RerankingParser.from_unified_model_dir("<PATH>")


def extract_noun_phrases(sentence):

    parsestr = parser.simple_parse(sentence)

    tree_ = bparser.Tree(parsestr)  # S1 (S *P ..)
    
    tree = tree_[0]   # S (*P ..)


    ttree = tree
    while(type(ttree) is bparser.Tree):
        
        nchilds = len(ttree)
        for i in range(nchilds):
            print(ttree[i])
            ttree = ttree[i]


def get_parse_tree(sentence):

    parsestr = parser.simple_parse(sentence)

    tree_ = bparser.Tree(parsestr)  # S1 (S *P ..)
        
    return tree_


def get_parse_tree2(parser, sentence):

    parsestr = parser.simple_parse(sentence)

    tree_ = bparser.Tree(parsestr)  # S1 (S *P ..)
        
    return tree_


def get_parse_tree3(sentence):

    bparser.RerankingParser._parser_model_loaded = False
    parser_ = bparser.RerankingParser.from_unified_model_dir("<PATH>")

    parsestr = parser_.simple_parse(sentence)

    tree_ = bparser.Tree(parsestr)  # S1 (S *P ..)
    del parser_
    return tree_



def traverse(tree):
    
    for subtree in tree.subtrees():
        print(" * ", subtree, " ** ", subtree.label)
        
        traverse(subtree)
    
    

def get_phrases(tree, phrase_type="NP"):
    
    all_subtrees = tree.all_subtrees()
    phrases = [st for st in all_subtrees if st.label == phrase_type]
    return phrases

def _get_nouns(tree):
    
    nps = get_phrases(tree, phrase_type="NP")
    #print("phrases: ", nps)
    
    np_nouns = []
    
    for np in nps:
        subtrees = np.subtrees()
        
        #print(np, "   ", list(subtrees))
        for st in subtrees:
            if st.label == "NN":
                #print(st)
                np_nouns.append((np, st))

    return np_nouns



def _get_np_items(tree):
    
    nps = get_phrases(tree, phrase_type="NP")
    #print("phrases: ", nps)
    
    np_nouns = []
    
    for np in nps:
        subtrees = np.subtrees()
        
        #print(np, "   ", list(subtrees))
        add = False
        
        noun = []
        others = []
        phrase = np
            
        for st in subtrees:
            
            if st.label in ["NN", "NNS", "NNP"]:
                #print(st)
                noun.append(st)
                add = True
            elif st.label != "DT":
                others.append(st)
        if add:
            np_nouns.append(dict(phrase=phrase, noun=noun, others=others))

    return np_nouns

def get_nouns(input_):
    
    if type(input_) is str:
        tree = get_parse_tree(input_)
    elif type(input_) is bparser.Tree:
        tree = input_
    else:
        print("inappropriate input_.")
        return
    
    _np_nouns = _get_nouns(tree)
    np_nouns = []
    for np, noun in _np_nouns:
        #print("np: ", np.tokens(), " noun: ", noun.tokens())
        np_nouns.append((" ".join(np.tokens()), " ".join(noun.tokens())))

    return np_nouns


def get_np_items(input_):
    
    if type(input_) is str:
        tree = get_parse_tree(input_)
    elif type(input_) is bparser.Tree:
        tree = input_
    else:
        print("inappropriate input_.")
        return
    
    _np_items = _get_np_items(tree)  # {phrase:Tree, others:[Tree], noun:[Tree]} - noun is the noun words, others are the non-NN and non-DT words in the phrase
    np_items = []
    for npitem in _np_items:
        #print("np: ", np.tokens(), " noun: ", noun.tokens())
        #print(npitem)
        phrase = " ".join(npitem["phrase"].tokens())
        #print(type(npitem["noun"][0]))
        
        #noun = " ".join([n.token for n in npitem["noun"]])
        noun = [n.token for n in npitem["noun"]]
        #print(type(npitem["others"]), npitem["others"][0].tokens)
        
        #nonnouns = " ".join([" ".join(n.tokens()) for n in npitem["others"]])
        nonnouns = [" ".join(n.tokens()) for n in npitem["others"]]

        np_items.append({"phrase": phrase, "noun" : noun, "non-noun" : nonnouns})
    return np_items



def get_np_items2(parser, sentence):
    

    tree = get_parse_tree2(parser, sentence)

    _np_items = _get_np_items(tree)  # {phrase:Tree, others:[Tree], noun:[Tree]} - noun is the noun words, others are the non-NN and non-DT words in the phrase
    np_items = []
    for npitem in _np_items:
        #print("np: ", np.tokens(), " noun: ", noun.tokens())
        #print(npitem)
        phrase = " ".join(npitem["phrase"].tokens())
        #print(type(npitem["noun"][0]))
        
        #noun = " ".join([n.token for n in npitem["noun"]])
        noun = [n.token for n in npitem["noun"]]
        #print(type(npitem["others"]), npitem["others"][0].tokens)
        
        #nonnouns = " ".join([" ".join(n.tokens()) for n in npitem["others"]])
        nonnouns = [" ".join(n.tokens()) for n in npitem["others"]]

        np_items.append({"phrase": phrase, "noun" : noun, "non-noun" : nonnouns})
    return np_items


def get_np_items3(sentence):
    

    tree = get_parse_tree3(sentence)

    _np_items = _get_np_items(tree)  # {phrase:Tree, others:[Tree], noun:[Tree]} - noun is the noun words, others are the non-NN and non-DT words in the phrase
    np_items = []
    for npitem in _np_items:
        #print("np: ", np.tokens(), " noun: ", noun.tokens())
        #print(npitem)
        phrase = " ".join(npitem["phrase"].tokens())
        #print(type(npitem["noun"][0]))
        
        #noun = " ".join([n.token for n in npitem["noun"]])
        noun = [n.token for n in npitem["noun"]]
        #print(type(npitem["others"]), npitem["others"][0].tokens)
        
        #nonnouns = " ".join([" ".join(n.tokens()) for n in npitem["others"]])
        nonnouns = [" ".join(n.tokens()) for n in npitem["others"]]

        np_items.append({"phrase": phrase, "noun" : noun, "non-noun" : nonnouns})
    return np_items

    #main()
    
    sentence = "i'm looking for a nice black coloured iphone with lightning to usb cable feature"
    sentence = "I want a golden nice 64 gb iphone and a black phone."
    sentence = "i want 2 mbps fast adsl"
    tree = get_parse_tree(sentence)
    print(tree)
    print(get_np_items(tree))
    print("\n")
    sentence = "i want 8 mbps fast adsl"
    tree = get_parse_tree(sentence)
    print(tree)
    print(get_np_items(tree))
    
    
    
    