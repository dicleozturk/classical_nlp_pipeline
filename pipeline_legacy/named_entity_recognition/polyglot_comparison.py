'''
Created on Jul 14, 2017

@author: dicle
'''

import os


from named_entity_recognition import ner_utils, polyglot_NE_tagger

from dataset import io_utils

from named_entity_recognition.ner_dataset.rawdata_to_brat.lines_to_ann import lines_to_ann
import named_entity_recognition.polyglot_NE_tagger as pner
from named_entity_recognition.ner_dataset.brat_to_conll import convert_brat_to_conll



'''
# intxtpath: path to the txt file having one sentence in each line and the sentences are space-sep for its tokens.
def polyglot_tagged_sentences_to_conll(intxtpath, out_lines_path, out_conll_path, lang="tr"):
    
    inf = open(intxtpath, "r")
    sentences = inf.readlines()
    sentences = [s.strip() for s in sentences]
    
    ptagged_sentences = [pner.get_sentence_entities(sentence, lang, biomark=True) for sentence in sentences]
    
    tagged_lines = ner_output_to_line_tags(sentences, ptagged_sentences, docid="X")
    
    if out_lines_path:
        io_utils.write_lines(tagged_lines, out_lines_path)
    

    conll_lines = wikiset_to_conll.wiki_to_conll(tagged_lines)
    if out_conll_path:
        io_utils.write_lines(conll_lines, out_conll_path)
'''

# intxtpath: path to the txt file having one sentence in each line and the sentences are space-sep for its tokens.
def polyglot_tagged_sentences_to_ann(intxtpath, out_lines_path, out_conll_path, lang="tr"):
    
    inf = open(intxtpath, "r")
    sentences = inf.readlines()
    sentences = [s.strip() for s in sentences]
    
    ptagged_sentences = [pner.get_sentence_entities(sentence, lang, biomark=True) for sentence in sentences]
    
    tagged_lines = ner_utils.ner_output_to_line_tags(sentences, ptagged_sentences, docid="X")
    
    if out_lines_path:
        io_utils.write_lines(tagged_lines, out_lines_path)
    
    print(intxtpath)
    
    ann_lines = lines_to_ann(tagged_lines)
    if out_conll_path:
        io_utils.write_lines(ann_lines, out_conll_path)
        
        

def dataset_polyglot_to_ann():    
    '''    
    infolder = "<PATH>"
    fnames = ["test_text_00090.txt"]
    outfolder = "<PATH>"
    '''
    
    infolder = "<PATH>"
    allfnames = io_utils.getfilenames_of_dir(infolder, removeextension=False)
    txtfiles = [fname for fname in allfnames if fname.endswith(".txt")]
    
    outfolder = "<PATH>"
    
    lang = "tr"
    
    
    
    for fname in txtfiles:
        inpath = os.path.join(infolder, fname)
        
        
        #text = io_utils.readtxtfile(inpath)
        #outlinepath = os.path.join(outfolder, "tagged_lines_"+fname)
        #outconllpath = os.path.join(outfolder, "conll_"+fname)       
        #ner_utils.polyglot_tagged_sentences_to_conll(inpath, outlinepath, outconllpath, lang=lang)
        
        annfolder = io_utils.ensure_dir(os.path.join(outfolder, "polyglot_ann2"))
        outannpath = os.path.join(annfolder, fname[:-4]+".ann")
        polyglot_tagged_sentences_to_ann(inpath, None, outannpath)
        
        txtcopypath = os.path.join(annfolder, fname)
        import shutil
        shutil.copy2(inpath, txtcopypath)
        
        #tagged_sentence_list = get_tagged_sentences(text)
        #io_utils.write_lines(tagged_sentence_list, outpath)                   
    
    

def dataset_ann_to_conll():
    
    annfolder = "<PATH>"
    conllfolder = "<PATH>"
    conllpath = os.path.join(conllfolder, "annotation1_fullset-test.conll")
    
    convert_brat_to_conll.brat_to_conll2(annfolder, conllpath)




###########################  polyglot evaluation ##################


# tag the texts in txt files in the given folder; output word list
def tag_texts_with_polyglot(infolder, lang="tr"):    
    
    fnames = io_utils.getfilenames_of_dir(infolder, removeextension=False)
    fnames = [i for i in fnames if i.endswith(".txt")]
    
    all_tagged_words = []
    for fname in fnames:
        inp = os.path.join(infolder, fname)
        txt = open(inp, "r").read()
        tagged_words = polyglot_NE_tagger.get_word_tags(txt, lang=lang, biomark=True)  # [(word, BIO NE Tag)]
        all_tagged_words.extend(tagged_words)
    
    return all_tagged_words


# conll_filepath has the gold labels
def get_text_true_tags(conll_filepath):
    
    lines = open(conll_filepath, "r").readlines()

    all_tagged_words = []
    for line in lines:
        
        if len(line.strip()) > 0:
            
            pieces = line.split()
            word = pieces[0]
            correct_tag = pieces[-2]
            all_tagged_words.append((word, correct_tag))
        else:
            all_tagged_words.append(())
    
    return all_tagged_words

    #dataset_ann_to_conll()
        
    #dataset_polyglot_to_ann()

    infolder_ = "<PATH>"
    conll_path_ = "<PATH>"
    
    r = get_text_true_tags(conll_path_)
    print(r)
    
    
    
    