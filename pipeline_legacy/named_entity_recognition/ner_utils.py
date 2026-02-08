'''
Created on Jul 14, 2017

@author: dicle
'''

import sys
sys.path.append("..")

import re, os


from named_entity_recognition import CONS_TAGS
from named_entity_recognition.ner_dataset.rawdata_to_brat import wikiset_to_conll
from dataset import io_utils

# pner_entities = [(token*, tag)] : token might be multiple words. tag has no position marker.
def assign_bio_markers(pner_entities):    
    begin_mark = "B"
    in_mark = "I"
    
    new_entities = []
    for entity, tag in pner_entities:
        items = entity.split()
        if len(items) > 1:
            beginner_entity = items[0]
            beginner_tag = begin_mark + "-" + tag
            new_entities.append((beginner_entity, beginner_tag))
            for item in items[1:]:
                insider_entity = item
                insider_tag = in_mark + "-" + tag
                new_entities.append((insider_entity, insider_tag))
        else:
            new_entity = entity
            new_tag = begin_mark + "-" + tag
            new_entities.append((new_entity, new_tag))
    
    return new_entities



'''
the same test data used in the acceptable neuroner model.
 test folder in the dataset folder of the neuroner model contains both docs and ann-version tags.
'''



'''
# input text has a sentence in each line.
# returns sentence list = [X tab ..tag_i tag_j.. tab .. token_i token_i+1..]
def get_tagged_sentences(text, lang="tr"):
    
    sentences = text.split(os.linesep)
    #stokens = [s.split() for s in sentences]
    
    print(sentences)
    
    default_tag = CONS_TAGS.TAG_OUTSIDE
    outlines = []
    for sent in sentences:
        
        pner_entities = pner.get_sentence_entities(sent, lang=lang, clean_tags=True)  # @TODO pass this in param.s. other NERs can also be called.
        tokens = sent.split()
        print(sent, " - ", pner_entities)
        
        # assign B - O marks
        pner_entities = assign_bio_markers(pner_entities)
        tokentags1 = [(token, default_tag) for token in tokens]
        for ptoken, ptag in pner_entities:
            
            pindex = get_token_index(ptoken, tokens)
            if pindex > -1:
                tokentags1[pindex] = (ptoken, ptag)
            
        docid = "X"
        entity_part = " ".join([tag for _,tag in tokentags1])
        token_part = " ".join([token for token,_ in tokentags1])
        line = "\t".join([docid, entity_part, token_part])
        outlines.append(line)
    
    return outlines
'''


def get_token_index(token, tokens):
    
    try:
        index_ = tokens.index(token)
    except ValueError:
        index_ = -1
    finally:
        return index_
    


# tokenized_sentences = [tokens] : tokens is a string having tokens sep by space.
# sentence_entities = [[(token-j_sent-i, tag_token-j)]] : token is single word; tag is bio-marked NE tag.
# returns sentence list = [X tab ..tag_i tag_j.. tab .. token_i token_i+1..]
def ner_output_to_line_tags(tokenized_sentences, sentence_entities, docid="X"):
    
    if len(tokenized_sentences) != len(sentence_entities):
        print("Error:  # of plain sentences and # of tagged sentences do not match.")
        return
    
    default_tag = CONS_TAGS.TAG_OUTSIDE
    
    outlines = []
    for sentence, tagged_tokens in zip(tokenized_sentences, sentence_entities):
        
        tokens = sentence.split()
        tokentags1 = [(token, default_tag) for token in tokens]
        for ptoken, ptag in tagged_tokens:
            
            pindex = get_token_index(ptoken, tokens)
            if pindex > -1:
                tokentags1[pindex] = (ptoken, ptag)
            
        
        # checked num of nes equality: not necessarily match because polyglot might tag in-token parts (Erzincan-Ankara single token. polyglot tags them separately and these aren't found in our tag list.)
        '''
        x = [tag for _,tag in tokentags1 if tag != default_tag]
        nout = len(x)
        npoly = len(tagged_tokens)
        print(nout, npoly, "  ", sentence.strip())
        print(tagged_tokens, "  ---- ", tokentags1)
        print()
        '''
                    
        docid = docid
        entity_part = " ".join([tag for _,tag in tokentags1])
        token_part = " ".join([token for token,_ in tokentags1])
        line = "\t".join([docid, entity_part, token_part])
        outlines.append(line)

    return outlines


# remove apostrophe
# remove subset NEs
# return list of NEs
def _filter_ann_NEs(annpath):
    
    NEs = ann_to_NElist(annpath)
    
    # apostrophe
    NEs = [re.split("[\`\"']",e)[0] for e in NEs]
        
    unique_NEs = sorted(list(set(NEs)))
    n = len(unique_NEs)
    
    final_NEs = unique_NEs.copy()
    
    print("" in unique_NEs)
    print("" in NEs)
    print(NEs)
    
    removables = []
    for i in range(n):
        others = unique_NEs[0:i] + unique_NEs[i+1:]
        current_ne = unique_NEs[i]
        #print("others: ", others)
        for other_ne in others:
            if current_ne in other_ne:   # and not_equal
                # len(current_ne) < len(other_ne)
                #other_i = unique_NEs.index(other_ne)
                print(i, current_ne, " - ", other_ne, "#")
                removables.append(current_ne)
    
    final_NEs = [i for i in final_NEs if i not in removables]
    return final_NEs



# remove apostrophe
# remove subset NEs
# return list of NEs
def filter_ann_NEs2(annpath):
    
    f = open(annpath, "r")
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    
    ne_tags = []
    for line in lines:
        items = line.split("\t")
        items = [i.strip() for i in items]
        entity = items[-1]
        entity = re.split("[\`\"']",entity)[0]
        tagpart = items[1].strip()
        tag = tagpart.split()[0]
        ne_tags.append((entity.strip(), tag.strip()))
    

        
    unique_pairs = sorted(list(set(ne_tags)))
    n = len(unique_pairs)
    
    final_NEs = unique_pairs.copy()
       
    removables = []
    for i in range(n):
        others = unique_pairs[0:i] + unique_pairs[i+1:]
        current_pair = unique_pairs[i]
        current_ne = current_pair[0]
        current_tag = current_pair[1]
        #print("others: ", others)
        for other_ne, other_tag in others:
            
            if current_ne in other_ne and current_tag == other_tag:   # and not_equal
                # len(current_ne) < len(other_ne)
                #other_i = unique_NEs.index(other_ne)
                print(i, current_ne, " - ", other_ne, "#")
                removables.append((current_ne, current_tag))
    
    final_NEs = [i for i in final_NEs if i not in removables]
    return final_NEs


    
def ann_to_NElist(annpath):
    
    f = open(annpath, "r")
    lines = f.readlines()
    
    NEs = [line.split("\t")[-1].strip() for line in lines]
    
    return NEs


def bulk_filter_NEs(infolder, outfolder):

    fnames = io_utils.getfilenames_of_dir(infolder, removeextension=True)
    inext = ".ann"
    outext = ".entities"
    
    
    for fname in fnames:
        inp = os.path.join(infolder, fname+inext)
        filtered_NEs = filter_ann_NEs2(inp)
        outp = os.path.join(outfolder, fname+outext)
        outputlines = ["\t".join(list(ne_pair)) for ne_pair in filtered_NEs]
        io_utils.write_lines(outputlines, outp, linesep="\n")


# collectionpath has txt files, which have one sentence in each line and the sentences are tokenized; the tokens are space-sep.
def get_collection_stats(collectionpath):
    
    from glob import glob
    
    txtfnames = glob(os.path.join(collectionpath, "*.txt"))
    ndocs = len(txtfnames)
    
    nsentences = 0
    ntokens = 0
    for fname in txtfnames:
        txtpath = os.path.join(collectionpath, fname)
        lines = open(txtpath, "r").readlines()
        lines = [line.strip() for line in lines]
        
        nsentences += len(lines)
        
        ntokens1 = 0
        for line in lines:
            ntokens1 += len(line.split())
        ntokens += ntokens1
    
    print()
    print(collectionpath)
    print("\tndocs: ", ndocs, "\tnsent: ", nsentences, "\tntokens: ", ntokens)
    print()
    
  

    '''
    annpath = "<PATH>"
    l0 = ann_to_NElist(annpath)
    l0.sort()
    l = filter_ann_NEs2(annpath)
    print("given: ", l0)
    print("filtered: ", l)
    '''
    
    '''
    infolder = "<PATH>"
    outfolder = "<PATH>"
    bulk_filter_NEs(infolder, outfolder)
    '''
    
    
    
    cpaths = ["<PATH>",
              "<PATH>",
              "<PATH>",
              "<PATH>"]
    
    for cpath in cpaths:
        get_collection_stats(cpath)
    
    
    
    
    

   

