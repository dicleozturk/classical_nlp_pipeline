'''
Created on Apr 2, 2019

@author: dicle
'''

import re
import codecs
import os

from dataset import io_utils


'''
 convert dataset from Quaero News (http://catalogue.elra.info/en-us/repository/browse/ELRA-S0349/)

'''




def in_marked_to_brat(text):
    
    pattern = "(<(?P<label>(org|loc|pers|prod)\.(\w+\.?)+)>.*?<\/(?P=label)>)"
    extracts = re.findall(pattern, text)
    
    pieces = []
    for e in extracts:
        entity1 = e[0]
        marker1 = e[1]
        entity_str = " ".join([w for w in entity1.split() if "<" not in w])
        
        label = ""
        if re.search("^pers\.", marker1):
            label = "PER"
        elif re.search("^org\.", marker1) or re.search("prod\.((?=[^(art)])\w)", marker1) \
            or re.search("^<pers.coll> <demonym>", marker1):
            label = "ORG"
        elif re.search("^loc\.", marker1):
            label = "LOC"
        elif re.search("prod\.art", marker1):
            label = "MISC"

        pieces.append((entity_str, label))
    
    return pieces


def in_marked_to_brat2(text):
    
    pos = 0
    
    entity_items = []
    new_text = ""
    entity_no = 1
    
    while pos < len(text):
        
        current_text = text[pos:]
       
        re_search = re.search("<(?P<label>(org|loc|pers|prod)\.(\w+\.?)+)>", current_text)    #("<(pers|prod|loc|org)", current_text)
        if not re_search:
            remaining_text = " ".join([w for w in current_text.split() if "<" not in w])
            new_text = new_text + remaining_text
            break
        marker = re_search.group()
        marker_bpos = re_search.span()[0]
        marker_epos = re_search.span()[1]
        
        print(marker)
        
        # take the text before the entity marker
        unmarked_text1 = " ".join([w for w in current_text[:marker_bpos].split() if "<" not in w])
        #print(unmarked_text1)
        new_text = new_text + unmarked_text1 + " "
        
        # take the text surrounded by entity marker
        end_marker = "<\/" + marker[1:]
        end_pos1 = re.search(end_marker, current_text[marker_epos:]).span()[0] + marker_epos
        end_pos2 = re.search(end_marker, current_text[marker_epos:]).span()[1] + marker_epos
        print(end_pos1, end_pos2)
        
        etext = current_text[marker_epos:end_pos1]
        entity_text = " ".join([w for w in etext.split() if "<" not in w])
        entity_bpos = len(new_text) 
        entity_epos = entity_bpos + len(entity_text)
        
        # detect the label
        label = ""
        if re.search("pers\.", marker):
            label = "PER"
            
            #re.search("prod\.((?=[^(art)])\w)", marker) 
        elif re.search("org\.", marker) or re.search("prod\.((?!art)\w)", marker) \
            or re.search("^<pers.coll> <demonym>", marker):
            label = "ORG"
        elif re.search("loc\.", marker):
            label = "LOC"
        elif re.search("prod\.art", marker):
            label = "MISC"
         
        
        entity_items.append(("T"+str(entity_no), label, entity_bpos, entity_epos, entity_text))
        entity_no += 1
        
        new_text = new_text + entity_text + " "

        old_pos = pos
        pos = pos + end_pos2 #+ pos #+ len(end_marker)
        print(pos, text[old_pos:pos])
        #print(text[pos:pos+35])

    return new_text, entity_items

def read_txt(path):
    f = codecs.open(path, encoding="ISO-8859-15")
    text = f.read()
    return text
    

def find_types(text):
    
    label_types = re.findall("(<(\w+\.)+\w+>)", text)
    
    label_types = [i for i,_ in label_types]
    label_types = list(set(label_types))
    
    ltypes2 = re.findall("(<\w+\.?>)", text)
    ltypes2 = list(set(ltypes2))
    
    types = label_types + ltypes2
    types = sorted(types)
    
    return types


def extract_outer_entities(text, types=["loc", "pers", "prod", "org"]):
    
    all_lspans = []
    for label in types:
        #pattern = "(<" + label +"\.(\w+\.?)+>.+<\/" + label +"\.(\w+\.?)>)"
        pattern = "(<(?P<label>" + label + "\.(\w+\.?)+)>.*?<\/(?P=label)>)"
        lspans = re.findall(pattern, text)
        lspans = [i[0] for i in lspans]
        lspans = list(set(lspans))
        all_lspans.extend(lspans)
    
    all_lspans.sort()
    return all_lspans



# remove markers
def list_entities(marked_entities):
    
    unmarked_entities = []
    for me in marked_entities:
        words = me.split()
        words2 = [w for w in words if "<" not in w]
        unmarked_entity = " ".join(words2)
        unmarked_entities.append(unmarked_entity)
    
    return unmarked_entities
    


# infolder has .ne extended annotations in the text. we extract plain texts and annotations as well as their positions and write them as plain text and ann files, separately in outfolder.
def inannotations_to_brat(infolder, outfolder):  
    
    fnames = io_utils.getfilenames_of_dir(infolder, removeextension=False)
    fnames = [s for s in fnames if s.endswith(".ne")]
    
    for fname in fnames:
        
        inp = os.path.join(infolder, fname)
        annotation_txt = read_txt(inp)
        plaintxt, entitylist = in_marked_to_brat2(annotation_txt)
        
        fname2 = fname[:-3]
        open(os.path.join(outfolder, fname2+".txt"), "w").write(plaintxt)
        annotations = "\n".join(["\t".join([str(s) for s in entity_pieces]) for entity_pieces in entitylist])
        open(os.path.join(outfolder, fname2+".ann"), "w").write(annotations)
    
    print("finished")
    
    
    text = """ <pers.ind> <name.first> Patricia </name.first> <name.last> Martin </name.last> </pers.ind> , que voici , que voilà !
oh , bonjour <pers.ind> <name.first> Nicolas </name.first> <name.last> Stoufflet </name.last> </pers.ind> .
 <prod.media><name> France Inter </name></prod.media> , <time.hour.abs> <val> 7 </val> <unit> heures </unit> </time.hour.abs> .
le journal , <pers.ind> <name.first> Simon </name.first> <name.last> Tivolle </name.last> </pers.ind> .
bonjour ! <time.date.abs> <week> lundi </week> <day> 7 </day> <month> décembre </month> </time.date.abs> ."""


    text = """ vous êtes avec <pers.ind> <name.first> Patricia </name.first> <name.last> Martin </name.last> </pers.ind> sur <prod.media> <name> France Inter </name> </prod.media> , il est <time.hour.abs> <val> 7 </val> <unit> heures </unit> </time.hour.abs> .
le journal , <pers.ind> <name.first> Simon </name.first> <name.last> Tivolle </name.last> </pers.ind> .
bonjour <time.date.abs> <week> mardi </week> <day> 8 </day> <month> décembre </month> </time.date.abs> , les <func.coll> <kind> contrôleurs </kind> de la <org.ent> <name> SNCF </name> </org.ent> </func.coll> décident <time.date.rel> <name> aujourd'hui </name> </time.date.rel> s' ils poursuivent leur mouvement de grève qui dure <time.date.rel> <time-modifier> depuis </time-modifier> <amount> <val> 12 </val> <unit> jours </unit> </amount> </time.date.rel> .
la direction leur promet qu' ils seront plus nombreux <time.date.rel> <time-modifier> l' </time-modifier> <name> année </name> <time-modifier> prochaine </time-modifier> </time.date.rel> , et que la question des effectifs va être négociée immédiatement dans les régions .
le <func.ind> <kind> préfet </kind> de la <loc.adm.reg> <kind> région </kind> <name> Provence-Alpes-Côte d' Azur </name> </loc.adm.reg> </func.ind> annonce des réponses concrètes aux chômeurs <time.date.rel> <time-modifier> avant </time-modifier> <name> Noël </name> </time.date.rel> .
il y a eu des incidents <time.date.rel> <name> hier </name> </time.date.rel> lorsque les <func.coll> <kind> forces de l' ordre </kind> </func.coll> ont évacué , les <loc.fac> <kind> antennes </kind> <org.ent><name> Assedic </name></org.ent> </loc.fac> de <loc.adm.town><name> Marseille </name></loc.adm.town> qui avaient été occupées ."""


    from pprint import pprint
    txt, entities = in_marked_to_brat2(text)
    pprint(txt)
    pprint(entities)
    
    for unit in entities:
        print(unit[1],unit[4],txt[unit[2]:unit[3]])

    '''
    text = """ <pers.ind> <name.first> Patricia </name.first> <name.last> Martin </name.last> </pers.ind> , que voici , que voilà !\noh , bonjour <pers.ind> <name.first> Nicolas </name.first> <name.last> Stoufflet </name.last> </pers.ind> .\n <prod.media><name> France Inter </name></prod.media> , <time.hour.abs> <val> 7 </val> <unit> heures </unit> </time.hour.abs> .\nle journal , <pers.ind> <name.first> Simon </name.first> <name.last> Tivolle </name.last> </pers.ind> .\nbonjour ! <time.date.abs> <week> lundi </week> <day> 7 </day> <month> décembre </month> </time.date.abs> .\n<amount> <val> deux </val> <object> incendies </object> </amount> <time.hour.rel> <time-modifier> cette </time-modifier> <name> nuit </name> </time.hour.rel> en <loc.adm.reg> <kind> région </kind> <demonym> parisienne </demonym> </loc.adm.reg> , dans une maison de retraite de <loc.adm.town><name> Livry-Gargan </name></loc.adm.town> en <loc.adm.reg><name> Seine-Saint-Denis </name></loc.adm.reg> , <amount> <val> 7 </val> <object> personnes </object> </amount> ont péri dans les flammes .\net puis dans le <loc.adm.town> <name> neuvième </name> <kind> arrondissement </kind> de <name> Paris </name> </loc.adm.town> , le feu a pris dans un immeuble d' habitation et <amount> <val> 3 </val> <object> personnes </object> </amount> ont été tuées ."""
    new_text = in_marked_to_brat2(text)
    
    print(new_text)
    '''  
    
    
    

    