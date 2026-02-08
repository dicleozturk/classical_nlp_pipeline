'''
Created on Sep 28, 2017

@author: dicle
'''


import xmltodict

import xml.etree.ElementTree as ET

import os

import pandas as pd

# b3c corpus


def read_xml(xmlpath):

    tree = ET.parse(xmlpath)
    root = tree.getroot()
    
    nthreads = len(root.findall("thread"))
    
    for thread in root:
        for annotation in thread:
            for summary in annotation.findall("summary"):
                for sent in summary.findall("sent"):
                    print(sent.tag, sent.attrib, sent.text)
    
    print(nthreads)
    return root



'''
 collect intent annotations in separate xls files in one csv.
'''
def xmls_to_csv(annot_xls, corpus_xls):
    
    atree = ET.parse(annot_xls)
    aroot = atree.getroot()
    
    for thread in aroot:
        
        for annotation in thread.findall("annotation"):
            
            for i,annot in enumerate(annotation):
                print("annot ", i)
                print(list(annot))
                for labels in annot.findall("labels"):
                    for tag in labels:
                        print(tag.tag, tag.attrib)
                    


def generate_sentence_id(listno, sentid):
    
    return listno + "--" + sentid




'''  choose the labels of the most crowded annotation in terms of the number of labels '''
def xml_to_csv2(annot_xml, corpus_xml):
    
    
    with open(annot_xml) as fd:
        annotdoc = xmltodict.parse(fd.read())
        
    with open(corpus_xml) as fd:
        corpusdoc = xmltodict.parse(fd.read())
    
    athreads = annotdoc["root"]["thread"]
    
    chosen_annotations = {}
    for i,athread in enumerate(athreads):
        
        annotations = athread["annotation"]
        
        listno = athread["listno"]
        print("Thread ", i, listno)

        chosen_annot_size = 0
        chosen_annot = 0
        for j,annot in enumerate(annotations):
            labels = annot["labels"]
            
            l = []
            for x in labels.values():
                l.extend(x)
            nlabels = len(l)
            if nlabels > chosen_annot_size:
                chosen_annot = j
                chosen_annot_size = nlabels
        
        chosen_annotations[listno] = annotations[chosen_annot]
    
    
    labelsdict = {}
    for threadid, annotation in chosen_annotations.items():
        labels = annotation["labels"]
            
        print(" annotation ", j)
        print("  ", labels)
        print(labels.keys())
        
        
        for label in labels:
            iddict = labels[label]
            if type(labels[label]) != list:
                iddict = [iddict]
            print(label, labels[label])
            ids = [item["@id"] for item in iddict]
            print(ids)
            
            for sentid in ids:
                if label not in labelsdict.keys():
                    labelsdict[label] = []
                labelsdict[label].append(generate_sentence_id(threadid, sentid))
    
    print(labelsdict)
    print(labelsdict.keys())
    '''
    labelsdict = {}
    for i,athread in enumerate(athreads):
        
        annotations = athread["annotation"]
        
        listno = athread["listno"]
        print("Thread ", i, listno)

        for j,annot in enumerate(annotations):
            
            annotid = str(j)
            
            labels = annot["labels"]
            
            print(" annotation ", j)
            print("  ", labels)
            print(labels.keys())
            
            
            for label in labels:
                iddict = labels[label]
                if type(labels[label]) != list:
                    iddict = [iddict]
                print(label, labels[label])
                ids = [item["@id"] for item in iddict]
                print(ids)
                
                for sentid in ids:
                    if label not in labelsdict.keys():
                        labelsdict[label] = []
                    labelsdict[label].append(generate_sentence_id(listno, sentid))
                    
            print()
            
        print()
    
    print(labelsdict)
    print(labelsdict.keys())
    '''
    
    senttextdict = {}
    corpus = {}       
    cthreads = corpusdoc["root"]["thread"]
    for i,cthread in enumerate(cthreads):
        listno = cthread["listno"]
        emails = cthread["DOC"]
                
        thread_emails = {}
        for j,email in enumerate(emails):
            sentences = email["Text"]["Sent"]
    
            import collections
            if type(sentences) == collections.OrderedDict:
                sentences = [sentences]
                
            
            email_sentences = {}
            for k, sentence in enumerate(sentences):
                
                '''
                try:    
                    sentid = sentence["@id"]
                except:
                    print("!!!!!!!!!!!!", sentence, listno)
                '''
                sentid = sentence["@id"]
                senttext = sentence["#text"]
                
                email_sentences[sentid] = senttext
                
                senttextdict[generate_sentence_id(listno, sentid)] = senttext

            thread_emails[j+1] = email_sentences
            
        corpus[listno] = thread_emails
    
    from pprint import pprint
    #pprint(corpus)
    #pprint(senttextdict)



    lsents = []
    for i in labelsdict.values():
        lsents.extend(i)
    lsents = list(set(lsents))
    
    ssents = list(senttextdict.keys())
    commonids = [i for i in lsents if i in ssents]
    print("commonids ", len(commonids))
    print("nsents: ", len(ssents))
    
    # make csv
    sentlist = []
    label_list = list(labelsdict.keys())
    for sentid, senttext in senttextdict.items():
        sent_dict = dict.fromkeys(label_list, 0)
        sent_dict["senttext"] = senttext
        sent_dict["sentid"] = sentid
        sentlist.append(sent_dict)
    
    df = pd.DataFrame(sentlist)
    df = df.set_index(keys="sentid")
    for label, sentids in labelsdict.items():
         
        for sentid in sentids:
            df.loc[sentid, label] = 1


    for label, sentids in labelsdict.items():
        print(label, len(sentids), sum(df[label].tolist()))
    


    return df







def _get_annotation_list(annotations_xml_path):

    with open(annotations_xml_path) as fd:
        annotdoc = xmltodict.parse(fd.read())
 
    athreads = annotdoc["root"]["thread"]
    
    annotations_list = []
    for i,athread in enumerate(athreads):
        
        annotations = athread["annotation"]
        
        listno = athread["listno"]
        print("Thread ", i, listno)

        chosen_annot_size = 0
        chosen_annot = 0
        for j,annot in enumerate(annotations):
            labels = annot["labels"]
            
            l = []
            for x in labels.values():
                l.extend(x)
            nlabels = len(l)
            if nlabels > chosen_annot_size:
                chosen_annot = j
                chosen_annot_size = nlabels
        
        chosen_annotations[listno] = annotations[chosen_annot]
    
    
    labelsdict = {}
    for threadid, annotation in chosen_annotations.items():
        labels = annotation["labels"]
            
        print(" annotation ", j)
        print("  ", labels)
        print(labels.keys())
        
        
        for label in labels:
            iddict = labels[label]
            if type(labels[label]) != list:
                iddict = [iddict]
            print(label, labels[label])
            ids = [item["@id"] for item in iddict]
            print(ids)
            
            for sentid in ids:
                if label not in labelsdict.keys():
                    labelsdict[label] = []
                labelsdict[label].append(generate_sentence_id(threadid, sentid))
    
    print(labelsdict)
    print(labelsdict.keys())
    
    

def _corpus_to_sent_dict(corpus_xml_path):

    with open(corpus_xml_path) as fd:
        corpusdoc = xmltodict.parse(fd.read())

    senttextdict = {}
    corpus = {}       
    cthreads = corpusdoc["root"]["thread"]
    for i,cthread in enumerate(cthreads):
        listno = cthread["listno"]
        emails = cthread["DOC"]
                
        thread_emails = {}
        for j,email in enumerate(emails):
            sentences = email["Text"]["Sent"]
    
            import collections
            if type(sentences) == collections.OrderedDict:
                sentences = [sentences]
                
            
            email_sentences = {}
            for k, sentence in enumerate(sentences):
                
                '''
                try:    
                    sentid = sentence["@id"]
                except:
                    print("!!!!!!!!!!!!", sentence, listno)
                '''
                sentid = sentence["@id"]
                senttext = sentence["#text"]
                
                email_sentences[sentid] = senttext
                
                senttextdict[generate_sentence_id(listno, sentid)] = senttext

            thread_emails[j+1] = email_sentences
            
        corpus[listno] = thread_emails
    
    
    return senttextdict


'''
 senttextdict = {threadno--sentid : senttext}
 chosen_annotations = {threadno--sentid : annotations}
'''
def _annotations_to_df(senttextdict, annotations):
    
    labelsdict = {}
    for threadid, annotation in annotations.items():
        labels = annotation["labels"]
                    
        for label in labels:
            iddict = labels[label]
            if type(labels[label]) != list:
                iddict = [iddict]
            print(label, labels[label])
            ids = [item["@id"] for item in iddict]
            print(ids)
            
            for sentid in ids:
                if label not in labelsdict.keys():
                    labelsdict[label] = []
                labelsdict[label].append(generate_sentence_id(threadid, sentid))
    
    
    

    lsents = []
    for i in labelsdict.values():
        lsents.extend(i)
    lsents = list(set(lsents))
    
    ssents = list(senttextdict.keys())
    commonids = [i for i in lsents if i in ssents]
    print("commonids ", len(commonids))
    print("nsents: ", len(ssents))
    
    # make csv
    sentlist = []
    label_list = list(labelsdict.keys())
    for sentid, senttext in senttextdict.items():
        sent_dict = dict.fromkeys(label_list, 0)
        sent_dict["senttext"] = senttext
        sent_dict["sentid"] = sentid
        sentlist.append(sent_dict)
    
    df = pd.DataFrame(sentlist)
    df = df.set_index(keys="sentid")
    for label, sentids in labelsdict.items():
         
        for sentid in sentids:
            df.loc[sentid, label] = 1


    for label, sentids in labelsdict.items():
        print(label, len(sentids), sum(df[label].tolist()))

    return df





'''  reflect the labels of all the three annotators '''

def xml_to_csv__(annot_xml, corpus_xml):
    
    import xmltodict
    
    with open(annot_xml) as fd:
        annotdoc = xmltodict.parse(fd.read())
        
    with open(corpus_xml) as fd:
        corpusdoc = xmltodict.parse(fd.read())
    
    
    
    athreads = annotdoc["root"]["thread"]
    labelsdict = {}
    for i,athread in enumerate(athreads):
        
        annotations = athread["annotation"]
        
        listno = athread["listno"]
        print("Thread ", i, listno)

        for j,annot in enumerate(annotations):
            
            annotid = str(j)
            
            labels = annot["labels"]
            
            print(" annotation ", j)
            print("  ", labels)
            print(labels.keys())
            
            
            for label in labels:
                iddict = labels[label]
                if type(labels[label]) != list:
                    iddict = [iddict]
                print(label, labels[label])
                ids = [item["@id"] for item in iddict]
                print(ids)
                
                for sentid in ids:
                    if label not in labelsdict.keys():
                        labelsdict[label] = []
                    labelsdict[label].append(generate_sentence_id(listno, sentid))
                    
            print()
            
        print()
    
    print(labelsdict)
    print(labelsdict.keys())
    
    
    senttextdict = {}
    corpus = {}       
    cthreads = corpusdoc["root"]["thread"]
    for i,cthread in enumerate(cthreads):
        listno = cthread["listno"]
        emails = cthread["DOC"]
                
        thread_emails = {}
        for j,email in enumerate(emails):
            sentences = email["Text"]["Sent"]
    
            import collections
            if type(sentences) == collections.OrderedDict:
                sentences = [sentences]
                
            
            email_sentences = {}
            for k, sentence in enumerate(sentences):
                
                '''
                try:    
                    sentid = sentence["@id"]
                except:
                    print("!!!!!!!!!!!!", sentence, listno)
                '''
                sentid = sentence["@id"]
                senttext = sentence["#text"]
                
                email_sentences[sentid] = senttext
                
                senttextdict[generate_sentence_id(listno, sentid)] = senttext

            thread_emails[j+1] = email_sentences
            
        corpus[listno] = thread_emails
    
    from pprint import pprint
    #pprint(corpus)
    #pprint(senttextdict)



def _xml_to_csv2(annot_xml, corpus_xml):
    
    import xmltodict
    
    with open(annot_xml) as fd:
        annotdoc = xmltodict.parse(fd.read())
        
    with open(corpus_xml) as fd:
        corpusdoc = xmltodict.parse(fd.read())
    
    
    
    athreads = annotdoc["root"]["thread"]
    for i,athread in enumerate(athreads):
        
        annotations = athread["annotation"]
        
        listno = athread["listno"]
        print("Thread ", i, listno)

        for j,annot in enumerate(annotations):
            
            annotid = str(j)
            
            labels = annot["labels"]
            
            print(" annotation ", j)
            print("  ", labels)
        print()
    
    
    corpus = {}       
    cthreads = corpusdoc["root"]["thread"]
    for i,cthread in enumerate(cthreads):
        listno = cthread["listno"]
        emails = cthread["DOC"]
        
        print("Thread ", i, " nemail: ", len(emails))
        
        thread_emails = {}
        for j,email in enumerate(emails):
            print(email)
            sentences = email["Text"]["Sent"]
            
            
            
            import collections
            if type(sentences) == collections.OrderedDict:
                sentences = [sentences]
                
            print(len(sentences), sentences)   
            
            email_sentences = {}
            for k, sentence in enumerate(sentences):
                print("email ", j, "sentence ", k)
                #print(sentence," \n")
                print(type(sentence))
                '''
                try:    
                    sentid = sentence["@id"]
                except:
                    print("!!!!!!!!!!!!", sentence, listno)
                '''
                sentid = sentence["@id"]
                senttext = sentence["#text"]
                print(sentid, senttext)
                
                email_sentences[sentid] = senttext
            print()
            thread_emails[j] = email_sentences
            
        print()
        corpus["listno"] = thread_emails
    
    from pprint import pprint
    pprint(corpus)


    main2()
    
    
    
    
    
    