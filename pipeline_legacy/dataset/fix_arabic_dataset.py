'''
Created on Mar 15, 2017

@author: dicle
'''

import os
import codecs

import pandas as pd

from dataset import io_utils


###### files to csv  #####
def files_to_csv(mainfolder, outpath, fixfolder, in_encoding="utf-8"):

    textcol = "text"
    catcol = "polarity"
    other = "domain"

    labels = io_utils.getfoldernames_of_dir(mainfolder)
    
    rows  = []
    
    for label in labels:
        p1 = os.path.join(mainfolder, label)
        fnames = io_utils.getfilenames_of_dir(p1, removeextension=False)

        print("Reading in ", label)

        for fname in fnames:
            
            print(" ", fname)          
            p2 = os.path.join(p1, fname)
                
            try:
                f = codecs.open(p2, "r", encoding=in_encoding) 
                text = f.read()
            
            except UnicodeDecodeError:
                f = codecs.open(p2, "r", encoding="cp1256")
                text = f.read()
                f2 = codecs.open(os.path.join(io_utils.ensure_dir(os.path.join(fixfolder, label)), fname), "w", encoding="utf")
                f2.write(text)
            
            text = text.strip()
            
            row = {textcol : text, catcol : label, other : fname}
            rows.append(row)
            
            f.close()
    
    df = pd.DataFrame(rows)
    df = df.sample(frac=1).reset_index(drop=True)
    
    if outpath:
        df.to_csv(outpath, sep="\t", index=False)
    return df

####### large text cat. corpus  ####

def fix_file_encodings(inmainfolder, outmainfolder,
                       in_ext="html", in_encoding="cp1256",                       
                       out_ext="txt", out_encoding="utf8"):
    
    subfolders = io_utils.getfoldernames_of_dir(inmainfolder)
    
    for subf in subfolders:
        
        p1 = os.path.join(inmainfolder, subf)
        fnames = io_utils.getfilenames_of_dir(p1, removeextension=True)

        o1 = io_utils.ensure_dir(os.path.join(outmainfolder, subf))
        print("In ", subf)
        for fname in fnames:
            
            p2 = os.path.join(p1, fname+"."+in_ext)
            o2 = os.path.join(o1, fname+"."+out_ext)
            fix_file(p2, o2, in_encoding, out_encoding)
            '''
            infile = codecs.open(p2, "r", encoding=in_encoding)
            text = infile.read()
            o2 = os.path.join(o1, fname+"."+out_ext)
            outfile = codecs.open(o2, "w", encoding=out_encoding)
            outfile.write(text)
            infile.close()
            outfile.close()
            '''
            
            
        print("Finished..\n")
    
    print("Done.")


def fix_file(inpath, outpath, in_encoding="cp1256", out_encoding="utf-8"):

    infile = codecs.open(inpath, "r", encoding=in_encoding)
    text = infile.read()
    
    outfile = codecs.open(outpath, "w", encoding=out_encoding)
    outfile.write(text)
    infile.close()
    outfile.close()
    


def run_fix_files():
    
    infolder = "<PATH>"
    infile = "negative189.txt"
    outfolder = "<PATH>"
    fix_file(os.path.join(infolder, infile), os.path.join(outfolder, infile))


def run_csv():
    '''
    mainfolder = "<PATH>"
    outpath = "<PATH>"
    '''
    mainfolder = "<PATH>"
    outpath = os.path.join(mainfolder, "ar_polar-moviereviewsOCAcorpus.csv")
    fix_folder = io_utils.ensure_dir(os.path.join(mainfolder, "fix_files"))
    files_to_csv(mainfolder, outpath, fix_folder)


    # fix one file
    #run_fix_files()
    
    
    
    # collect polar tweets in csv
    run_csv()
    
    
    
    '''
    # fix encoding of the text cat. corpus
    f1 = "<PATH>"
    f2 = "<PATH>"
    fix_file_encodings(f1, f2)
    '''
    