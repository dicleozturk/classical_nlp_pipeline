'''
Created on Sep 7, 2016

@author: dicle
'''


import sys
sys.path.append("..")


import os, codecs, json, shutil

import pandas as pd

#from sklearn.externals import joblib
import joblib


def merge_subfolders(subfolderspath, singlefolderpath):
    
    subfolders = getfoldernames_of_dir(subfolderspath)
    
    for subfolder in subfolders:
        
        p = os.path.join(subfolderspath, subfolder)
        copy_folder(from_folder=p, to_folder=singlefolderpath)
        print("Copied ", subfolder)
    
    


def read_json_file(filepath):
    
    content = open(filepath, "rb").read()
    content2 = content.decode("utf-8-sig").replace('\t', '\\t').replace("\x15", "ı").replace("\x0b", "\\n")
    jcontent = json.loads(content2, encoding="utf-8")
    return jcontent



def json_to_csv(jsonpath, csvpath, csvsep):

    jcontent = read_json_file(jsonpath)
    
    df = pd.DataFrame(jcontent)
    
    if csvpath:
        df.to_csv(csvpath, sep=csvsep, index=False)
    
    return df


def write_lines(lines, path, linesep=os.linesep):
    
    open(path, "w").close()
    f = open(path, "a")
    for line in lines:
        f.write(line + linesep)
    
    f.close()
    

def read_lines(path):
    
    f = open(path, "r")
    lines = f.readlines()
    return lines


def copy_folder(from_folder, to_folder):
    
    
    fnames = getfilenames_of_dir(from_folder, removeextension=False)
    
    for fname in fnames:
        shutil.copy2(os.path.join(from_folder, fname), os.path.join(to_folder, fname))  
    


# returns the names of the files and dirs in the given directory *path*
def getfilenames_of_dir(path, removeextension=True):
    files = os.listdir(path)
    filenames = []
    for fileitem in files:
        if os.path.isfile(path + os.sep + fileitem):
            if removeextension:
                filename = fileitem.split(".")[0]  # remove extension if any
            else:
                filename = fileitem
            filenames.append(filename)
        
    return filenames

def getfoldernames_of_dir(path):
    files = os.listdir(path)
    foldernames = []
    for fileitem in files:
        if os.path.isdir(path + os.sep + fileitem):
            foldernames.append(fileitem)
    return foldernames


# ensures if the directory given on *f* exists. if not creates it.
def ensure_dir(f):
    # d = os.path.dirname(f)
    if not os.path.exists(f):
        os.makedirs(f)
    return f 



def append_csv_cell_items(items, path, sep="\t"):

    with open(path, "a") as f:
        f.write(sep.join(items) + "\n")




def todisc_list(path, lst, mode="w"):
    f = codecs.open(path, mode, encoding='utf8')
  
    for item in lst:
        f.write(item + "\n")
        
    f.close()

def todisc_df_byrow(path, df, keepIndex=False, csvsep="\t"):
    
    colids = df.columns.values.tolist()
    if keepIndex:
        rowids = df.index.values.tolist()
    nr, _ = df.shape
    
    colids2 = [item + "," for colid in colids for item in colid]
    # header = csvsep.join([str(s).encode("utf-8") for s in colids])
    header = csvsep.join(colids2)
    
    if keepIndex:
        header = "\t" + header
    todisc_txt(header, path, mode="w")
    
    for i in range(nr):
        rowitems = df.iloc[i, :].tolist()
        rowstr = csvsep.join([str(s) for s in rowitems])
        if keepIndex:
            rowstr = rowids[i] + rowstr
        todisc_txt("\n" + rowstr, path, mode="a")

def readtxtfile(path):
    f = codecs.open(path, encoding='utf8')
    rawtext = f.read()
    f.close()
    return rawtext


def readtxtfile2(path):
    rawtext = open(path, "r").read()
    return rawtext

def todisc_txt(txt, path, mode="w"):
    f = codecs.open(path, mode, encoding='utf8')
    f.write(txt)
    f.close()
    
    

def readcsv(csvpath, keepindex=False, sep="\t"):
    if keepindex:
        df = pd.read_csv(csvpath, sep=sep, header=0, index_col=0, encoding='utf-8')
    else:
        df = pd.read_csv(csvpath, sep=sep, header=0, encoding='utf-8')
 
    try:
        df = df.drop('Unnamed: 1', 1)
    except:
        pass
    return df


def read_excel(filepath):
    
    df = pd.read_excel(filepath)
    return df

def tocsv(df, csvpath, keepindex=False):
    df.to_csv(csvpath, index=keepindex, header=True, sep="\t", encoding='utf-8')



def initialize_csv_file(header, path, sep="\t"):
    
    headerstr = sep.join(header)
    with open(path, "w") as f:
        f.write(headerstr + "\n")
    
    
###################
## utils

def dump_object(obj, dumpfolder, fname):
    joblib.dump(obj, os.path.join(dumpfolder, fname))


def load_object(dumpfolder, fname):
    obj = joblib.load(os.path.join(dumpfolder, fname))
    return obj

########################



    
    
