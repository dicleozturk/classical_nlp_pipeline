'''
Created on Aug 18, 2017

@author: dicle
'''

import sys
sys.path.append("..")


import os

from dataset import io_utils



'''
wikipedia plain text dump available (as cirrus-search) at https://dumps.wikimedia.org/other/cirrussearch/current/

the files contain each article surrounded by
<doc id=XXX url=XXX title="XX">
ARTICLE
</doc>
------
read the files line by line; excluded those having doc-markers.
'''


def clean_wikipedia_dump(filepath, outpath):
    
    f = open(filepath, "r")
    lines = f.readlines()
    
    newlines = []
    
    i = 0
    nlines = len(lines)
    while(i < nlines):
        
        line = lines[i]
        if line.startswith("<doc id="):
            i += 1
        elif line.startswith("</doc>"):
            pass
        else:
            newlines.append(line)
        i += 1
    
    
    io_utils.write_lines(newlines, outpath, linesep="\n")



def clean_nested_folders(inmainfolder,
                         outmainfolder):
    
    
    subfolders = io_utils.getfoldernames_of_dir(inmainfolder)
    
    for subfolder in subfolders:
        
        inp1 = os.path.join(inmainfolder, subfolder)
        outp1 = io_utils.ensure_dir(os.path.join(outmainfolder, subfolder))
        
        fnames = io_utils.getfilenames_of_dir(inp1, removeextension=False)
        
        print("On ", inp1)
        for fname in fnames:
            
            infpath = os.path.join(inp1, fname)
            outfpath = os.path.join(outp1, fname)
            
            clean_wikipedia_dump(infpath, outfpath)
            print("  Cleaned: ", outfpath)
    




        
    