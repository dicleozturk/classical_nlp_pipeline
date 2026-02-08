'''
Created on Nov 2, 2018

@author: dicle
'''

import sys
sys.path.append("..")



import re



def ends_with(str_, pattern):
    
    c = re.compile(pattern+"$")
    if c.search(str_):
        return True
    else:
        return False


    print(ends_with("gel..", ".."))
    
    
    c = re.compile("\.{2,}$")
    if c.search("gel.."):
        print("-")
    
    
    
    
    
    
    