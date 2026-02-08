'''
Created on Jun 1, 2017

@author: dicle
'''

import os
import re


import sys
sys.path.append("..")



# @todo get dates separately

def extract_numbers(text):
    
    num_pattern = r"(\w{0,4}\d{3,}\s*)+"
    nums = re.findall(num_pattern, text)
    nums = [n.strip() for n in nums]
    return nums

