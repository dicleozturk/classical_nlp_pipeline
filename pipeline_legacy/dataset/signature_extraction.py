# -*- coding: utf-8 -*-
'''
Created on Dec 1, 2016

@author: dicle
'''

import os
import pandas as pd

from talon.signature.bruteforce import extract_signature
import talon
from talon import signature
    



def extract_signatures_rb(emails):
    
    items = [extract_signature(email) for email in emails]
    bodies = [body for body,_ in items]
    signatures = [str(signature) for _,signature in items]
    return signatures


def extract_signatures_ml(emails, senders):
    items = [signature.extract(email, sender=sender) for email, sender in zip(emails, senders)]
    bodies = [body for body,_ in items]
    signatures = [str(signature) for _,signature in items]
    return signatures


def record_signatures(signatures, outfolder, fname1, fname2):
    
    distinct_signatures = list(set(signatures))
    
    open(os.path.join(outfolder, fname1), "w").write("\n".join(signatures))
    open(os.path.join(outfolder, fname2), "w").write("<imza>" + "</imza>\n<imza>".join(distinct_signatures) + "</imza>")

    return distinct_signatures




    
    