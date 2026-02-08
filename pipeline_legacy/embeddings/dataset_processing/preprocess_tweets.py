'''
Created on Apr 13, 2018

@author: dicle
'''

import os, re
import pandas as pd
from dataset import io_utils

# This is for Py2k.  For Py3k, use http.client and urllib.parse instead, and
# use // instead of / for the division
#import http.client as httplib
import http
import urllib.parse as urlparse
import socket

def unshorten_url(url):
    parsed = urlparse.urlparse(url)
    h = http.client.HTTPConnection(parsed.netloc)
    
    try:
        h.connect()
        h.request('HEAD', parsed.path)
        response = h.getresponse()
        
        return response.getheader('Location')
    except socket.gaierror:
        return None
    
    '''
    if response.status/100 == 3 and response.getheader('Location'):
        return response.getheader('Location')
    else:
        return url
    '''
    
def read_tweets(xlspath, txtcol):
    
    df = pd.read_excel(xlspath)
    
    texts = df[txtcol].tolist()
    
    return texts


def extract_links(texts):
    
    txt_links = []
    
    for t in texts:
        links = re.findall("https?\:\S+", t)
        txt_links.append((t, links))
    
    return txt_links
        

def find_avg_length(texts):
    # by # of words
    avg_word_len = sum([len(t.split()) for t in texts]) / float(len(texts))
    
    # by str length
    avg_str_len = sum([len(t) for t in texts]) / float(len(texts))
    
    return avg_word_len, avg_str_len

    
    
    
    