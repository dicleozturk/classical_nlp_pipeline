'''
Created on Apr 25, 2017

@author: dicle
'''

import os, json
import random
import string, re
import numpy as np

from dataset import io_utils
from dataset.twitter import twitter_preprocessing


# counts the tweets in the given folder which has days as subfolders
def count_tweets(folderpath, outfolder):
    
    N = 0
    Nr = 0
    Ntr = 0
    
    days = io_utils.getfoldernames_of_dir(folderpath)
    
    print(folderpath)
    for day in days:
        
        p1 = os.path.join(folderpath, day)
        
        fnames = io_utils.getfilenames_of_dir(p1, removeextension=False)
        
        for fname in fnames:
            
            p2 = os.path.join(p1, fname)
            '''
            lines = open(p2, "r").readlines()
            nlines = len(lines)
            '''
            
            tweets = lines2tweets(p2)
            ntweets = len(tweets)
            
            tr_tweets = count_lang_tweets(tweets, lang="tr")
            ntrtweets = len(tr_tweets)
            
            plain_tweets = count_nonreply_tweets(tr_tweets)
            nptweets = len(plain_tweets)
            
            print(" ", day," / ", fname, "  # lines: ", ntweets, " # tr_tweets: ", ntrtweets, " # non-reply tweeets: ", nptweets)
            
            N += ntweets
            Nr += nptweets
            Ntr += ntrtweets
            
            
            if ntrtweets > 0:
                outpath_tr = os.path.join(outfolder, day+"_"+fname)
                json.dump(tr_tweets, open(outpath_tr, "w"))
            
            if nptweets > 0:
                outpath_nr = os.path.join(outfolder, day+"_"+fname+"-nonreply")
                json.dump(plain_tweets, open(outpath_nr, "w"))
    
    return N, Ntr, Nr
            
            

def lines2tweets(filepath):
    
    lines = open(filepath, "r").readlines()
    
    lines = [line.strip() for line in lines]
    
    tweets = [json.loads(line) for line in lines]
    
    return tweets

     
def count_lang_tweets(tweets, lang="tr"):    

    lang_tweets = []

    for tweet in tweets:
        
        tlang = tweet["gnip"]["language"]["value"]
        if tlang == lang:
            lang_tweets.append(tweet)
    
    return lang_tweets
    

def count_nonreply_tweets(tweets):
    
    plain_tweets = []
    
    for tweet in tweets:
        
        keys = list(tweet.keys())
        
        if "inReplyTo" not in keys:
            plain_tweets.append(tweet)

    return plain_tweets



def _sample_N_tweets(folderpath, N, filtrate=None, keywords=None):

    print(folderpath)
    fnames = io_utils.getfilenames_of_dir(folderpath, removeextension=False)
    
    fnames = [i for i in fnames if i.endswith("-nonreply")]

    all_tweets = []
    for fname in fnames:
        
        p = os.path.join(folderpath, fname)
        tweets = json.load(open(p, "r"))
        all_tweets.extend(tweets)
        #print(fname, len(tweets), len(all_tweets))
    
    if filtrate and keywords:
        all_tweets = filtrate(keywords, all_tweets)
    
    random.shuffle(all_tweets)
    print(len(all_tweets), N)
    selected_tweets = random.sample(all_tweets, min(len(all_tweets), N))
    return selected_tweets




def ignore_containing(keywords, tweets):
    
    filtrated_tweets = []
    for tweet in tweets:
        _text = tweet["body"]
        
        accept = True
        for keyword in keywords:
            if re.search(keyword, _text, re.IGNORECASE):
                accept = False
                pass
        
        if accept:
            filtrated_tweets.append(tweet)         
        
    
    return filtrated_tweets

def select_tweets():
    
    '''
    sep = ";"
    inroot = "<PATH>"
    outroot = "<PATH>"
    #folders = ["tr_201301", "tr_201302", "tr_201303"]
    folders = ["tr_201302", "tr_201303", "tr_201301"]
    N = 700
    '''
    sep = "\t"
    inroot = "<PATH>"
    outroot = "<PATH>"
    #folders = ["tr_201301", "tr_201302", "tr_201303"]
    folders = ["tr_201302", "tr_201303", "tr_201301"]
    N = 10000
    
    
    stweets = []
    for folder in folders:
        fpath = os.path.join(inroot, folder)
        tweets = _sample_N_tweets(fpath, N, ignore_containing, keywords=["ttnet_muzik", "ttnet müzik", "I'm at"])
        print(" ", len(tweets))
        stweets.extend(tweets)
    

    rows = []
    print("stweets: ", len(stweets))
    for tweet in stweets:
        _id = tweet["object"]["id"]
        _link = tweet["object"]["link"]
        _text = tweet["body"]
        
        #_text = _text.replace(";", ",,,")   # replace semicolon
        # replace url ??
        _text = twitter_preprocessing._clean_tweets2([_text])[0]  
        rows.append({"id" : _id,
                     "link" : _link,
                     "body" : _text})
    
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.shape)
    
    df = df.drop_duplicates()   # 2- remove duplicates
    not_empty = lambda x : True if len(x.strip()) > 0 else False
    df = df[df["body"].apply(not_empty)]   # 3- clean empty instances
    
    outpath = os.path.join(outroot, 
                           str(df.shape[0])+"tweets_"+str(random.choice(range(100)))+"".join(random.sample(string.ascii_letters, 4))+".csv")  
    
    
    
    df.to_csv(outpath, sep=sep, index=False)




def tweets2csv(tweets):
    
    header = ["id", "link", "body"]
    
    rows = []
        
    for tweet in tweets:
        _id = tweet["object"]["id"]
        _link = tweet["object"]["link"]
        _text = tweet["body"]
        rows.append({"id" : _id,
                     "link" : _link,
                     "body" : _text})
    
    import pandas as pd
    df = pd.DataFrame(rows)
    return df





