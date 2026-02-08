'''
Created on Apr 26, 2017

@author: dicle
'''

import re
import emoji
import numpy as np
import pandas as pd
import codecs, os


# remove emojis
def remove_emojis(text):
    emojis = list(emoji.UNICODE_EMOJI.keys())
    return "".join([c for c in text if c not in emojis])
    




# Removes hashtags, mentions, links
# from https://github.com/mertkahyaoglu/twitter-sentiment-analysis/blob/master/utils.py
def __cleanTweets(tweets):
    clean_data = []
    for tweet in tweets:
        item = ' '.join(word.lower() for word in tweet.split() \
            if not word.startswith('#') and \
               not word.startswith('@') and \
               not word.startswith('http') and \
               not word.startswith('RT'))
        if item == "" or item == "RT":
            continue
        clean_data.append(item)
    return clean_data



# Replaces links and mentions (usernames) with constants; removes rt's.
# Adapted from https://github.com/mertkahyaoglu/twitter-sentiment-analysis/blob/master/utils.py
def _clean_tweets(tweets):
    clean_tweets = []
    for tweet in tweets:
        
        tweet = re.compile("RT \@.+:").sub("", tweet).strip()
        
        words = tweet.split()
        
        newwords = []
        for word in words:
            if word.startswith("@"):
                word = "@<USER>"
            elif word.startswith("http"):
                word = "<URL>"
            newwords.append(word)

        tweet = " ".join(newwords)
        tweet = tweet.strip()
        clean_tweets.append(tweet)
        
        
        # todo: 1) via @user  2) @user aracılığıyla       
        
    return clean_tweets

# only for embeddings, replacement symbol is different
def _clean_tweets2(tweets):
    clean_tweets = []
    for tweet in tweets:
        
        tweet = re.compile("RT \@.+:").sub("", tweet).strip()
        
        words = tweet.split()
        
        newwords = []
        for word in words:
            if word.startswith("@"):
                word = "<@USER>"  #"@<USER>"
            elif word.startswith("http") or word.startswith("//t.co"):
                word = "<URL>"  #"*URL*"
            newwords.append(word)

        tweet = " ".join(newwords)
        tweet = tweet.strip()
        clean_tweets.append(tweet)
        
        
        # todo: 1) via @user  2) @user aracılığıyla       
        
    return clean_tweets


def preprocess_twitter_dataset(_tweets, _labels):

    tweets1 = _clean_tweets(_tweets) # 1- clean twitter symbols
    
    labels = _labels
    if labels is None:
        labels = [None]*len(tweets1)
    
    df = pd.DataFrame(data=np.array([tweets1, labels]).T, columns=["text", "label"])
    
    
    df = df.drop_duplicates()   # 2- remove duplicates
    not_empty = lambda x : True if len(x.strip()) > 0 else False
    df = df[df["text"].apply(not_empty)]   # 3- clean empty instances
    
    # todo: replace coooool+ with coool+
    
    tweets = df["text"].tolist()
    labels = df["label"].tolist()
    
    if _labels is None:
        labels = _labels
    
    return tweets, labels


# partition large twitter dataset & preprocess
# 20M tweets
def partition_large_file(filepath,
                         outfolder):
    
    i = 0
    
    f = codecs.open(filepath, encoding="utf-8")  #"iso-8859-9")
    break_ = False
    while True:
    
        lines = []
        for _ in range(1000000):
            line = f.readline()
            print(line)
            if line == "":
                print(" break")
                break_ = True
                break
            lines.append(line)
        
        i += 1   
        print(i)     
        lines = [i.strip() for i in lines if len(i.strip()) > 0]
        lines = [i for i in lines if "..." not in i]
        outf = codecs.open(os.path.join(outfolder, str(i)+".txt"), "w", encoding="utf-8")
        outf.write("\n".join(lines))
        #open(os.path.join(outfolder, str(i)+".txt"), "w").write("\n".join(lines))
        if break_:
            break





    
    
    
    
    