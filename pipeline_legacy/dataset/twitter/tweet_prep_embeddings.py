'''
Created on Jun 29, 2018

@author: dicle
'''

import re
import pandas as pd


import dataset.twitter.twitter_preprocessing as tweetprep

def count_tweets():
    # 1) raw count
    # 2) clean tweet keywords
    # 3) count clean texts
    
    fpath = "<PATH>"
    f = open(fpath, "r")
    tweets = f.readlines()
    print(len(tweets))

    return tweets



def read_tweets_txt(fpath):
    f = open(fpath, "r")
    tweets = f.readlines()
    return tweets


def read_tweets_csv(fpath, sep, tweetcol):
    
    df = pd.read_csv(fpath, sep=sep)
    texts = df[tweetcol].tolist()
    
    texts = [str(t).strip() for t in texts]
    texts = [t for t in texts if len(t) > 0 or t != "nan"]

    return texts




'''
cleaning: 1) tweet words 2) nwords < 4

'''
    
def clean_tweets(tweets, remove_emojis=False):
    
    threshold_wordcount = 3
    
    
    ctweets = tweetprep._clean_tweets2(tweets)
    print(len(ctweets))
    
    # 1) remove empty lines
    ctweets = [i.strip() for i in ctweets]
    ctweets = [i for i in ctweets if len(i) > 0]
    print("1) after removing empty texts: ", len(ctweets))
    
    # 2) remove duplicates
    ctweets = list(set(ctweets))
    print("2) after removing duplicates: ", len(ctweets))
    
    if remove_emojis:
        # 3) remove emojis
        ctweets = [tweetprep.remove_emojis(tweet) for tweet in ctweets]
        ctweets = list(set(ctweets))
        print("3) after removing emojis and re-removing duplicates: ", len(ctweets))
    
    
    # 4) reduce to 1 successive occurrences of @<USER>
    ctweets = [re.compile("(\<\@USER\>\s)+").sub("<@USER> ", t) for t in ctweets]
    ctweets = list(set(ctweets))
    print("3) after removing multiple @<USER> and re-removing duplicates: ", len(ctweets))
    
    
    # 5) remove tweets with less than threshold_wordcount+1 words
    ctweets = list(filter(lambda x : len(x.split()) > threshold_wordcount, ctweets))
    print("4) after removing less than 3-words tweets: ", len(ctweets))
    
    
    # 4) reduce to 1 successive occurrences of @<USER>
    ctweets = [re.compile("(\<\@USER\> aracılığıyla)").sub("", t) for t in ctweets]
    ctweets = list(set(ctweets))
    print("3) after removing multiple @<USER> and re-removing duplicates: ", len(ctweets))
    
    return ctweets




def try_cleaning():    
    tweets = ["Japonya da Honda oyuna girdi. Toyota ve Nissan ısınıyor.",
                "Japonyada oyuna honda girdi ben olsam toyota’yı oynatırdım",
                "Japonya milli takımı galeri gibi Suzuki çıkıyor Honda giriyor sagdan Toyota bindirme yapiyor",
                "@DemirelEns @SunaVarol_ Kardeşim sen 2018’de kalkıp araba yapacam dersen sana Mercedes Volkswagen götüyle güler. İn… https://t.co/Hs70rVvYSA",
                "Saatlerdir gülüyorum muhteşem espiri :D:D https://t.co/SHdlKKbDEb",
                "Toyota şu kamyoneti üretmeseydi Ortadoğu'da terör diye bir şey olmayacaktı net https://t.co/aGW04LdytM",
                "honda'dan istedigini alamayan japon hoca onun yerine toyota'yı oyuna sokacak🔜😂",
                "RT @merveincesuv: 25 GD 791 plakalı Wolksvagen Polo marka araç tarafından yol boyu takip edildim, sözlü tacize uğradım ve üstüne tehdit edi…",
                "Ne bicim Japon milli takimi lan",
                "Ne Suzuki var Ne Honda var Ne Toyota var",
                "Cakmami bunlar acaba",
                "2011 Model Volkswagen Scirocco 1.4 TSİ 150 HP Otomatik Vites Hatasız Boyasız 90 Bin Km de İlk Sahibinden 🚘🚙🚘🚙 — Dia… https://t.co/NWCrwH1NGt",
                "2012 Model Toyota Auris 1.4 D-4D (dizel) Comfort Plus 99 Bin Km de Hatasız Boyasız İlk Sahibinden 🚙🚘🚙🚘 — Diamond Mo… https://t.co/fWKoPt3Fhu",
                "@Ginaseb5 toyota da uğursuz geçen seneki le mansta son 3 dkda galibiyeti kaçırmışlar.bu ikili bir araya gelip nasıl… https://t.co/GAli9lmrCE",
                "RT @merveincesuv: 25 GD 791 plakalı Wolksvagen Polo marka araç tarafından yol boyu takip edildim, sözlü tacize uğradım ve üstüne tehdit edi…",
                "RT @BuzzeSpor: 🎟 LG PUBG TAMGAME Ligi Büyük Finali’ne 20 kişiye bilet hediye ediyoruz."]
    
    
    ctweets = tweetprep._clean_tweets(tweets)

    for c,t in zip(ctweets, tweets):
        print(c, " ---- ", t)
        print()
        


def prep_somemto_tweets():

    tweetspath = "<PATH>" 
    tweets = read_tweets_txt(tweetspath)
    
    tweets = [re.sub("@<USER>", "<@USER>", t) for t in tweets]
    ctweets = clean_tweets(tweets)

    
    ctweetspath = "<PATH>"
    f = open(ctweetspath, "w")
    f.write("\n".join(ctweets))
    



def prep_2013_tweets():
    
    csvpath = "<PATH>"
    sep = "\t"
    tweetcol = "body"
    tweets = read_tweets_csv(csvpath, sep, tweetcol)
    
    ctweets = clean_tweets(tweets)
    
    ctweetspath = "<PATH>"
    f = open(ctweetspath, "w")
    f.write("\n".join(ctweets))
    
    
    
    print()
    
    prep_2013_tweets()
    #prep_somemto_tweets()
    
    
    
    
    
    
    
    
    
    
    
    
    
    