'''
Created on Aug 28, 2018

@author: dicle
'''

import sys
sys.path.append("..")

import networkx
from sklearn.feature_extraction.text import TfidfVectorizer
from html.parser import HTMLParser
import pandas as pd

import sentence_splitting




def preprocess_text(text):

    html_parser = HTMLParser()
    text = html_parser.unescape(text)
    
    return text

def textrank_text_summarizer(text, lang = "tr", summary_ratio=0.5):
    
    text = preprocess_text(text)
    
    # sentence splitting & normalization
    sentences = sentence_splitting.text_to_sentences(text, lang)
    #sentences = normalization.normalize_corpus_tr(sentences)                        
    
    # vectorizing
    tfidf_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1))
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences).astype(float)
    
    
    # graph building & ranking
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T)
    similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
    scores = networkx.pagerank(similarity_graph)   
    
    
    ranked_sentences = sorted(((score, index) 
                                for index, score 
                                in list(scores.items())), 
                              reverse=True)

    # summary elements
    nsentences = len(sentences)
    nselection = int(summary_ratio * nsentences)
    top_sentence_indices = [ranked_sentences[index][1] 
                            for index in range(nselection)]
    top_sentence_indices.sort()
    
    summary_elements = []
    for index in top_sentence_indices:
        summary_elements.append(sentences[index])   
        
    return summary_elements   
        



This specific study stops short of saying that fatigue is the primary cause of this increased risk, but there is ample evidence to suggest this might be the case.
            It makes accidents more likely, boosts stress levels, and even causes physical pain. But the real problem is that many people just can’t afford not to do it.
According to latest International Labour Organization statistics, more than 400 million employed people worldwide work 49 or more hours per week, a sizeable proportion of the near 1.8 billion total employed people worldwide.
But wearing exhaustion like a badge of honour sets a dangerous precedent. Hustling over long hours and weekends has become a staple of start-up culture in Silicon Valley - hence, it has also filtered out to many parts of the world. 
The problem is that this 'long hours' culture likely defeats the purpose of getting more things done, or at least puts a very hefty price on doing them. 
            """ 

    
    ssents = textrank_text_summarizer(text, lang="en", summary_ratio=0.4)
    summary_text = " ".join(ssents)
    print("english")
    print(summary_text)
    
    
    
    tr_text = """Tek tek her kişide, başkalarının vereceği zarardan korunan kişisel nitelik ve eylemlere hak adını veren, hakkı “bireyin başka herkes karşısında olumlayıp, öne sürdüğü ve koruduğu bir şey olma veya bir şey yapma ya da bir şeye sahip olma özgürlüğü” olarak tanımlayan Smith’te “haklar”, “zarar” ve “ahlaki kişilik” arasında yakın bir ilişki bulunur. Çünkü ona göre, muhayyile sosyal deneyime bağlı olup, bir sosyal düzeyden diğerine farklılık gösterdiği için hakların ve ahlaki kişiliğin kendisi de bir toplum biçiminden bir başkasına farklılık gösterir. Onun sadece ahlaklılığa değil adalet ve hukuka yönelik tarihsel yaklaşımı da işte burada karşımıza çıkar. Gerçekten de insan doğasını anlamak için kullanılacak bir araç olarak “doğa durumu” düşüncesini reddeden ve insana insanlığını kazandıran şeyin kendisine tutulan sosyal ayna olduğu için insan türünün moral hayatının kaçınılmaz olarak toplumsal olduğunu öne süren Smith’e göre, sadece ahlaki özellik ve erdemlerimizle değil fakat haklarla ilgili değerlendirmelerin de sosyal bir çerçeveye oturtulması gerekir. Dört Aşamalı Sosyal Gelişme Teorisi Smith, işte bu çerçeve içinde toplumsal gelişmenin dört ayrı evresini kişilik kavramındaki genişleme ve hakların kapsamında kaydedilen ilerleme bakımından birbirinden ayırır. Onun aynı zamanda “dört evreli bir tarihsel ilerleme” anlayışından oluşan bu tarih felsefesi görüşü, Aydınlanmanın veya Aydınlanma filozoflarının tarihe bilimsel bakışlarının bir başka ifadesi olmasının dışında, ilerlemenin temelinde entelektüel gelişme ve bilimsel bilgi birikiminden ziyade iktisadi faktörlerin olduğunu öne sürmek açısından da büyük önem taşır. Benzerlerini sadece Aydınlanma filozoflarında değil, fakat Hegel, Marx ve Comte gibi 19. yüzyıl filozoflarından da görebileceğimiz bu ilerleme ya da sosyal gelişme teorisi tarihte, haklar, özgürlük, kişilik bakımından gerçek bir ilerleme olduğu varsayımına dayanır."""
    tr_sents = textrank_text_summarizer(tr_text, lang="tr", summary_ratio=0.5)
    summary_tr = " ".join(tr_sents)
    print("turkish")
    print(summary_tr)
    


def mass_summarize(incsvpath="<PATH>",
                   textcol="text",                   
                   summarycol="summary",
                   lang="tr"
                   ):
    
    df = pd.read_excel(incsvpath)  # pd.read_csv(incsvpath, sep="\t")
    
    df[summarycol] = ""
    
    for i in df.index.values:
        
        print(i)
        text = df.loc[i, textcol]
        sum_sents = textrank_text_summarizer(text, lang, summary_ratio=0.5)
        sum_sents = [s.strip() for s in sum_sents]
        summary = " ".join(sum_sents)
        df.loc[i, summarycol] = summary
        
 
    return df   
    

def turkish_service():

    while(True):
        tr_text = input("Enter the text to be summarized \n")
        if len(tr_text)>0:
            print("\n")
            tr_sents = textrank_text_summarizer(tr_text, lang="tr", summary_ratio=0.5)
            summary_tr = " ".join(tr_sents)
            print(summary_tr+"\n")
        else :
            print("What you've just entered is empty or unappropriate. \n")

def english_service():

    while(True):
        text = input("Enter the text to be summarized \n")
        if len(text)>0:

            ssents = textrank_text_summarizer(text, lang="en", summary_ratio=0.4)
            summary_text = " ".join(ssents)
            print(summary_text+"\n")
        else :
            print("What you've just entered is empty or unappropriate. \n")


    '''
    outdf = mass_summarize()
    outcsvpath = "<PATH>"
    io_utils.tocsv(outdf, outcsvpath, keepindex=False)
    '''
    
    #main()
    #
    # turkish_service()
    english_service()



    
    