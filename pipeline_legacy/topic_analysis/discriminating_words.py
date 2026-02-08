'''
Created on Jun 13, 2017

@author: dicle
'''


import sys
sys.path.append("..")

import os
import pandas as pd

# cdf = [(cl_id, term, weight)]
def get_discriminating_words_per_cat(cdf,
                                     clid_col,
                                     term_col,
                                     weight_col):
        
    clids = list(set(cdf[clid_col].tolist()))


    cl_discr_terms = []
    for clid in clids:
        
        current_df = cdf.loc[(cdf[clid_col] == clid), :]
        other_df = cdf.loc[~(cdf[clid_col] == clid), :]
        
        other_terms = other_df[term_col].tolist()
        other_terms = list(set(other_terms))

        for i in current_df.index.values:
            
            current_term = current_df.loc[i, term_col]
            if current_term not in other_terms:
                
                cl_discr_terms.append(current_df.loc[i, :].values.tolist())
    
    
    result = pd.DataFrame(data=cl_discr_terms)
    
    result = result.sort_values(by=[0, 1], ascending=False)
    return result





    cfolder = ""
    cfname = "cluster_keywords.csv"
    cpath = os.path.join(cfolder, cfname)
    cdf = pd.read_csv(cpath, sep="\t")
    
    cl_discr_terms = get_discriminating_words_per_cat(cdf, 
                                                      clid_col="cluster_id", term_col="term", weight_col="proximity")
    
    cl_discr_terms.to_csv(os.path.join(cfolder, "discriminating_words_by_cluster.csv"), sep="\t", index=False)
    
    
    print(cl_discr_terms)
    
    
    g = cl_discr_terms.groupby(0)

    keys = list(g.groups.keys())
    dfs = [(k,g.get_group(k)) for k in keys]
    print([(k, dfi.shape) for k,dfi in dfs])
    
    
    