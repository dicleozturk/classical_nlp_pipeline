'''
Created on Dec 12, 2017

@author: dicle
'''


import os

import pandas as pd
'''
 - take agreement, split unagreed instances
'''



def process_agreement(df, idcol, textcol, labelcol):
    
    g = df.groupby(idcol)
    
    ids = list(g.groups.keys())
    
    unagreed = pd.DataFrame()
    agreed = pd.DataFrame()
    for id_ in ids:
        pair_df = g.get_group(id_)
        annotations = list(set(pair_df[labelcol].tolist()))
        if len(annotations) == 1:
            agreed = agreed.append(pair_df.iloc[0, :])
        else:
            unagreed = unagreed.append(pair_df)
    
    return agreed, unagreed

    filepath = "<PATH>"
    df = pd.read_excel(filepath)
    
    agreed, unagreed = process_agreement(df, idcol="Content Id", textcol="Content", labelcol="cat")
    
    folder = "<PATH>"
    agreed.to_excel(os.path.join(folder, "agreed.xlsx"), index=False)
    unagreed.to_excel(os.path.join(folder, "unagreed.xlsx"), index=False)
    print(agreed)
    print()
    print(unagreed)
    
    
    
    
    