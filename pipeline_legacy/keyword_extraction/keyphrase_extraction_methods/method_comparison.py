'''
Created on Nov 6, 2018

@author: dicle
'''

import os
from pprint import pprint

from keyword_extraction.keyphrase_extraction_methods import generic_keyphrase_extractor,\
    tfidf_keyword_extractor, baseline_chunks_as_keyphrases


def keyphrases_all_methods(text, lang):
    
    keyphrases_generic = generic_keyphrase_extractor.keyphrase_extraction_pipeline(docs=[text], lang=lang, window=1, nphrases=None)
    keyphrases_tfidf = tfidf_keyword_extractor.extract_keyphrases_tfidf_single(text=text, lang=lang, candidate_type="all", top_n_phrases=50)
    keyphrases_chunks = baseline_chunks_as_keyphrases.extract_candidate_chunks(text=text, lang=lang)
    
    
    print("KEYPHRASES\n*************\n")
    
    print("  by generic method of Wang et al. 2015:")
    pprint(keyphrases_generic)
    
    print("  by tfidf weighting:")
    pprint(keyphrases_tfidf)
    
    print("  by shallow parsing (chunking):")
    pprint(keyphrases_chunks)
    


    main_files()



    