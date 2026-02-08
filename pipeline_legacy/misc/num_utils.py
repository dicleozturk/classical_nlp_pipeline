'''
Created on May 16, 2017

@author: dicle
'''

import sys
sys.path.append("..")


import random


'''
generates N integer random numbers whose sum equals to the given total
'''
def generate_random_numbers_with_sum(N, total, vals):

    if N < 1:
        return
  
    val = random.randint(0, total)
    vals.append(val)
    
    if N > 1:
        generate_random_numbers_with_sum(N-1, total-val, vals)


'''
def generate_random_numbers_with_sum(N, total, vals):


    if N == 1:
        last_val = random.randint(0, total)
        vals.append(last_val)
    else:
        val = random.randint(0, total)
        vals.append(val)
        generate_random_numbers_with_sum(N-1, total-val, vals)  
'''
        
    vals = []
    x = generate_random_numbers_with_sum(1, 10, vals)
    
    print(x)
    print(vals)
    