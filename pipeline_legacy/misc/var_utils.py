'''
Created on Oct 9, 2017

@author: dicle
'''


import sys
sys.path.append("..")

import time


def get_time_stamp():
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    return timestamp

    print(get_time_stamp())