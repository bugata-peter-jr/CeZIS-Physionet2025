# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:58:32 2024

@author: bugatap
"""

from inspect import getmembers

class Config(object):
    
    def __init__(self):
        # resnet config
        self.kernel = 5
        self.width = 1/4
        self.normalization = 'bn'
        self.dropout = 0.0
        self.n_groups = 16

        # dataset and loader params
        self.inplen = 8.96
        self.fs = 200
        self.leads = 12
        self.standardize_sig = True
        
        # augmentation
        self.test_n_reps = 5
    
    def __str__(self):
        s = ''
        for name, value in getmembers(self):
            if name.startswith('__'):
                continue
            s += name + ' : ' + str(value) + '\n'
        return s

    
if __name__ == '__main__':
    
    cfg = Config()
    print(cfg)