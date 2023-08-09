#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 23:50:42 2023

@author: yongchunli
"""

import lu_bounds
import rspca
import pandas as pd
import numpy as np
import datetime

gen_data = lu_bounds.lower_upper_bounds
localsearch = lu_bounds.localsearch
spca_rel_thirteen = lu_bounds.spca_rel_thirteen
truncation = lu_bounds.truncation
greedy = lu_bounds.greedy

randomized_SPCA = rspca.randomized_SPCA

n = 57
gen_data(n)

# Lower bounds
loc = 0
df_LB = pd.DataFrame(columns=('n', 's', 'truncation', 'ttime', 'random', 'rtime', 'sdp', 'stime',
                              'greedy', 'gtime', 'localsearch', 'ltime'))

for s in range(10, 30, 10): # set the values of s
    print("This is case", loc+1)

###### truncation ######
    gtime, tval = truncation(n, s)
    gtime, x, gval = greedy(n, s)

###### randomized spca ######
    start = datetime.datetime.now()
    N = 50
    best = 0.0
    for i in range(N):
        x, z, supp = randomized_SPCA(n, s)
        if supp <= s:
            best = max(best, z)
    end = datetime.datetime.now()
    time = end - start
    print(best, time)
    
    root_LB, xsol, time  = localsearch(n ,d, s) 
    print("The lower bound at root is", root_LB)
    df_LB.loc[loc] = np.array([n, d, s, root_LB, time])
    loc = loc+1 
    
    gen_data(n)
    UB, time = spca_rel_thirteen(n, s)

