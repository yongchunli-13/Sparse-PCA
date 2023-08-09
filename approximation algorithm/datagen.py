#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 23:50:42 2023

@author: yongchunli
"""

import lower_bounds
import spca_sdp
import pandas as pd
import numpy as np
import datetime

gen_data = lower_bounds.gen_data
localsearch = lower_bounds.localsearch
truncation = lower_bounds.truncation
greedy = lower_bounds.greedy
randomized_SPCA = lower_bounds.randomized_SPCA
spca_sdp = spca_sdp.spca_sdp

n = 34
gen_data(n)

# Lower bounds
df_LB = pd.DataFrame(columns=('n', 's', 'truncation', 'ttime', 'random', 'rtime', 'sdp', 'stime',
                              'greedy', 'gtime', 'localsearch', 'ltime'))
loc = 0
for s in range(5, 20, 10): # set the values of s
    print("This is case", loc+1)

#------ truncation ------
    ttime, tval = truncation(n, s)

#------ greedy ------
    gtime, x, gval = greedy(n, s)
    
#------ local search ------
    ltime, lval, bestz = localsearch(n, s)

#------ randomized spca ------
    start = datetime.datetime.now()
    N = 50
    rval = 0.0
    for t in range(N):
        x, val, supp = randomized_SPCA(n, s)
        if supp <= s:
            rval = max(rval, val)
    end = datetime.datetime.now()
    rtime = (end-start).seconds

#------ SDP spca ------    
    stime, sval = spca_sdp(n, s)

    
    df_LB.loc[loc] = np.array([n, s, (lval-tval)/lval, ttime, 
                               (lval-rval)/lval, rtime,  
                               (lval-sval)/lval, stime, 
                               (lval-gval)/lval, gtime, lval, ltime])
    loc = loc+1 