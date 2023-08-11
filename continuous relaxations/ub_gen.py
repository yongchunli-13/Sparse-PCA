#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 20:21:22 2023

@author: yongchunli
"""


import sdp_rels
import sdp_rel_thirteen
import pandas as pd
import numpy as np

gen_data = sdp_rels.gen_data
sdp_rel_eight = sdp_rels.sdp_rel_eight
sdp_rel_sixteen = sdp_rels.sdp_rel_sixteen
sdp_rel_seventeen = sdp_rels.sdp_rel_seventeen

spca_rel_thirteen = sdp_rel_thirteen.spca_rel_thirteen

n = 26
gen_data(n)

#### Upper bounds of SDP relaxations (8), (16), and (17)

df_SDPRel = pd.DataFrame(columns=('n', 's', 'SDP_rel 8', '8time', 'SDP_rel 16', '16time', 'benchmark 17', '17time'))
loc = 0
for s in range(5, 20, 10): # set the values of s
    print("This is case", loc+1)

#------ SDP relaxation (8) ------
    val_eight, time_eight = sdp_rel_eight(n, s)

#------ SDP relaxation (16) ------
    val_sixteen, time_sixteen = sdp_rel_sixteen(n, s)
     
#------ benchmark (17) ------
    bval, btime = sdp_rel_seventeen(n, s)
    
    df_SDPRel.loc[loc] = np.array([n, s, val_eight, time_eight, val_sixteen, time_sixteen, bval, btime])
    loc = loc+1 
    



#### Upper bounds of SDP relaxation (13) solved by subgradient descent method
n = 26

df_Rel = pd.DataFrame(columns=('n', 's', 'SDP_rel 13', '13time'))
loc = 0
for s in range(5, 20, 10): # set the values of s
    print("This is case", loc+1)

#------ Relaxation (13) ------
    val_thirteen, time_thirteen = spca_rel_thirteen(n, s)

   
    df_Rel.loc[loc] = np.array([n, s, val_thirteen, time_thirteen])
    loc = loc+1     
    