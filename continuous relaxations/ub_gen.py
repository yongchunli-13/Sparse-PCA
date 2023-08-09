#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 20:21:22 2023

@author: yongchunli
"""


import sdp_rels
import upper_bounds
import pandas as pd
import numpy as np

gen_data = sdp_rels.gen_data
sdp_rel_eight = sdp_rels.sdp_rel_eight
sdp_rel_sixteen = sdp_rels.sdp_rel_sixteen
sdp_rel_seventeen = sdp_rels.sdp_rel_seventeen

ub_gen_data= upper_bounds.gen_data
spca_rel_thirteen = upper_bounds.spca_rel_thirteen
spca_rel_twenty = upper_bounds.spca_rel_twenty

n = 118
gen_data(n)

# Upper bounds of SDP relaxation
df_SDPRel = pd.DataFrame(columns=('n', 's', 'SDP_rel 8', '8time', 'SDP_rel 16', '16time', 'benchmark 17', '17time'))
loc = 0
for s in range(10, 30, 10): # set the values of s
    print("This is case", loc+1)

# #------ SDP relaxation (8) ------
    # val_eight, time_eight = sdp_rel_eight(n, s)

#------ SDP relaxation (16) ------
    val_sixteen, time_sixteen = sdp_rel_sixteen(n, s)
     
#------ benchmark (17) ------
    bval, btime = sdp_rel_seventeen(n, s)
    
    df_SDPRel.loc[loc] = np.array([n, s, 0, 0, val_sixteen, time_sixteen, bval, btime])
    loc = loc+1 
    



######## Upper bounds of SDP relaxation: relaxation (13) and (20)  #########
####### solve by subgradient descent method
# n = 79
# ub_gen_data(n)

# df_Rel = pd.DataFrame(columns=('n', 's', 'SDP_rel 13', '13time', 'SDP_rel 20', '20time'))
# loc = 0
# for s in range(10, 40, 10): # set the values of s
#     print("This is case", loc+1)

# #------ Relaxation (13) ------
#     musol, val_thirteen, time_thirteen = spca_rel_thirteen(n, s)

# #------ Relaxation (20) ------
#     val_twenty, time_twenty = spca_rel_twenty(n, s)
     
#     df_Rel.loc[loc] = np.array([n, s, val_thirteen, time_thirteen, val_twenty, time_twenty])
#     loc = loc+1     
    