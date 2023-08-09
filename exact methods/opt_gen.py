#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 18:08:42 2023

@author: yongchunli
"""


import MILP
import Branchandcut
import pandas as pd
import numpy as np
import spca_gurobi
import datetime
import uci_dataset as dataset


gen_data = MILP.gen_data
milp22 = MILP.milp22

spca_bc = Branchandcut.spca_bc

n = 13
gen_data(n)

# # Optimal values of SPCA using MILP and Branch-and-cut
# df_Opt = pd.DataFrame(columns=('n', 's', 'MISDP 6', 'MISDP 6 UB', 'MISDPtime', 'MILP 22', 'MILP 22 UB','MILPtime'))
# loc = 0
# for s in range(20, 30, 10): # set the values of s
#     print("This is case", loc+1)

# # #------ MISDP (6) ------
#     LB_6, UB_6, time_6 = spca_bc(n, s)

# # #------ MILP (22) ------
#     # LB_22, UB_22, time_22 = milp22(n, s)
#     # print(LB_22, UB_22, time_22 )
    
     
# # #------ benchmark (17) ------
    
#     df_Opt.loc[loc] = np.array([n, s, LB_6, UB_6, time_6, LB_22, UB_22, time_22])
#     loc = loc+1 
    


# # Optimal values of SPCA using Solver Gorubi
# n = 64
# spca = spca_gurobi.spca
# df_Opt1 = pd.DataFrame(columns=('n', 's', 'SPCA 24', 'SPCA 24 UB', 'SPCAtime'))
# loc = 0
# for s in range(10, 30, 10): # set the values of s
#     print("This is case", loc+1)

# # #------ SPCA (24) ------
#     LB_24, UB_24, time_24 = spca(n, s)

#     df_Opt1.loc[loc] = np.array([n, s, LB_24, UB_24, time_24])
#     loc = loc+1 
    

# Upper bounds of SDP (13)
spca_rel_thirteen = MILP.spca_rel_thirteen
n=13
gen_data(n)
df_UB = pd.DataFrame(columns=('n', 's', 'LB', 'SDP 13', 'SDP 13 time'))
loc = 0
for s in range(4, 11): # set the values of s
    print("This is case", loc+1)

# #------ SPCA (13) ------
    start = datetime.datetime.now()
    LB, UB = spca_rel_thirteen(n, s)
    end = datetime.datetime.now()
    time = (end-start).seconds

    df_UB.loc[loc] = np.array([n, s, LB, (UB-LB)/LB, time])
    loc = loc+1 
   
    