#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 18:08:42 2023

@author: yongchunli
"""


import MILP
import Branchandcut
import Branchandcut_15
import pandas as pd
import numpy as np
import spca_gurobi
import datetime
import uci_dataset as dataset


gen_data = MILP.gen_data
milp22 = MILP.milp22

spca_bc = Branchandcut.spca_bc
spca_bc_15 = Branchandcut_15.spca_bc

n = 26
gen_data(n)

# Optimal values of SPCA using MILP and Branch-and-cut
df_Opt = pd.DataFrame(columns=('n', 's', 'MISDP 6', 'MISDP 6 UB', 'MISDPtime', 'MISDP 15', 'MISDP 15 UB', 'MISDP15time', 'MILP 22', 'MILP 22 UB','MILPtime'))
loc = 0
for s in range(5, 10, 10): # set the values of s
    print("This is case", loc+1)

#------ MISDP (6) ------
    LB_6, UB_6, time_6 = spca_bc(n, s)
    
#------ MISDP (15) ------
    LB_15, UB_15, time_15 = spca_bc_15(n, s)

#------ MILP (22) ------
    LB_22, UB_22, time_22 = milp22(n, s)
    print(LB_22, UB_22, time_22 )
    
     
    df_Opt.loc[loc] = np.array([n, s, LB_6, UB_6, time_6, LB_15, UB_15, time_15, LB_22, UB_22, time_22])
    loc = loc+1 
    


# Optimal values of SPCA using Solver Gorubi
n = 26
spca = spca_gurobi.spca
df_Opt1 = pd.DataFrame(columns=('n', 's', 'SPCA 24', 'SPCA 24 UB', 'SPCAtime'))
loc = 0
for s in range(5, 10, 10): # set the values of s
    print("This is case", loc+1)

# #------ SPCA (24) ------
    LB_24, UB_24, time_24 = spca(n, s)

    df_Opt1.loc[loc] = np.array([n, s, LB_24, UB_24, time_24])
    loc = loc+1 
    
