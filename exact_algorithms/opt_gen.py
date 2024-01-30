import MILP
import Branchandcut
import Branchandcut_15
import pandas as pd
import numpy as np
import spca_gurobi
import os

gen_data = MILP.gen_data
milp22 = MILP.milp22

spca_bc = Branchandcut.spca_bc
spca_bc_15 = Branchandcut_15.spca_bc

spca = spca_gurobi.spca
        
def reproduce_table(tablenumber):
    if tablenumber==2:
        
        # Optimal values of SPCA using MILP and Branch-and-cut
        df_Opt = pd.DataFrame(columns=('n', 's', 'MISDP 6', 'MISDP 6 UB', 'MISDPtime', 'MISDP 15', 'MISDP 15 UB', 'MISDP15time', 'MILP 22', 'MILP 22 UB','MILPtime', 'Gurobi', 'Gurobi UB', 'Gurobitime'))
        df_table2 = pd.DataFrame(columns=('n', 's', 'opt_val',  'MISDPtime', 'opt_val', 'MISDP15time', 'opt_val', 'MILPtime', 'opt_val','Gurobitime'))
        loc = 0
        n = 13
        data_name = 'pitprops'
        gen_data(n, data_name)
        
        for s in range(4, 11): # set the values of s
            print("This is case", loc+1)
        
        #------ MISDP (6) ------
            LB_6, UB_6, time_6 = spca_bc(n, data_name, s)
            if time_6 > 3600:
                LB_6 = '-'
                time_6 = '-'
            
        #------ MISDP (15) ------
            LB_15, UB_15, time_15 = spca_bc(n, data_name, s)
            if time_15 >= 3600:
                LB_15 = '-'
                time_15 = '-'
        
        #------ MILP (22) ------
            LB_22, UB_22, time_22 = milp22(n, data_name, s)
            if time_22 >= 3600:
                LB_22 = '-'
                time_22 = '-'
            
        #------ Gurobi solver ------
            LB_Gu, UB_Gu, time_Gu = spca(n, data_name, s)
            if time_Gu >= 3600:
                LB_Gu = '-'
                time_Gu = '-'
            
             
            df_Opt.loc[loc] = np.array([n, s, LB_6, UB_6, time_6, LB_15, UB_15, time_15, LB_22, UB_22, time_22, LB_Gu, UB_Gu, time_Gu])
            df_table2.loc[loc] = np.array([n, s, LB_6, time_6, LB_15, time_15, LB_22, time_22, LB_Gu, time_Gu])
            loc = loc+1 
    
        df_table2.to_csv(os.path.dirname(os.getcwd())+'/table_generation/table2.csv')
                
     
        
    if tablenumber==4:
        
        list_n = [13, 20, 26, 30, 34, 57, 64, 77, 90, 128]
        list_name = [ 'housing', 'keggdirected', 'pol', 'wdbc', 'dermatology',
                      'spambase', 'digits', 'buzz', 'song', 'gas']
        
        # Optimal values of SPCA using MILP and Branch-and-cut
        df_Opt = pd.DataFrame(columns=('n', 's', 'MISDP 6', 'MISDP 6 UB', 'MISDPtime', 'MISDP 15', 'MISDP 15 UB', 'MISDP15time', 'MILP 22', 'MILP 22 UB','MILPtime', 'Gurobi', 'Gurobi UB', 'Gurobitime'))
        df_table4 = pd.DataFrame(columns=('n', 's', 'opt_val',  'MISDPtime', 'opt_val', 'MISDP15time', 'opt_val', 'MILPtime', 'opt_val','Gurobitime'))
        loc = 0
        
        for i in range(10):
            n = list_n[i]
            data_name = list_name[i]
            gen_data(n, data_name)
            if n < 15:
                u = 5
            else:
                u = 10
            if n < 50:
                l = 5
            else:
                l = 10
            for s in range(l, min(21, n), u): # set the values of s
                print("This is case", loc+1)
            
            #------ MISDP (6) ------
                LB_6, UB_6, time_6 = spca_bc(n, data_name, s)
                if time_6 > 3600:
                    LB_6 = '-'
                    time_6 = '-'
                
            #------ MISDP (15) ------
                LB_15, UB_15, time_15 = spca_bc_15(n, data_name, s)
                if time_15 >= 3600:
                    LB_15 = '-'
                    time_15 = '-'
            
            #------ MILP (22) ------
                LB_22, UB_22, time_22 = milp22(n, data_name, s)
                if time_22 >= 3600:
                    LB_22 = '-'
                    time_22 = '-'
                
            #------ Gurobi solver ------
                LB_Gu, UB_Gu, time_Gu = spca(n, data_name, s)
                if time_Gu >= 3600:
                    LB_Gu = '-'
                    time_Gu = '-'
                
                 
                df_Opt.loc[loc] = np.array([n, s, LB_6, UB_6, time_6, LB_15, UB_15, time_15, LB_22, UB_22, time_22, LB_Gu, UB_Gu, time_Gu])
                df_table4.loc[loc] = np.array([n, s, LB_6, time_6, LB_15, time_15, LB_22, time_22, LB_Gu, time_Gu])
                loc = loc+1 
        
        df_table4.to_csv(os.path.dirname(os.getcwd())+'/table_generation/table4.csv')
    
    