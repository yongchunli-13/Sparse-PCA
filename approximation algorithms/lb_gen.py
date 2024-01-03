
import lower_bounds
import spca_sdp
import pandas as pd
import numpy as np
import datetime
import os

gen_data = lower_bounds.gen_data
localsearch = lower_bounds.localsearch
truncation = lower_bounds.truncation
greedy = lower_bounds.greedy
randomized_SPCA = lower_bounds.randomized_SPCA
spca_sdp = spca_sdp.spca_sdp


def reproduce_table(tablenumber):
    
    if tablenumber==8:
        
        # Lower bounds
        df_LB = pd.DataFrame(columns=('n', 's', 'truncation', 'ttime', 'random', 'rtime', 'sdp', 'stime',
                                      'greedy', 'gtime', 'localsearch', 'ltime'))
       
        df_table8 = pd.DataFrame(columns=('n', 's', 'gap(%)',  'ttime', 'gap(%)', 'rtime', 'gap(%)', 'stime', 
                                           'gap(%)',  'gtime', 'gap(%)',  'ltime'))
        loc = 0
        
        n = 13
        data_name = 'pitprops'
        gen_data(n, data_name)
        
        for s in range(4, 11): # set the values of s
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
            stime, sval = spca_sdp(n, data_name, s)
        
            
            df_LB.loc[loc] = np.array([n, s, tval, ttime, 
                                       rval, rtime,  
                                       sval, stime, 
                                       gval, gtime, lval, ltime])
            
            lgap = (lval-lval)/lval*100
            ggap = (lval-gval)/lval*100
            tgap = (lval-tval)/lval*100
            rgap = (lval-rval)/lval*100
            sgap = (lval-sval)/lval*100
            df_table8.loc[loc] = np.array([n, s, tgap, ttime, 
                                       rgap, rtime,  
                                       sgap, stime, 
                                       ggap, gtime, lgap, ltime])
            
            loc = loc+1 
        df_table8.to_csv(os.path.dirname(os.getcwd())+'/table_generation/table8.csv')
    
        
        
    if tablenumber==9:
        
        list_n = [13, 20, 26, 30, 34, 57, 64, 77, 90, 128]
        list_name = [ 'housing', 'keggdirected', 'pol', 'wdbc', 'dermatology',
                      'spambase', 'digits', 'buzz', 'song', 'gas']
        
        # Lower bounds
        df_LB = pd.DataFrame(columns=('n', 's', 'truncation', 'ttime', 'random', 'rtime', 'sdp', 'stime',
                                      'greedy', 'gtime', 'localsearch', 'ltime'))
       
        df_table9 = pd.DataFrame(columns=('n', 's', 'gap(%)',  'ttime', 'gap(%)', 'rtime', 'gap(%)', 'stime', 
                                           'gap(%)',  'gtime', 'gap(%)',  'ltime'))
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
                stime, sval = spca_sdp(n, data_name, s)
            
                
                df_LB.loc[loc] = np.array([n, s, tval, ttime, 
                                           rval, rtime,  
                                           sval, stime, 
                                           gval, gtime, lval, ltime])
                
                lgap = (lval-lval)/lval*100
                ggap = (lval-gval)/lval*100
                tgap = (lval-tval)/lval*100
                rgap = (lval-rval)/lval*100
                sgap = (lval-sval)/lval*100
                df_table9.loc[loc] = np.array([n, s, tgap, ttime, 
                                           rgap, rtime,  
                                           sgap, stime, 
                                           ggap, gtime, lgap, ltime])
                
                loc = loc+1 
        df_table9.to_csv(os.path.dirname(os.getcwd())+'/table_generation/table9.csv')
    
        
        
    if tablenumber==10:
        
        list_n = [79, 118, 500, 2000]
        list_name = ['Eisen1', 'Eisen2', 'Colon', 'Reddit']
         
        # Lower bounds
        df_LB = pd.DataFrame(columns=('n', 's', 'truncation', 'ttime', 'random', 'rtime', 'sdp', 'stime',
                                      'greedy', 'gtime', 'localsearch', 'ltime'))
       
        df_table10 = pd.DataFrame(columns=('n', 's', 'gap(%)',  'ttime', 'gap(%)', 'rtime', 'gap(%)', 'stime', 
                                           'gap(%)',  'gtime', 'gap(%)',  'ltime'))
        
        loc = 0
        
        for i in range(4):
            n = list_n[i]
            data_name = list_name[i]
            gen_data(n, data_name)
                    
            for s in range(10, 21, 10): # set the values of s
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
                if n>=200:
                    stime = 3600 
                    sval = '-'
                    sgap = '-'
                else:
                    stime, sval = spca_sdp(n, data_name, s)
                    sgap = (lval-sval)/lval*100
            
                
                df_LB.loc[loc] = np.array([n, s, tval, ttime, 
                                           rval, rtime,  
                                           sval, stime, 
                                           gval, gtime, lval, ltime])
                
                lgap = (lval-lval)/lval*100
                ggap = (lval-gval)/lval*100
                tgap = (lval-tval)/lval*100
                rgap = (lval-rval)/lval*100
                df_table10.loc[loc] = np.array([n, s, tgap, ttime, 
                                           rgap, rtime,  
                                           sgap, stime, 
                                           ggap, gtime, lgap, ltime])
                
                loc = loc+1 
        df_table10.to_csv(os.path.dirname(os.getcwd())+'/table_generation/table10.csv')
    