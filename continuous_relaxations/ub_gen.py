
import sdp_rels
import sdp_rel_thirteen
import pandas as pd
import numpy as np
import os

gen_data = sdp_rels.gen_data
sdp_rel_eight = sdp_rels.sdp_rel_eight
sdp_rel_sixteen = sdp_rels.sdp_rel_sixteen
sdp_rel_seventeen = sdp_rels.sdp_rel_seventeen

spca_rel_thirteen = sdp_rel_thirteen.spca_rel_thirteen

def reproduce_table(tablenumber):
    if tablenumber==5:
        
        # continuous relaxation values
        df_SDPRel = pd.DataFrame(columns=('n', 's', 'SDP_rel 8', '8time', 'SDP_rel 13', '13time', 'SDP_rel 16', '16time', 'benchmark 17', '17time'))
        df_table5 = pd.DataFrame(columns=('n', 's', 'gap(%)',  'rel8_time', 'gap(%)', 'rel13_time', 'gap(%)', 'rel16_time', 
                                           'gap(%)',  'rel17_time'))
        loc = 0
        n = 13
        data_name = 'pitprops'
        gen_data(n, data_name)
        
        for s in range(4, 11): # set the values of s
        
        #------ SDP relaxation (8) ------
            val_eight, time_eight = sdp_rel_eight(n, s)
        
        #------ Relaxation (13) ------
            val_thirteen, LB, time_thirteen = spca_rel_thirteen(n, data_name, s)

        #------ SDP relaxation (16) ------
            val_sixteen, time_sixteen = sdp_rel_sixteen(n, s)
             
        #------ benchmark (17) ------
            bval, btime = sdp_rel_seventeen(n, s)
            
            df_SDPRel.loc[loc] = np.array([n, s, val_eight, time_eight, val_thirteen, time_thirteen, val_sixteen, time_sixteen, bval, btime])
            gap_eight = (val_eight-LB)/LB*100
            gap_thirteen = (val_thirteen-LB)/LB*100
            gap_sixteen = (val_sixteen-LB)/LB*100
            gap_b = (bval-LB)/LB*100
            df_table5.loc[loc] = np.array([n, s, gap_eight, time_eight, 
                                       gap_thirteen, time_thirteen,  
                                       gap_sixteen, time_sixteen, 
                                       gap_b, btime])
            
            loc = loc+1
            
        df_table5.to_csv(os.path.dirname(os.getcwd())+'/table_generation/table5.csv')
        
    if tablenumber==6:
        
        list_n = [13, 20, 26, 30, 34, 57, 64, 77, 90, 128]
        list_name = [ 'housing', 'keggdirected', 'pol', 'wdbc', 'dermatology',
                      'spambase', 'digits', 'buzz', 'song', 'gas']
        
        # continuous relaxation values
        df_SDPRel = pd.DataFrame(columns=('n', 's', 'SDP_rel 8', '8time', 'SDP_rel 13', '13time', 'SDP_rel 16', '16time', 'benchmark 17', '17time'))
        df_table6 = pd.DataFrame(columns=('n', 's', 'gap(%)',  'rel8_time', 'gap(%)', 'rel13_time', 'gap(%)', 'rel16_time', 
                                           'gap(%)',  'rel17_time'))
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
            #------ SDP relaxation (8) ------
                if n >=50:
                    time_eight = 3600
                    val_eight = '-'
                else:
                    val_eight, time_eight = sdp_rel_eight(n, s)
            
            #------ Relaxation (13) ------
                val_thirteen, LB, time_thirteen = spca_rel_thirteen(n, data_name, s)
        
            #------ SDP relaxation (16) ------
                if n >= 100:
                    time_sixteen = 3600
                    val_sixteen = '-'
                else:
                    val_sixteen, time_sixteen = sdp_rel_sixteen(n, s)
                 
            #------ benchmark (17) ------
                if n >= 100:
                    btime = 3600
                    bval = '-'
                else:
                    bval, btime = sdp_rel_seventeen(n, s)
                
                df_SDPRel.loc[loc] = np.array([n, s, val_eight, time_eight, val_thirteen, time_thirteen, val_sixteen, time_sixteen, bval, btime])
                if n >=50:
                    gap_eight = '-'
                else:
                    gap_eight = (val_eight-LB)/LB*100
                    
                gap_thirteen = (val_thirteen-LB)/LB*100
                if n >=100:
                    gap_sixteen = '-'
                    gap_b = '-'
                else:
                    gap_sixteen = (val_sixteen-LB)/LB*100
                    gap_b = (bval-LB)/LB*100
                df_table6.loc[loc] = np.array([n, s, gap_eight, time_eight, 
                                           gap_thirteen, time_thirteen,  
                                           gap_sixteen, time_sixteen, 
                                           gap_b, btime])
                
                loc = loc+1
                
        df_table6.to_csv(os.path.dirname(os.getcwd())+'/table_generation/table6.csv')
    
    if tablenumber==7:
        
        list_n = [79, 118, 500, 2000]
        list_name = ['Eisen1', 'Eisen2', 'Colon', 'Reddit']
        # continuous relaxation values
        df_SDPRel = pd.DataFrame(columns=('n', 's', 'SDP_rel 8', '8time', 'SDP_rel 13', '13time', 'SDP_rel 16', '16time', 'benchmark 17', '17time'))
        df_table7 = pd.DataFrame(columns=('n', 's', 'gap(%)',  'rel8_time', 'gap(%)', 'rel13_time', 'gap(%)', 'rel16_time', 
                                           'gap(%)',  'rel17_time'))
        loc = 0
        
        
        for i in range(4):
            n = list_n[i]
            data_name = list_name[i]
            gen_data(n, data_name)
                    
            for s in range(10, 21, 10): # set the values of s
                print("This is case", loc+1)
                #------ SDP relaxation (8) ------
                if n >=50:
                    time_eight = 3600
                    val_eight = '-'
                else:
                    val_eight, time_eight = sdp_rel_eight(n, s)
            
            #------ Relaxation (13) ------
                if n >= 1000:
                    time_thirteen = 3600
                    val_thirteen = '-'
                else:
                    val_thirteen, LB, time_thirteen = spca_rel_thirteen(n, data_name, s)
            
            #------ SDP relaxation (16) ------
                if n >= 100:
                    time_sixteen = 3600
                    val_sixteen = '-'
                else:
                    val_sixteen, time_sixteen = sdp_rel_sixteen(n, s)
                 
            #------ benchmark (17) ------
                if n >= 100:
                    btime = 3600
                    bval = '-'
                else:
                    bval, btime = sdp_rel_seventeen(n, s)
                
                df_SDPRel.loc[loc] = np.array([n, s, val_eight, time_eight, val_thirteen, time_thirteen, val_sixteen, time_sixteen, bval, btime])
                if n >=50:
                    gap_eight = '-'
                else:
                    gap_eight = (val_eight-LB)/LB*100
                    
                if n>=1000:
                    gap_thirteen = '-'
                else:
                    gap_thirteen = (val_thirteen-LB)/LB*100
                if n >=100:
                    gap_sixteen = '-'
                    gap_b = '-'
                else:
                    gap_sixteen = (val_sixteen-LB)/LB*100
                    gap_b = (bval-LB)/LB*100
                    
                df_table7.loc[loc] = np.array([n, s, gap_eight, time_eight, 
                                           gap_thirteen, time_thirteen,  
                                           gap_sixteen, time_sixteen, 
                                           gap_b, btime])
                
                loc = loc+1
                
        df_table7.to_csv(os.path.dirname(os.getcwd())+'/table_generation/table7.csv')
    
            
        
