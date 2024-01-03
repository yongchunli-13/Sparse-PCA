import sys
import os

########  reproduce the lower bounds in Table 9 #########
sys.path.append(os.path.dirname(os.getcwd())+'/approximation_algorithms')
import lb_gen
reproduce_lb_table =  lb_gen.reproduce_table 
reproduce_lb_table(9) # Table 9