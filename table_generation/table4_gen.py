import sys
import os

##########  reproduce the results of exact methods in Table 4  #########
sys.path.append(os.path.dirname(os.getcwd())+'/exact_algorithms')
import opt_gen
reproduce_opt_table =  opt_gen.reproduce_table
reproduce_opt_table(4) # Table 4
