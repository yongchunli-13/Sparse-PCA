import sys
import os

#########  reproduce the results of continuous relaxations in Table 6 #########
sys.path.append(os.path.dirname(os.getcwd())+'/continuous_relaxations')

import ub_gen
reproduce_ub_table =  ub_gen.reproduce_table
reproduce_ub_table(6) # Table 6
