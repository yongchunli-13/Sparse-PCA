
from gurobipy import *
from gurobipy import GRB
import math
import datetime
import os
import numpy as np
import pandas as pd


## Import data
def gen_data(n, data_name):
    global S
    global E
    global V
    global A
    global T
    global d
    global A
    
    data = pd.read_table(os.path.dirname(os.getcwd())+'/datasets/'+data_name+'_txt',
          encoding = 'utf-8',sep=',')
    temp = data.drop(['Unnamed: 0'], axis=1)
    A = np.matrix(np.array(temp))
    
    ## Cholesky factorization of A  
    s, V = np.linalg.eigh(A) # eigen decomposition
    sqrt_eigen = [0]*n
    for i in range(n):
        if s[i] > 0:
            sqrt_eigen[i] = np.sqrt(s[i])
        else:
            sqrt_eigen[i] = 0
               
    V = np.diag(sqrt_eigen)*V.T
    d = len([i for i in range(n) if s[i]>=1e-10]) ## the rank of the matrix A
    inx = [i for i in range(n) if s[i]>=1e-10]
    V = V[inx,:]    
    V = np.matrix(V) #V: d*n


    S = [[V[:,i] * V[:,i].T for i in range(n)]] # set of matrix v_i*v_i^T
    S = S[0]
    
    T = [0]*n
    for i in range(n):
        T[i] = (V[:,i].T * V[:, i])[0,0]  # set of norm v_i^T*v_i
        
    E=np.eye(n, dtype=int)
    
    

def spca(n, data_name, s):
    start = datetime.datetime.now()
    gen_data(n, data_name)
    #### Set model ####
    m = Model("spca")
    
    m.params.timelimit = 3600.0
    
    #### Creat variables ####
    x = m.addVars(n, lb=-GRB.INFINITY, name="x")
    v = m.addVars(n, lb =0.0, ub=1.0, name="v")
    z = m.addVars(n, vtype=GRB.BINARY, name="z")
 
    #### Set objective ####
    obj = sum(x[i]*A[i,j]*x[j] for i in range(n)for j in range(n))
    m.setObjective(obj, GRB.MAXIMIZE)
    
    #### Define constraints ####
    m.addConstr(z.sum() == s)
    m.addConstr(sum(x[i]*x[i] for i in range(n)) == 1.0, "c1")
    
    for i in range(n):
        m.addConstr(v[i] >= x[i]) 
        m.addConstr(-v[i]<= x[i]) 
        m.addConstr(z[i] >= x[i]) 
        m.addConstr(z[i] >= -x[i]) 
        m.addConstr(v[i] <= z[i])
#        
    m.addConstr(v.sum() <= math.sqrt(s))
    m.params.NonConvex=2
    m.params.OutputFlag = 1  
    m.Params.MIPGap= 1e-4
    m.optimize()
    
    end = datetime.datetime.now()
    time = (end-start).seconds
    return m.objval, m.ObjBound, time
            
