from gurobipy import *
import math
import datetime
import os
import numpy as np
import pandas as pd
from scipy.linalg import svd
from gurobipy import GRB
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

## Import data
def gen_data(n):
    global V
    global A
    
    temp = pd.read_table(os.getcwd()+'/pitdata.csv',
                        header=None,encoding = 'utf-8',sep=',')
    temp = np.array(temp)
    A = np.matrix(temp)
    
    ## Cholesky factorization of A   
    U, s, V = svd(A) # eigen decomposition
    s[s<1e-6]=0 
    sqrt_eigen = [0]*n
    for i in range(n):
        if s[i]>0:
            sqrt_eigen[i] = np.sqrt(s[i])
        else:
            sqrt_eigen[i] = 0
               
    V = np.dot(np.diag(sqrt_eigen),V)  
    d = np.linalg.matrix_rank(A) ## the rank of the matrix A
    V = V[0:d,:]    
    V = np.matrix(V) #V: d*n

## Input a lower bound LB and an upper bound UB
## Solve the MILP for SPCA
def spca(n, d, nm, k, LB, UB):   
    #### Set model ####
    start = datetime.datetime.now()
    m = Model("milpspca")

    #### Solve MILP ####
    coeff = [0]*nm
    for i in range(nm):
        coeff[i] = (UB-LB)*2**(-i)
       
    ### ones vector ### 
    earray = np.array([1.0]*d)
    evect = np.matrix(earray).T    
    
    #### Creat variables ####
    lambdavar = m.addMVar(shape = 1, lb=0.0,  name="lambda")
    x = m.addMVar(shape=d, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    R = m.addMVar(shape=d, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="r")
    B = m.addMVar(shape=n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
    
    z = m.addMVar(n, vtype=GRB.BINARY, name="z")

    u1 = m.addMVar((d,n), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u1")
    u2 = m.addMVar((d,n), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u2")
    
    w = m.addMVar((d,d), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name = "w") 
    y = m.addMVar(d, vtype=GRB.BINARY, name="y")
    
    mu1 = m.addMVar((d,nm), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu1")
    mu2 = m.addMVar((d,nm), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu2")
    alpha = m.addMVar(nm, vtype=GRB.BINARY, name="alpha")
    
    #### Set objective ####
    m.setObjective(1*lambdavar, GRB.MAXIMIZE)
    
    #### Define constraints ####    
    m.addConstr(z.sum() == k) # size constraint
    
    ## eigenvalue ##
    for i in range(n):
        m.addConstr(B[i] == V[:,i].T @ u1[:,i])
          
    m.addConstr(V @ B - UB*x + sum(coeff[i] * mu1[:,i] for i in range(nm)) == R) # eigenvalues
    
    m.addConstr(R <= (UB-LB)*2**(-nm))
    m.addConstr(R >= -(UB-LB)*2**(-nm))
    
 
    ## lambda approx ###
    m.addConstr(sum(coeff[i]*alpha[i] for i in range(nm)) + lambdavar == UB)
    
    ## inf norm ##
    m.addConstr(sum(w[:,i] for i in range(d)) == x)
    
    for i in range(d):
        m.addConstr(w[i,i] == y[i])
        m.addConstr(w[:,i] <= evect @ y[i])
        m.addConstr(w[:,i] >= -evect @ y[i])
    m.addConstr(y.sum() == 1)
            
    ## left-hand binary ##  
    #eone = np.array([1.0]*d)    
    for i in range(n):
        m.addConstr(u1[:,i] + u2[:,i] == x)
        
        m.addConstr(u1[:,i] <= evect @ z[i])
        m.addConstr(u1[:,i] >= -evect @ z[i])
        
        m.addConstr(u2[:,i] <= earray - evect @ z[i])        
        m.addConstr(u2[:,i] >= -earray + evect @ z[i])

    ## right-hand binary ##
    for i in range(nm):
        m.addConstr(mu1[:,i] + mu2[:,i] == x)
        
        m.addConstr(mu1[:,i] <= evect @ alpha[i])
        m.addConstr(mu1[:,i] >= -evect @ alpha[i])
        
        m.addConstr(mu2[:,i] <= earray - evect @ alpha[i])
        m.addConstr(mu2[:,i] >= -earray + evect @ alpha[i])

    m.params.OutputFlag = 1    
    m.params.timelimit = 3600 
    m.optimize()
    
    end = datetime.datetime.now()
    time = (end-start).seconds
    return m.objval, time
            
n = 13
d = 13
k = 5 # sparsity
UB = 4.2186 # can be computed by continuous relaxation
LB = 3.4062 # can be computed by local search algorithm
nm = int(math.log((UB-LB)*1e+4))+1 # epsilon = 1e-4

gen_data(n)
fval, time = spca(n, d, nm, k, LB, UB)
