import os
import numpy as np
import pandas as pd
from numpy import matrix
from scipy.linalg import svd
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
import datetime
import math
from gurobipy import *
from gurobipy import GRB

## Import data
def gen_data(n):
    global S
    global E
    global V
    global A
    global T
    
    temp = pd.read_table(os.getcwd()+'/pitdata.csv',
                        header=None,encoding = 'utf-8',sep=',')

    temp = np.array(temp)
    A = np.matrix(temp)
    
    ## Cholesky factorization of A  
    U, s, V = svd(A) # SVD decomposition
    s[s<1e-6]=0 
    
    sqrt_eigen = [0]*n
    for i in range(n):
        if s[i]>0:
            sqrt_eigen[i] = np.sqrt(s[i])
        else:
            sqrt_eigen[i] = 0
               
    V = np.dot(np.diag(sqrt_eigen),V) 
    d = np.linalg.matrix_rank(A) # d is the rank of A
    V = V[0:d,:]
    
    V = matrix(V)
    V = V.T # V: n*d

    S = [[V[i].T * V[i] for i in range(n)]] # set of matrix v_i*v_i^T
    S = S[0]
    
    T = [0]*n
    for i in range(n):
        T[i] = (V[i] * V[i].T)[0,0]  # set of norm v_i^T*v_i
        
    E=np.eye(n, dtype=int)
    
## given any binary z, compute the coefficients of vaild inequality    
def validcut(z, n):
    nu = 0
    mu = [0]*n
    
    sel_index = [i for i in range(n) if z[i] ==1]
    uns_index = [i for i in range(n) if z[i] == 0]

    val = 0
    for i in sel_index:
        val = val + S[i]
        
    a,b = largest_eigsh(val,1) # compute the largest eigenvalue
    nu = max(a)
    mu = [0]*n
    for i in uns_index:
        mu[i] = T[i]
    return nu, mu

## projection oracle of continuous variable z
def proj(zsol, n, k):
    #### create model ####
    m = Model()
    
    # add variables
    zvar = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")

    # add constraint
    m.addConstr(zvar.sum() == k)
    m.update()

    # specify optimization direction
    m.setObjective(quicksum((zsol[i]-zvar[i])*(zsol[i]-zvar[i]) for i in range(n)), GRB.MINIMIZE)
    m.update()

    m.params.OutputFlag=0
    m.optimize()

    z = [0]*n
    for i in range(n):
        z[i] = zvar[i].x

    return z
    
## compute the subgradient and update (z, mu) in the next iteration  
# gamma_t: step size
def subgrad(z, mu, k, gamma_t):
    nx = len(z)  
    uns_index = [i for i in range(nx) if mu[i] < T[i]] 
    val = 0.0  
    for i in uns_index:
        val = val + (1-mu[i]/T[i])*S[i]
    a,b = largest_eigsh(val,1) # compute the largest eigenvalue

    val = 0.0
    subg = [0.0]*nx
    for i in range(nx):
        subg[i] = math.pow(float(b.T*V[i].T),2.0) 
    for i in range(nx):
        subg[i] = z[i] - subg[i]/float(T[i]) #supgradient at mu
        
    musol = np.array(mu) - gamma_t*np.array(subg)
    # compute projection and find mu_{t+1}
    musol[musol < 0] = 0
    for i in range(nx):
        if musol[i] > T[i]:
            musol[i] = T[i] 

    zsol = np.array(z) + gamma_t * np.array(mu)
    zsol = proj(zsol, nx, k) # compute projection and find z_{t+1}

    lb = a[0] + np.dot(mu, z) # lower bound
    
    zub = [0]*nx
    index = np.argsort(-np.array(mu))
    for i in range(k):
        ind = index[i]
        zub[ind] = 1
    ub = a[0] + np.dot(mu, zub) # upper bound
    
    return lb, ub, a, musol, zsol

## compute the continuous relaxation problem (13)        
def spca_rel(n, k):   
    gen_data(n)
            
    #### Set model ####
    start = datetime.datetime.now() 
    m = Model("spca")

    #### Creat variables ####
    zvar = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")
    wvar = m.addVar(obj=1.0, lb=0.0, name="w")
    
    #### Set objective ####
    m.setObjective(wvar, GRB.MAXIMIZE)
    m.addConstr(zvar.sum() == k)
    
    #zsol = localsearch(n, k)
    zsol = [0]*n
    for i in range(k):
        zsol[i] = 1
    
    nu, mu = validcut(zsol, n)    
    m.addConstr(wvar <= nu + sum(mu[i]*zvar[i] for i in range(n)))
    
    zsol = [0]*n
    for i in range(n-k,n):
        zsol[i] = 1
    nu, mu = validcut(zsol, n)    
    m.addConstr(wvar <= nu + sum(mu[i]*zvar[i] for i in range(n)))
    
    m.update()  
    m.optimize()
    print(m.objval)
    
        
    itert = 0 # number of iterations
    lb = -1e20
    bestub = m.objval  # initial upper bound
    
    while(itert <= 2000): 
        itert = itert + 1
        
        gamma_t = 0.01
        
        m.addConstr(wvar <= nu + sum(mu[i]*zvar[i] for i in range(n)))
        m.update()
        
        lb, ub, a, mut, zt = subgrad(zsol, mu, k, gamma_t)
        nu = a[0]
        mu = mut
        zsol = zt
        
        bestub = min(bestub, ub)
        print(lb, bestub)
    
    end = datetime.datetime.now()
    time = (end-start).seconds
    
    return bestub, time

n = 13
k = 5
UB, time = spca_rel(n, k)

