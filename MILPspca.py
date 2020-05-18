#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:19:19 2020

@author: yongchun
"""

from gurobipy import *
import math
import datetime
import os
import numpy as np
import pandas as pd
from scipy.linalg import svd
import sys
from gurobipy import GRB
from scipy.linalg import eigh as largest_eigh

def gen_data(n):
    
    global S
    global E
    global V
    global C
    global T
    
#    temp = pd.read_table(os.getcwd()+'/pitdata.csv',
#                        header=None,encoding = 'utf-8',sep=',')
    
    temp = pd.read_table(os.getcwd()+'/Matrix_Eisen_Data_2_txt',
                         header=None,encoding = 'utf-8',sep=',')

    temp = np.array(temp)
    C = np.matrix(temp)
    
#    
    ## Cholesky factorization of C    
    U, s, V = svd(C) # SVD decomposition
    s[s<1e-6]=0 
    
    
    sqrt_eigen = [0]*n
    for i in range(n):
        if s[i]>0:
            sqrt_eigen[i] = np.sqrt(s[i])
        else:
            sqrt_eigen[i] = 0
               
    V = np.dot(np.diag(sqrt_eigen),V)  
    d = np.linalg.matrix_rank(C)  
    V = V[0:d,:]
    
    V = np.matrix(V)
    V = V.T #size of V is n*d

    S = [[V[i].T * V[i] for i in range(n)]] # change the set
    S = S[0]
    
    T = [0]*n
    for i in range(n):
        T[i] = (V[i] * V[i].T)[0,0] 
    E=np.eye(n, dtype=int)
    V = V.T
    
def cuttwo(z, n):
    nu = 0
    mu = [0]*n
    
    sel_index = [i for i in range(n) if z[i] ==1]
    uns_index = [i for i in range(n) if z[i] == 0]

    val = 0
    for i in sel_index:
        val = val + S[i]
        
    a,b = largest_eigh(val)
    nu = max(a)
    mu = [0]*n
    for i in uns_index:
        mu[i] = T[i]
    #print(nu)
    return nu, mu

#### Ser parameters ####
def truncation(n, k):
    LB=[0]*n
    UB = [0]*n
    for i in range(n):
        a = C[i]
        a= abs(a)
        sindex = np.argsort(-a)  
        b = [0]*n
        for j in range(k):
            b[sindex[0,j]] = a[0,sindex[0,j]]
        
        LB[i] = np.linalg.norm(b,2)
        UB[i] = sum(b)
    return max(LB), max(UB)

def spca(n, d, nm, k):
    intzsol = [0]*n
    
    temp = np.array(T)
    index=np.argsort(-temp)
    
    val=0
    for i in range(k):
        #print(index[i])
        val =val + S[index[i]]
        
        intzsol[index[i]] = 1
        
    sa, sb = largest_eigh(val)
    LB1 = max(sa)
    LB2, UB2 = truncation(n, k)
    print(LB1, LB2)
    LB = max(LB1, LB2)
    
    a,b = largest_eigh(C)
    UB = min(max(a), UB2)
    #UB = 2.94949223505867
    print(max(a), UB2)
    #UB=4.1727
    print ("LB=%f,UB=%f" %(LB,UB))

        
    #### Set model ####
    coeff = [0]*nm
    for i in range(nm):
        coeff[i] = (UB-LB)*2**(-i)
            
    #### Set model ####
    m = Model("mipspca")
    
    m.params.timelimit = 3600.0
    
    #### Creat variables ####
    lambdavar = m.addMVar(shape = 1, lb=0.0,  name="lambda")
    #x = m.addVars(n, vtype=GRB.CONTINUOUS, name="xv")
    x = m.addMVar(shape=d, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    R = m.addMVar(shape=d, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="r")
    B = m.addMVar(shape=n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
    
    z = m.addMVar(n, vtype=GRB.BINARY, name="z")

    u1 = m.addMVar((d,n), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u1")
    #u1 = m.addVars(n,n, vtype=GRB.CONTINUOUS, name="u1")
    u2 = m.addMVar((d,n), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u2")
    
    w = m.addMVar((d,d), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name = "w") 
    y = m.addMVar(d, vtype=GRB.BINARY, name="y")
    #y = m.addVars(n, vtype=GRB.BINARY, name="y")
    
    mu1 = m.addMVar((d,nm), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu1")
    mu2 = m.addMVar((d,nm), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu2")
    alpha = m.addMVar(nm, vtype=GRB.BINARY, name="z")
    
    #### Set objective ####
    m.setObjective(1*lambdavar, GRB.MAXIMIZE)
    
    #### Define constraints ####    
    m.addConstr(z.sum() == k) # size constraint
    m.addConstr(y.sum() == 1)
    
    ## eigenvalue ##
    for i in range(n):
        m.addConstr(B[i] == V[:,i].T @ u1[:,i])
    
#    m.addConstr(sum(S[i] @ u1[:,i] for i in range(n)) - UB*x + sum(coeff[i] * mu1[:,i] for i in range(nm)) == R) # eigenvalues
        
    m.addConstr(V @ B - UB*x + sum(coeff[i] * mu1[:,i] for i in range(nm)) == R) # eigenvalues
    
    for i in range(d):
        m.addConstr(R[i] <= (UB-LB)*2**(-nm))
        m.addConstr(R[i] >= -(UB-LB)*2**(-nm))
    
    #m.addConstr(V.T @ B - UB*x + sum(coeff[i] * mu1[:,i] for i in range(nm)) >= -(UB-LB)*2**(-nm)) # eigenvalues
 
    ## lambda approx ###
    m.addConstr(sum(coeff[i]*alpha[i] for i in range(nm)) + lambdavar == UB)
    
    ## inf norm ##
    m.addConstr(sum(w[:,i] for i in range(d)) == x)
    
    for i in range(d):
        m.addConstr(w[i,i] == y[i])
        for j in range(d):
            m.addConstr(w[j,i] <= y[i])
            m.addConstr(w[j,i] >= -y[i])
            
    ## left-hand binary ##      
    for i in range(n):
        m.addConstr(u1[:,i] + u2[:,i] == x)
        for j in range(d):
            m.addConstr(u1[j,i] <= z[i])
            m.addConstr(u1[j,i] >= -z[i])
            m.addConstr(u2[j,i] <= 1-z[i])
            m.addConstr(u2[j,i] >= z[i]-1)
  
    ## right-hand binary ##
    for i in range(nm):
        m.addConstr(mu1[:,i] + mu2[:,i] == x)
        for j in range(d):
            m.addConstr(mu1[j,i] <= alpha[i])
            m.addConstr(mu1[j,i] >= -alpha[i])
            m.addConstr(mu2[j,i] <= 1-alpha[i])
            m.addConstr(mu2[j,i] >= alpha[i]-1)
            
    for i in range(n):
        z[i].start = intzsol[i]
        
    m.params.timelimit = 3600 
    m.optimize()
            
   # MIPGap = 0.00001
#    m.Params.MIPGap= 1e-4
#    m.params.LazyConstraints = 1
#    m.params.OutputFlag = 1
#    m.params.timelimit = 3600 
#    m.optimize(lazycuts)

n=118
d=78
nm=10
k=20
gen_data(n)
spca(n, d, nm, k)

#def spca(n, nm, k):
#    
#    val=0
#    arr=np.array([-np.trace(S[i]) for i in range(n)])
#    index=np.argsort(arr)
#    for i in range(k):
#        val =val+S[index[i]]
#    U, sa, V = svd(val)
#    LB1 = max(sa)
#    LB2 = truncation(n, k)
#    LB = max(LB1, LB2)
#    UB = min(max(s), math.sqrt(k)*LB2)
#    print ("LB=%f,UB=%f" %(LB,UB))
#
#        
#    #### Set model ####
#    m = Model("mipspca")
#    
#    m.params.timelimit = 3600.0
#    coeff = [0]*nm
#    for i in range(nm):
#        coeff[i] = (UB-LB)*2**(-i)
#    #### Set model ####
#    m = Model("mipspca")
#    
#    m.params.timelimit = 3600.0
#    
#    #### Creat variables ####
#    lambdavar = m.addMVar(shape = 1, lb=0.0,  name="lambda")
#    #x = m.addVars(n, vtype=GRB.CONTINUOUS, name="xv")
#    x = m.addMVar(shape=n, lb=-1.0, ub = 1.0, vtype=GRB.CONTINUOUS, name="x")
#    z = m.addMVar(n, vtype=GRB.BINARY, name="z")
#
#    u1 = m.addMVar((n,n), lb=-1.0,ub = 1.0, vtype=GRB.CONTINUOUS, name="u1")
#    #u1 = m.addVars(n,n, vtype=GRB.CONTINUOUS, name="u1")
#    u2 = m.addMVar((n,n), lb=-1.0,ub = 1.0, vtype=GRB.CONTINUOUS, name="u2")
#    
#    w = m.addMVar((n,n), lb=-1.0,ub = 1.0, vtype=GRB.CONTINUOUS, name = "w") 
#    y = m.addMVar(shape=n, vtype=GRB.BINARY, name="y")
#    #y = m.addVars(n, vtype=GRB.BINARY, name="y")
#    
#    mu1 = m.addMVar((n,nm), lb=-1.0,ub = 1.0, vtype=GRB.CONTINUOUS, name="mu1")
#    mu2 = m.addMVar((n,nm), lb=-1.0,ub = 1.0, vtype=GRB.CONTINUOUS, name="mu2")
#    alpha = m.addMVar(nm, vtype=GRB.BINARY, name="z")
#    
#    #### Set objective ####
#    m.setObjective(1*lambdavar, GRB.MAXIMIZE)
#    
#    #### Define constraints ####    
#    m.addConstr(z.sum() == k) # size constraint
#    m.addConstr(y.sum() == 1)
#    
#    ## eigenvalue ##
#    m.addConstr(sum(S[i] @ u1[:,i] for i in range(n)) - UB*x + sum(coeff[i] * mu1[:,i] for i in range(nm)) <= (UB-LB)*2**(-nm)) # eigenvalues
#    
#    m.addConstr(sum(S[i] @ u1[:,i] for i in range(n)) - UB*x + sum(coeff[i] * mu1[:,i] for i in range(nm)) >= -(UB-LB)*2**(-nm)) # eigenvalues
# 
#    ## lambda approx ###
#    m.addConstr(sum(coeff[i]*alpha[i] for i in range(nm)) + lambdavar == UB)
#    #m.addConstr(lambdavar <= UB)
#    #m.addConstr(lambdavar >= LB)
#    
#    ## inf norm ##
#    m.addConstr(sum(w[:,i] for i in range(n)) == x)
#    
#    for i in range(n):
#        m.addConstr(w[i,i] == y[i])
#        for j in range(n):
#            m.addConstr(w[j,i] <= y[i])
#            m.addConstr(w[j,i] >= -y[i])
#            
#    ## left-hand binary ##      
#    for i in range(n):
#        m.addConstr(u1[:,i] + u2[:,i] == x)
#        for j in range(n):
#            m.addConstr(u1[j,i] <= z[i])
#            m.addConstr(u1[j,i] >= -z[i])
#            m.addConstr(u2[j,i] <= 1-z[i])
#            m.addConstr(u2[j,i] >= z[i]-1)
#  
#    ## right-hand binary ##
#    for i in range(nm):
#        m.addConstr(mu1[:,i] + mu2[:,i] == x)
#        for j in range(n):
#            m.addConstr(mu1[j,i] <= alpha[i])
#            m.addConstr(mu1[j,i] >= -alpha[i])
#            m.addConstr(mu2[j,i] <= 1-alpha[i])
#            m.addConstr(mu2[j,i] >= alpha[i]-1)
#            
#   # MIPGap = 0.00001
#  
#    m.optimize()
#
#n=63
#nm=13
#k=10
#gen_data(n)
#spca(n, nm, k)
