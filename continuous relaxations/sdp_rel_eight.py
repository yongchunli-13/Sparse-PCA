#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:28:42 2019

@author: yongchun
"""


import os
import numpy as np
import pandas as pd
import mosek
from mosek.fusion import *
from numpy import matrix
from numpy import array

n=13
d=13
k=5

temp = pd.read_table(os.getcwd()+'/pitdata.csv',
                    header=None,encoding = 'utf-8',sep=',')

# temp = pd.read_table(os.getcwd()+'/Matrix_Eisen_Data_2_txt',
#                 header=None,encoding = 'utf-8',sep=',')

temp = np.array(temp)
A = np.matrix(temp)

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
    
#V = pd.read_csv(os.getcwd()+'/data2000.csv',encoding = 'utf-8',sep=',')
#V = matrix(V)
#V = V/100
#V  = V[:, 1:950]
#
#S = [[V[i].T * V[i] for i in range(n)]] # change the set
#S = S[0]

def abs(M, t, x):
    M.constraint(Expr.add(t,x), Domain.greaterThan(0.0))
    M.constraint(Expr.sub(t,x), Domain.greaterThan(0.0))
    
def norm1(M, t, x):
    u = M.variable(x.getShape(), Domain.unbounded())
    abs(M, u, x)
    M.constraint(Expr.sub(t, Expr.sum(u)), Domain.equalsTo(0.0))


with Model("SDPrelaxation") as M:
    X = M.variable('X', Domain.inPSDCone(d))
    W1 = M.variable('W1', Domain.inPSDCone(d,n))
    z = M.variable('z',n, Domain.inRange(0.0,1.0))
    y = M.variable('y',n, Domain.unbounded())
   
    
    M.constraint(Expr.sum(X.diag()), Domain.equalsTo(1.0))
    #M.constraint(Expr.sub(X.diag(),z), Domain.lessThan(0.0))
    M.constraint(Expr.sum(z), Domain.equalsTo(k))
    
    mu = [0]*n
    Q = [0]*n
    
    for i in range(n):
        mu[i] = M.constraint(Expr.sub(Expr.sum(W1.pick([[i,j1,j2] for j1 in range(d) for j2 in range(d)]).reshape(d,d).diag()),
                              z.index(i)), Domain.equalsTo(0.0))
       
#        M.constraint(Expr.add(Expr.sum(W2.pick([[i,j1,j2] for j1 in range(n) for j2 in range(n)]).reshape(n,n).diag()), 
#                              z.index(i)), Domain.lessThan(1.0))
#        
        Q[i] = M.constraint(Expr.sub(X, W1.pick([[i,j1,j2] for j1 in range(d) for j2 in range(d)]).reshape(d,d)), Domain.inPSDCone(d))
        
        M.constraint(Expr.sub(y.index(i), Expr.sum(Expr.mulElm(S[i], 
                                            W1.pick([[i,j1,j2] for j1 in range(d) for j2 in range(d)]).reshape(d,d)))), Domain.equalsTo(0.0))
        
    
    M.objective("obj",ObjectiveSense.Maximize, Expr.sum(y) ) 
     
    M.setSolverParam('mioMaxTime', 100.0)
    
    M.solve()
    print (M.getPrimalSolutionStatus())
    print (M.primalObjValue())
    print (M.getSolverDoubleInfo("optimizerTime"))
    zvar = M.getVariable('z').level()
    x=M.getVariable('X').level()
    x= x.reshape(n,n)
    a,b=np.linalg.eig(x)
    
    musol = [0]*n
    for i in range(n):
        musol[i] = mu[i].dual()[0]
        
    Qsol = [0]*n
    for i in range(n):
        Qsol[i] = -(Q[i].dual()).reshape(n,n)
    
    val = 0.0  
    for i in range(n):
        val = val + Qsol[i]    
    a,b = np.linalg.eigh(Qsol[0]) # compute the largest eigenvalue

    max(a)+sum(musol[i]*zvar[i] for i in range(n))
    
    val = 0.0  
    for i in range(n):
        val = val + (1-musol[i]/T[i])*S[i]    
    a,b = np.linalg.eigh(val)
    
    