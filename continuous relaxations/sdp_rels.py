#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 16:16:10 2023

@author: yongchunli
"""

import os
import numpy as np
import pandas as pd
import mosek
from mosek.fusion import *
from numpy import matrix
from numpy import array
from scipy.linalg import sqrtm
from sklearn import preprocessing
from uci_datasets import Dataset
import datetime
from sklearn.datasets import load_digits
from uci_dataset.load_data import *

## Import data
def gen_data(n):
    global S
    global E
    global V
    global A
    global T
    global d
    global A
    
    temp = pd.read_table(os.getcwd()+'/Matrix_Eisen_Data_2_txt',
                header=None,encoding = 'utf-8',sep=',')
    temp = np.array(temp)
    A = np.matrix(temp)

    data = Dataset("pol")
    temp = data.x
    temp = preprocessing.normalize(temp)/10 ## normalize data
    temp = np.array(temp)
    A = np.matrix(temp)
    A = A.T*A
    
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
 
    

##-------------- SDP relaxation (8) --------------
def sdp_rel_eight(n, k):
    start = datetime.datetime.now()
    with Model("SDPrelaxation") as M:
        X = M.variable('X', Domain.inPSDCone(d))
        W1 = M.variable('W1', Domain.inPSDCone(d,n))
        z = M.variable('z',n, Domain.inRange(0.0,1.0))
        y = M.variable('y',n, Domain.unbounded())
       
        
        M.constraint(Expr.sum(X.diag()), Domain.equalsTo(1.0))
        M.constraint(Expr.sum(z), Domain.equalsTo(k))
        
        for i in range(n):
            M.constraint(Expr.sub(Expr.sum(W1.pick([[i,j1,j2] for j1 in range(d) for j2 in range(d)]).reshape(d,d).diag()),
                                  z.index(i)), Domain.equalsTo(0.0))
            M.constraint(Expr.sub(X, W1.pick([[i,j1,j2] for j1 in range(d) for j2 in range(d)]).reshape(d,d)), Domain.inPSDCone(d))
            
            M.constraint(Expr.sub(y.index(i), Expr.sum(Expr.mulElm(S[i], 
                                                W1.pick([[i,j1,j2] for j1 in range(d) for j2 in range(d)]).reshape(d,d)))), Domain.equalsTo(0.0))
            
        
        M.objective("obj",ObjectiveSense.Maximize, Expr.sum(y) ) 
         
        M.setSolverParam('mioMaxTime', 3600.0)
        
        M.solve()
        print (M.getPrimalSolutionStatus())
        print (M.primalObjValue())
        print (M.getSolverDoubleInfo("optimizerTime"))
        fval = M.primalObjValue()
        
    end = datetime.datetime.now()
    time = (end-start).seconds
    return fval, time


##-------------- continuous relaxation (16) --------------
    
def abs(M, t, x):
    M.constraint(Expr.add(t,x), Domain.greaterThan(0.0))
    M.constraint(Expr.sub(t,x), Domain.greaterThan(0.0))
    
def norm1(M, t, x):
    u = M.variable(x.getShape(), Domain.unbounded())
    abs(M, u, x)
    M.constraint(Expr.sub(t, Expr.sum(u)), Domain.equalsTo(0.0))


def sdp_rel_sixteen(n, k):
    start = datetime.datetime.now()
    with Model("SDPrelaxation") as M:
        X = M.variable('X', Domain.inPSDCone(n))
        z = M.variable('z',n, Domain.inRange(0.0,1.0))
        u = M.variable(n, Domain.unbounded())
       
        
        M.constraint(Expr.sum(X.diag()), Domain.lessThan(1.0))
        M.constraint(Expr.sum(z), Domain.equalsTo(k))
        
        for i in range(n):
            index = [[i,j] for j in range(n)]
             #first inequality
            M.constraint(Expr.vstack(X.index(i,i), Expr.mul(z.index(i),0.5), X.pick([[i,j] for j in range(n)])), Domain.inRotatedQCone(n+2))
             #second inequality
            norm1(M, u.index(i), X.pick([[i,j] for j in range(n)]))
            M.constraint(Expr.vstack(X.index(i,i), Expr.mul(z.index(i),0.5*k), u.index(i)), Domain.inRotatedQCone(3))
        
        M.objective("obj",ObjectiveSense.Maximize, Expr.sum( Expr.mulElm(A, X) )) 
         
        M.setSolverParam('mioMaxTime', 3600.0)
        M.solve()
        print (M.getPrimalSolutionStatus())
        print (M.primalObjValue())
        print (M.getSolverDoubleInfo("optimizerTime"))
        fval = M.primalObjValue()
    
    end = datetime.datetime.now()
    time = (end-start).seconds
    return fval, time


##-------------- SDP relaxation (17) --------------
def sdp_rel_seventeen(n, k):
    start = datetime.datetime.now()
    with Model("SDPrelaxation") as M:
        X = M.variable('X', Domain.inPSDCone(n))
        t = M.variable(Domain.greaterThan(0.0))
        norm1(M, t, X)
        
        M.constraint(t, Domain.lessThan(k))
        M.constraint(Expr.sum(X.diag()), Domain.lessThan(1.0))
        M.objective("obj",ObjectiveSense.Maximize, Expr.sum( Expr.mulElm(A, X) )) 
         
        M.setSolverParam('mioMaxTime', 3600.0)
        M.solve()
        print (M.getPrimalSolutionStatus())
        print (M.primalObjValue())
        print (M.getSolverDoubleInfo("optimizerTime"))
        fval = M.primalObjValue()
    
    end = datetime.datetime.now()
    time = (end-start).seconds
    return fval, time



    
    
    