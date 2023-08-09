#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 22:38:49 2023

@author: yongchunli
"""
import datetime
import os
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
from uci_datasets import Dataset
from mosek.fusion import *

## Import data
def gen_data(n):
    global S
    global E, Eh
    global V
    global A
    global T
    global d
    global A
    
    temp = pd.read_table(os.getcwd()+'/Matrix_Eisen_Data_1_txt',
                header=None,encoding = 'utf-8',sep=',')

    temp = np.array(temp)
    A = np.matrix(temp)

    # data = Dataset("pumadyn32nm")
    # temp = data.x
    # temp = preprocessing.normalize(temp)/10 ## normalize data
    # temp = np.array(temp)
    # A = np.matrix(temp)
    # A = A.T*A
    E=np.eye(n, dtype=int)
    Eh = E/2

def benderscut19(z, n, k):
    gen_data(n)
    with Model("SDPcut") as M:
        ladavar = M.variable('lambdav', Domain.greaterThan(0.0))
        mu1 = M.variable('mu1v', n, Domain.unbounded())
        mu2 = M.variable('mu2v', n, Domain.greaterThan(0.0))
        nu1 = M.variable('nu1v', n, Domain.unbounded())
        nu2 = M.variable('nu2v', n, Domain.greaterThan(0.0))
        betavar = M.variable('betavar', n, Domain.unbounded()) 
        W1 = M.variable('W1', [n,n], Domain.greaterThan(0.0))
        W2 = M.variable('W2', [n,n], Domain.greaterThan(0.0))
        Lambda = M.variable('Lambda', [n,n], Domain.unbounded())
        
        
        M.constraint(Expr.sub(Expr.add(Expr.mulElm(E, Var.repeat(ladavar, n*n).reshape([n,n])), W1), 
                              Expr.add(A, Expr.add(Lambda, Expr.add(W2, 
                              Expr.mulElm(Eh, Expr.add(Var.repeat(mu1,1,n),Expr.add(Var.repeat(mu2,1,n),
                              Expr.add(Var.repeat(nu1,1,n), Var.repeat(nu2,1,n))))))))),
                     Domain.inPSDCone(n))
        
        for i in range(n):
            for j in range(n):
                # M.constraint(Expr.sub(W1.index(i,j), W1.index(j,i)), Domain.equalsTo(0.0))
                # M.constraint(Expr.sub(W2.index(i,j), W2.index(j,i)), Domain.equalsTo(0.0))
                # M.constraint(Expr.sub(Lambda.index(i,j), Lambda.index(j,i)), Domain.equalsTo(0.0))
                
                M.constraint(Expr.add(betavar.index(i), Expr.add(W1.index(i,j), W2.index(i,j))), Domain.lessThan(0.0))
                
        for i in range(n):
            M.constraint(Expr.vstack(mu2.index(i), mu1.index(i), Lambda.pick([[i,j] for j in range(n)])), Domain.inQCone(n+2))
            M.constraint(Expr.vstack(nu2.index(i), nu1.index(i), betavar.index(i)), Domain.inQCone(3))
           
 
        M.objective("obj", ObjectiveSense.Minimize, Expr.add(ladavar, 
                                                             Expr.add(Expr.mul(Expr.dot(Expr.sub(mu2,mu1),z), 0.5),
                                                                      Expr.mul(Expr.dot(Expr.sub(nu2,nu1),z), 0.5*k)))) 
        M.solve()
        
        print (M.getPrimalSolutionStatus())
        print (M.primalObjValue())
        
        mu1sol = M.getVariable('mu1v').level()
        mu2sol = M.getVariable('mu2v').level()
        nu1sol = M.getVariable('nu1v').level()
        nu2sol = M.getVariable('nu2v').level()
        betasol = M.getVariable('betavar').level()
        
        W1sol = M.getVariable('W1').level()
        W1sol = W1sol.reshape(n,n)
        W2sol = M.getVariable('W2').level()
        W2sol = W1sol.reshape(n,n)
        Lambdasol = M.getVariable('Lambda').level()
        Lambdasol = Lambdasol.reshape(n,n)
        
    return Lambdasol, betasol, mu1sol, mu2sol, nu1sol, nu2sol, W1sol, W2sol


