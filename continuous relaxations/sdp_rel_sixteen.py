#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 09:11:01 2019

@author: yongchun
"""

import os
import numpy as np
import pandas as pd
import mosek
from mosek.fusion import *
from numpy import matrix
from numpy import array
import sys


n=79
k=10
temp = pd.read_table(os.getcwd()+'/pitdata.csv',
                        header=None,encoding = 'utf-8',sep=',')
temp = pd.read_table(os.getcwd()+'/Matrix_Eisen_Data_1_txt',
                         header=None,encoding = 'utf-8',sep=',')

#temp = pd.read_table(os.getcwd()+'/Matrix2000.txt',
#                     header=None,encoding = 'utf-8',sep=',')
temp=np.array(temp)
temp=temp.reshape(n,n)
Q= np.matrix(temp)


def abs(M, t, x):
    M.constraint(Expr.add(t,x), Domain.greaterThan(0.0))
    M.constraint(Expr.sub(t,x), Domain.greaterThan(0.0))
    
def norm1(M, t, x):
    u = M.variable(x.getShape(), Domain.unbounded())
    abs(M, u, x)
    M.constraint(Expr.sub(t, Expr.sum(u)), Domain.equalsTo(0.0))


with Model("SDPrelaxation") as M:
    X = M.variable('X', Domain.inPSDCone(n))
    z = M.variable('z',n, Domain.inRange(0.0,1.0))
    #z = M.variable('z', Domain.binary(n))
    u = M.variable(n, Domain.unbounded())
   
    
    M.constraint(Expr.sum(X.diag()), Domain.lessThan(1.0))
   # M.constraint(Expr.sub(X.diag(),z), Domain.lessThan(0.0))
    M.constraint(Expr.sum(z), Domain.equalsTo(k))
   # M.constraint(Expr.sum(X), Domain.lessThan(k))
    
    for i in range(n):
        index = [[i,j] for j in range(n)]
         #first inequality
        M.constraint(Expr.vstack(X.index(i,i), Expr.mul(z.index(i),0.5), X.pick([[i,j] for j in range(n)])), Domain.inRotatedQCone(n+2))
         #second inequality
        norm1(M, u.index(i), X.pick([[i,j] for j in range(n)]))
        M.constraint(Expr.vstack(X.index(i,i), Expr.mul(z.index(i),0.5*k), u.index(i)), Domain.inRotatedQCone(3))
    
    M.objective("obj",ObjectiveSense.Maximize, Expr.sum( Expr.mulElm(Q, X) )) 
     
    #M.setSolverParam('mioMaxTime', 100.0)
    M.setLogHandler(sys.stdout)
    M.solve()
    print (M.getPrimalSolutionStatus())
    print (M.primalObjValue())
    print (M.getSolverDoubleInfo("optimizerTime"))
    x=M.getVariable('X').level()

x= x.reshape(n,n)
