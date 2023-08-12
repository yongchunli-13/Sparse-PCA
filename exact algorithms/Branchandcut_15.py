#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:18:53 2020

@author: yongchun
"""


import os
from gurobipy import *
from gurobipy import GRB
import math
import numpy as np
import math
import datetime
import pandas as pd
from numpy import matrix
from numpy import array
import cuts_15


def spca_bc(n, k):   
    gen_data = cuts_15.gen_data
    cut = cuts_15.benderscut19
    localsearch = cuts_15.localsearch
    gen_data(n)

    def lazycuts(m,where):
        if where == GRB.callback.MIPSOL:
            y = m.cbGetSolution(zvar)
            yy = [0]*n
            for i in range(n):
                if y[i] > 0.5:
                    yy[i] = 1
                else:
                    yy[i] = 0
                    
            nu, mu  = cut(yy, n, k)
            
            expr = wvar
            rhs = nu
            for i in range(n):
                expr = expr - mu[i]*zvar[i]
        
            m.cbLazy(expr,GRB.LESS_EQUAL,rhs)
            
            
    #### Set model ####
    start = datetime.datetime.now()
    
    m = Model("spca")
    
    #### Creat variables ####
    zvar = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")
    wvar = m.addVar(obj=1.0, lb=0.0, name="w")
    
    #### Set objective ####
    m.setObjective(wvar, GRB.MAXIMIZE)
    m.addConstr(zvar.sum() == k)
    
    ltime, LB, zsol = localsearch(n, k)
    nu, mu  = cut(zsol, n, k)  
    m.addConstr(wvar <= nu + sum(mu[i]*zvar[i] for i in range(n)))
    m.optimize()
    
    zsol = [0]*n
    for i in range(n):
        zsol[i] = zvar[i].x
        
    itert = 0
    print('objetive value', nu + sum(mu[i]*zsol[i] for i in range(n)), 'upper bound', wvar.x)
    while(itert <= 50):
        itert  = itert + 1
        nu, mu  = cut(zsol, n, k)
        m.addConstr(wvar <= nu + sum(mu[i]*zvar[i] for i in range(n)))
        
        print('objetive value', nu + sum(mu[i]*zsol[i] for i in range(n)), 'upper bound', wvar.x)
        m.params.OutputFlag = 0
        m.optimize()
        zsol = [0]*n
        for i in range(n):
            zsol[i] = zvar[i].x
    
    for i in range(n):
        zvar[i].vtype=GRB.BINARY 
    
    m.Params.MIPGap= 1e-4
    m.params.LazyConstraints = 1
    m.params.OutputFlag = 1
    m.params.timelimit = 3600 
    m.optimize(lazycuts)
        
    end = datetime.datetime.now()
    time = (end-start).seconds
    print(time)
    
    zsol = [0]*n
    for i in range(n):
        zsol[i] = zvar[i].x
        
    return m.objval,  m.ObjBound, time
