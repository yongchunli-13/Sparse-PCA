#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:11:31 2020

@author: yongchun
"""
from gurobipy import *
from gurobipy import GRB
import math
import datetime
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from uci_datasets import Dataset
from uci_dataset.load_data import *
from sklearn.datasets import load_digits

## Import data
def gen_data(n):
    global S
    global E
    global V
    global A
    global T
    global d
    global A
    
    # temp = pd.read_table(os.getcwd()+'/Matrix_Eisen_Data_1_txt',
    #             header=None,encoding = 'utf-8',sep=',')

    # temp = np.array(temp)
    # A = np.matrix(temp)
    
    # temp = pd.read_table(os.getcwd()+'/pitdata.csv',
    #                     header=None,encoding = 'utf-8',sep=',')
    
    # temp = np.array(temp)
    # A = np.matrix(temp)
    
    # data = pd.read_table(os.getcwd()+'/spambase/spambase.data',  header=None,
    #           encoding = 'utf-8',sep=',')
    # temp = data.drop([57], axis=1)
    

    # data = pd.read_table(os.getcwd()+'/wdbc.csv',
    #           encoding = 'utf-8',sep=',')
    # temp = data.drop(['Unnamed: 0', '1'], axis=1)
    
    digits = load_digits()
    temp = digits.data
    
    # df = load_dermatology()
    # temp = df.drop(['class'], axis=1)
    # temp = temp.fillna(0.0)
    
    # data = Dataset("gas")
    # temp = data.x
    temp = preprocessing.normalize(temp)/10 ## normalize data
    temp = np.array(temp)
    A = np.matrix(temp)
    A = A.T*A
    
    

def spca(n, s):
    start = datetime.datetime.now()
    gen_data(n)
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
    m.Params.MIPGap= 1e-5
    m.optimize()
    
    end = datetime.datetime.now()
    time = (end-start).seconds
    return m.objval, m.ObjBound, time
            
