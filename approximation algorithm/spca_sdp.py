
import numpy as np #for all numerical computations
import pandas as pd #similar to dataframe in R
import scipy as sp #scientific computations (includes stats)
import matplotlib.pyplot as plt #plotting 
import scipy.io as sio
import cvxpy as cp # convex optimization
import random
import sys
import datetime
import gc
import math
import os
from sklearn.utils.extmath import randomized_svd
from sklearn import preprocessing
from uci_datasets import Dataset
from sklearn.datasets import load_breast_cancer
from uci_dataset.load_data import *

## Import data
def gen_data(n):
    global A
    
    # temp = pd.read_table(os.getcwd()+'/Matrix_CovColon_txt',
    #             header=None,encoding = 'utf-8',sep=',')
    # temp = np.array(temp)
    # A = np.matrix(temp)
    # df = load_dermatology()
    # temp = df.drop(['class'], axis=1)
    # temp = temp.fillna(0.0)
    
    df = load_dermatology()
    temp = df.drop(['class'], axis=1)
    temp = temp.fillna(0.0)
    
    # df = load_diabetic()
    
    # temp = df.drop(['outcome'], axis=1)
    
    # abolone = dataset.load_abolone()
    
    # # temp = pd.read_table(os.getcwd()+'/pitdata.csv',
    # #                     header=None,encoding = 'utf-8',sep=',')

    # data = pd.read_table(os.getcwd()+'/diabetes.csv',
    #           encoding = 'utf-8',sep=',')
    # temp = data.drop(['Unnamed: 0', '1'], axis=1)
    
    # data = Dataset("housing")
    # temp = data.x
        
    # digits = load_breast_cancer()
    # temp = digits.data
    # data = pd.read_table(os.getcwd()+'/spambase/spambase.data',  header=None,
    #           encoding = 'utf-8',sep=',')
    # temp = data.drop([57], axis=1)
    temp = preprocessing.normalize(temp)/10 ## normalize data
    
    temp = np.array(temp)
    A = np.matrix(temp)
    A = A.T*A
    
# def gen_data(n):
#     global A
    
#     # temp = pd.read_table(os.getcwd()+'/pitdata.csv',
#     #                     header=None,encoding = 'utf-8',sep=',')

#     data = pd.read_table(os.getcwd()+'/spambase/spambase.data',
#                     header=None,encoding = 'utf-8',sep=',')
    
#     temp = data.drop([57], axis=1)
#     temp = preprocessing.normalize(temp)/10 ## normalize data
    
#     temp = np.array(temp)
#     A = np.matrix(temp)
#     A = A.T*A

def spca_sdp(n, s):
    start = datetime.datetime.now()
    gen_data(n)
    #----------------------------------------------------------------
    #------------SPCA SDP--------------------------------------------
    #----------------------------------------------------------------
    # A=np.matmul(np.transpose(data),data)

    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((n,n), symmetric=True)

    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    constraints += [cp.trace(X) <= 1]
    constraints += [cp.sum(cp.abs(X)) <= s]

    prob = cp.Problem(cp.Maximize(cp.trace(A @ X)),
                      constraints)
    prob.solve()
    
    #####---------------
    M=300
    np.random.seed(500)
    G= np.random.normal(0, 1, (X.value.shape[0],M))
    P= np.matmul(X.value,G)
    
    y = np.array([0.0] * M)
    for i in range(M):
        y[i]= (np.matrix(P[:,i])*A*np.matrix(P[:,i]).T)[0,0]
        # np.dot(np.matrix(P[:,i]),np.matmul(A,P[:,i]))


    y_max=np.transpose(P)[np.where(y==np.max(y))].copy() ### maxinaum of all the elements of y
    y_max=y_max[0]
    
    y_max_s=y_max.copy()
    y_max_abs=np.absolute(y_max_s)

    sid=min(10*s,len(y_max))
    max_id_sdp= y_max_abs.argsort()[-sid:][::-1]
    
    prob=y_max_abs[max_id_sdp]/sum(y_max_abs[max_id_sdp])
    max_ids_sdp= set(np.random.choice(max_id_sdp,s,replace=False, p=prob))

    
    allidx_sdp = set(np.arange(0,len(y_max)))
    remidx_sdp = list(allidx_sdp - max_ids_sdp)
    y_max_s[remidx_sdp] = 0
    
    idx=np.where(y_max_s!=0)[0]
    A_red=A[np.ix_(idx, idx)]
    
    #UA_sdp, SigA_sdp, VTA_sdp=np.linalg.svd(A_red)
    UA_sdp, SigA_sdp, VTA_sdp =randomized_svd(A_red, n_components=1, random_state=10)
    z_norm_sdp = np.zeros(len(y_max))
    
    z_norm_sdp[idx]=VTA_sdp[0].copy()  ### padding other elements with zeros
    sel = [i for i in range (n) if z_norm_sdp[i] > 1e-4]
    print(len(sel))
    
    f_sdp= (np.matrix(z_norm_sdp)*A*np.matrix(z_norm_sdp).T)[0,0]
    # np.dot(z_norm_sdp,np.matmul(A,z_norm_sdp))
    nnz_sdp=np.count_nonzero(z_norm_sdp)
    print("sdp done")
    end = datetime.datetime.now()
    time = (end-start).seconds
    return time, f_sdp


