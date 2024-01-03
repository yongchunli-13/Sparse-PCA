
import numpy as np #for all numerical computations
import pandas as pd #similar to dataframe in R
import scipy as sp #scientific computations (includes stats)
import cvxpy as cp # convex optimization
import datetime
import os
from sklearn.utils.extmath import randomized_svd
from sklearn import preprocessing
from uci_datasets import Dataset


def gen_data(n, data_name):
    global A
    data = pd.read_table(os.path.dirname(os.getcwd())+'/datasets/'+data_name+'_txt',
          encoding = 'utf-8',sep=',')
    temp = data.drop(['Unnamed: 0'], axis=1)
    A = np.matrix(np.array(temp))


def spca_sdp(n, data_name, s):
    start = datetime.datetime.now()
    gen_data(n, data_name)
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
    # print(len(sel))
    
    f_sdp= (np.matrix(z_norm_sdp)*A*np.matrix(z_norm_sdp).T)[0,0]
    # np.dot(z_norm_sdp,np.matmul(A,z_norm_sdp))
    nnz_sdp=np.count_nonzero(z_norm_sdp)

    end = datetime.datetime.now()
    time = (end-start).seconds
    return time, f_sdp


