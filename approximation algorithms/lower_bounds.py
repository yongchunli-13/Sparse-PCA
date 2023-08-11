import datetime
import os
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
from uci_datasets import Dataset
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import sqrtm


## Import data
def gen_data(n):
    global A

    data = Dataset("pol")
    temp = data.x
    temp = preprocessing.normalize(temp)/10 ## normalize data
    temp = np.array(temp)
    A = np.matrix(temp)
    A = A.T*A
 
    
    
def power_iteration(B):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(B.shape[1])
    b_k1 = np.array(B).dot(np.array(b_k))
    b_k1_norm = np.linalg.norm(b_k1)
    b_k = b_k1 /max(b_k1_norm, 1e-10)
    eigenval = 0
    
    while(abs(b_k1_norm-eigenval)/max(b_k1_norm, 1e-10) >= 1e-6):
        eigenval = b_k1_norm
        
        # calculate the matrix-by-vector product Ab
        b_k1 = np.array(B).dot(np.array(b_k))

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
        
        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k1_norm, b_k    
 
#### greedy algorithm ####    
def greedy(n, k): 
    
    c = 1
    x = [0]*n # chosen set
    y = [1]*n # unchosen set
    indexN = np.flatnonzero(y)
     
    fval = 1 
    
    start = datetime.datetime.now()
    
    sel = []
 
    while c < k+1: 
        
        fval = []
        for i in indexN:
            sel.append(i)
            temp = A[np.ix_(sel,sel)]
            a,b = power_iteration(temp)
            fval.append(a)
            sel.remove(i)
            
        tempi = np.argmax(fval)
#        print(max(fval))
        opti = indexN[tempi]
        x[opti] = 1
        y[opti] = 0
        
        sel = np.flatnonzero(x)
        sel = list(sel)
        indexN = np.flatnonzero(y)

        c = c + 1
          
    end = datetime.datetime.now()
    time = (end - start).seconds
    bestf = max(fval)
    
    return time, x, bestf

#### local search algorithm ####
def localsearch(n, k):
    start = datetime.datetime.now()
    
    gtime, zsol, fval = greedy(n,k)
        
    sel = np.flatnonzero(zsol)
    t = [i for i in range(n) if zsol[i] == 0]
    bestz = zsol
    
    bestf = fval
#    print(bestf)
    optimal = False

    while(optimal == False):
        optimal = True
        for i in sel:
            for j in t:
                tempz = [0]*n
                for g in range(n):
                    tempz[g] = bestz[g]
                tempz[i] = 0
                tempz[j] = 1
                tempsel = np.flatnonzero(tempz)
                temp = A[np.ix_(tempsel,tempsel)]
                a,b = power_iteration(temp)

                if a > bestf:
#                    print(a)
                    optimal = False                
                    bestz = tempz  # update solution                 
                    bestf = a # update the objective value
                    
                    sel = np.flatnonzero(bestz) # update chosen set
                    t = [i for i in range(n) if bestz[i] == 0] # update the unchosen set
                    break
                
    end = datetime.datetime.now()
    time = (end-start).seconds
    
    return time, bestf, bestz


#### truncation algorithm ####
def truncation(n, k):
    start = datetime.datetime.now()
    LB=[0]*n
    for i in range(n):
        a = A[i]
        a= abs(a)
        sindex = np.argsort(-a)  
        b = [0]*n
        for j in range(k):
            b[sindex[0,j]] = a[0,sindex[0,j]]
        
        bnorm = np.linalg.norm(b,2)
        if bnorm == 0.0:
            LB[i] = 0.0
        else:
            b = b/bnorm
            
            b = np.matrix(b)
            
            LB[i] = (b*A*b.T)[0,0]
    LB1 =  max(max(LB),max(np.diag(A)))
    
    a, b = power_iteration(A)
    b = abs(b)
    sindex = np.argsort(-b)  
    x = [0]*n
    for j in range(k):
        x[sindex[j]] = b[sindex[j]]
    xnorm = np.linalg.norm(x,2)
    x = x/xnorm
    x =  np.matrix(x)
    LB2 = (x*A*x.T)[0,0]
    
    end = datetime.datetime.now()
    time = (end-start).seconds
            
    return time, max(LB1, LB2)


def randomized_SPCA(n, s):
    data = np.matrix(np.real(sqrtm(A)))

    Z_F2= np.linalg.norm(data, 'fro')**2 # Calculate the Frebenius norm
    p_tilde= np.linalg.norm(data,axis=0)**2/Z_F2
    p=np.minimum(np.array(p_tilde)*(s),np.ones(data.shape[1],dtype=int)) #min(sp,1)
        
    idx=np.array([],dtype="int32")

    while idx.shape[0]==0:

        sample=np.random.binomial(1,p,len(p))
        
        inxs = [i for i in range(n) if sample[i] > 0.5]
        
        S=[0]*n
        for i in inxs:
            if p[i] > 0:
                S[i] =  1/np.sqrt(p[i])
            else:
                S[i] = 0
        
        S= np.matrix(np.diag(S))
        # S= np.diag(np.multiply(sample,np.reciprocal(np.sqrt(p)))) # sampling and rescaling matrix

        # top right singular vector of ZS
        
        U, Sig, VT =randomized_svd(data*S, n_components=1, random_state=10)
        y_rspca=np.matmul(S,VT[0]) # VT[0]: top right singular vector of ZS

        # SVD normalization
        dummy=np.where(sample!=0)[0]
        idx=np.append(idx,dummy)

    A_red=A[np.ix_(idx, idx)].copy()
    
    UA, SigA, VTA =randomized_svd(A_red, n_components=1, random_state=10)

    y_rspca_norm = np.zeros(len(p))
    y_rspca_norm[idx]=VTA[0].copy()  ### padding other elements with zeros
    
    y_rspca_norm = y_rspca_norm/np.linalg.norm(y_rspca_norm,2)
    sel = np.flatnonzero(y_rspca_norm)
    # print(len(sel), np.linalg.norm(y_rspca_norm,2))

    f_rspca= (np.matrix(y_rspca_norm)*A*np.matrix(y_rspca_norm).T)[0,0]
    # np.sum(np.square(np.matmul(data,y_rspca_norm)))


    # print("rspca done.", f_rspca)
    return y_rspca_norm, f_rspca, len(sel)

