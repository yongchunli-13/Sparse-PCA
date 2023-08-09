import datetime
import os
import numpy as np
import pandas as pd
import math
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
    
    # digits = load_digits()
    # temp = digits.data
    
    data = Dataset("song")
    temp = data.x
    
    # df = load_dermatology()
    # temp = df.drop(['class'], axis=1)
    # temp = temp.fillna(0.0)
    
    # data = Dataset("gas")
    # temp = data.x
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
 
    

def power_iteration(B):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(B.shape[1])
    b_k1 = np.array(B).dot(np.array(b_k))
    b_k1_norm = np.linalg.norm(b_k1)
    b_k = b_k1 / max(b_k1_norm, 1e-10)
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
 
##-------------- greedy algorithm --------------  
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

##-------------- local search algorithm --------------
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


##-------------- given binary z, optimality cut in Proposition 1 --------------
## given any binary z, compute the coefficients of vaild inequality    
def validcut(z, n):
    nu = 0
    mu = [0]*n
    
    sel_index = [i for i in range(n) if z[i] > 0.5]
    uns_index = [i for i in range(n) if z[i] < 0.5]

    val = 0
    for i in sel_index:
        val = val + S[i]
        
    a,b = power_iteration(val) # compute the largest eigenvalue
    nu = a
    mu = [0]*n
    for i in uns_index:
        mu[i] = T[i]
    return nu, mu


##--------------given continuous z, obtain cut by continuous relaxation (12) --------------
## compute the continuous relaxation problem (12)        
def continuous_cut(zsol, n):
    start = datetime.datetime.now()
    mu = [0]*n
    for i in range(n):
        if zsol[i] < 0.5:
            mu[i] = T[i]  
                
    val = 0.0  
    for i in range(n):
        if T[i] > 0:
            val = val + (1-mu[i]/T[i])*S[i]  
        else:
            val = val + S[i]

    a,b = power_iteration(val) # compute the largest eigenvalue
    bestub = a + sum(mu[i]*zsol[i] for i in range(n)) # initial upper bound
    bestmu = mu
    besta = a
    itert = 0 # number of iterations
    
    while(itert <= 1000): 
        itert = itert + 1
        
        gamma_t = 1/math.sqrt(8*itert)
        
        subg = [0.0]*n
        for i in range(n):
            if T[i] > 0:
                subg[i] = zsol[i] - ((np.matrix(b)*V[:,i])[0,0])**2/T[i] 

            # subg[i] = zsol[i] - ((np.matrix(b)*V[:,i])[0,0])**2/T[i] 
        
        musol = np.array(mu) - gamma_t*np.array(subg)
        musol[musol < 0] = 0.0
        for i in range(n):
            if musol[i] > T[i]:
                musol[i] = T[i]
        mu = musol
        
        val = 0.0  
        for i in range(n):
            if T[i] > 0:
                val = val + (1-mu[i]/T[i])*S[i]  
            else:
                val = val + S[i]
        a,b = power_iteration(val) # compute the largest eigenvalue
        ub = a + sum(mu[i]*zsol[i] for i in range(n)) # upper bound
        
        if bestub > ub:
            besta = a
            bestub = ub
            bestmu = mu
        

    end = datetime.datetime.now()
    time = (end-start).seconds
    return  besta, bestmu

