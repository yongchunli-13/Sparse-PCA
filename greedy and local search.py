import datetime
import os
import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

def gen_data(n):
    
    global S
    global E
    global V
    global C
    global T
    
    
    temp = pd.read_table(os.getcwd()+'/Matrix_Reddit_2000_txt', header=None,encoding = 'utf-8',sep=',')    
    C = np.array(temp)
    C = np.matrix(C)
    

def power_iteration(A):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])
    b_k1 = np.array(A).dot(np.array(b_k))
    b_k1_norm = np.linalg.norm(b_k1)
    b_k = b_k1 / b_k1_norm
    eigenval = 0
    
    while(abs(b_k1_norm-eigenval)/b_k1_norm >= 1e-6):
        eigenval = b_k1_norm
        
        # calculate the matrix-by-vector product Ab
        b_k1 = np.array(A).dot(np.array(b_k))

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
            temp = C[np.ix_(sel,sel)]
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
                temp = C[np.ix_(tempsel,tempsel)]
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
        a = C[i]
        a= abs(a)
        sindex = np.argsort(-a)  
        b = [0]*n
        for j in range(k):
            b[sindex[0,j]] = a[0,sindex[0,j]]
        
        bnorm = np.linalg.norm(b,2)
        b = b/bnorm
        
        b = np.matrix(b)
        
        LB[i] = (b*C*b.T)[0,0]
    LB1 =  max(max(LB),max(np.diag(C)))
    
    a, b = largest_eigsh(C,1)
    b = b[:,0]
    b = abs(b)
    sindex = np.argsort(-b)  
    x = [0]*n
    for j in range(k):
        x[sindex[j]] = b[sindex[j]]
    xnorm = np.linalg.norm(x,2)
    x = x/xnorm
    x =  np.matrix(x)
    LB2 = (x*C*x.T)[0,0]
    
    end = datetime.datetime.now()
    time = (end-start).seconds
            
    return time, max(LB1, LB2)

n=2000
gen_data(n)
loc = 0
df = pd.DataFrame(columns=('n','s', 'truncation', 'time', 'greedy', 'time', 'local search', 'time'))

for k in range(10, 80,10): # set the values of k
    print("This is case ", loc+1)
    ttime, tfval = truncation(n, k)
    gtime, gx, gfval = greedy(n, k)
    ltime, lfval, bestz = localsearch(n, k)
    df.loc[loc] = np.array([n, k, tfval, ttime, gfval, gtime, lfval, ltime])
    loc = loc+1  

