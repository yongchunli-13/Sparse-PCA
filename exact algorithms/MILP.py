from gurobipy import *
import math
import datetime
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from uci_datasets import Dataset

## Import data
def gen_data(n, data_name):
    global S
    global E
    global V
    global A
    global T
    global d
    global A

    data = pd.read_table(os.path.dirname(os.getcwd())+'/datasets/'+data_name+'_txt',
          encoding = 'utf-8',sep=',')
    temp = data.drop(['Unnamed: 0'], axis=1)
    A = np.matrix(np.array(temp))

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




##-------------- continuous relaxation (13) --------------

## At each iteration, given the solution mu in (13), optimal solution z admits the closed form 
## then obtain the subgradient of mu
def subgrad(mu, k):
    nx = len(mu)  
    val = 0.0  
    for i in range(nx):
        if T[i] > 0:
            val = val + (1-mu[i]/T[i])*S[i]  
        else:
            val = val 
            
    # compute the largest eigenvalue        
    a, b = np.linalg.eigh(val)
    a = max(a)

    z = [0]*nx
    temp = mu
    index = np.argsort(-np.array(temp))
    for i in range(k):
        z[index[i]] = 1
  
    ub = a + sum(mu[i]*z[i] for i in range(nx)) # upper bound
    
    subg = [0.0]*nx
    for i in range(nx):
        if T[i] > 0:
            subg[i] = z[i] - ((np.matrix(b)*V[:,i])[0,0])**2/T[i] 
        
    return  ub, a, subg


## compute the continuous relaxation problem (13)        
def spca_rel_thirteen(n, k):
    ltime, LB, bestz = localsearch(n, k)
    zsol = bestz
    mu = [0]*n
    for i in range(n):
        if zsol[i] < 0.5:
            mu[i] = T[i]  
                
    val = 0.0  
    for i in range(n):
        if T[i] > 0:
            val = val + (1-mu[i]/T[i])*S[i]
        else:
            val = val
                        
    # compute the largest eigenvalue        
    a, b = np.linalg.eigh(val)
    a = max(a)

    z = [0]*n
    temp = mu
    index = np.argsort(-np.array(temp))
    for i in range(k):
        z[index[i]] = 1
    bestub = a + sum(mu[i]*z[i] for i in range(n)) # initial upper bound
    
    itert = 0 # number of iterations
    
    while(itert <= 30000): 
        itert = itert + 1
        
        gamma_t = 1/math.sqrt(8*itert)
        ub, a, new_mu = subgrad(mu, k)
        
        musol = np.array(mu) - gamma_t*np.array(new_mu)
        musol[musol < 0] = 0.0
        for i in range(n):
            if musol[i] > T[i]:
                musol[i] = T[i]
        mu = musol

        bestub = min(ub, bestub)
        # if itert%1000 ==0:  
        #     print('the current upper bound is', bestub)

    return  bestz, LB, bestub



## Solve the MILP (22) to obtain the optimal value for SPCA
def milp22(n, data_name, k):
    #### Set model ####
    start = datetime.datetime.now()
    
    ## Input a lower bound LB and an upper bound UB
    bestz, LB, UB = spca_rel_thirteen(n, k)
    print(LB, UB)
    
    epsilon = 1e+4
    nm = max(int(math.log((UB-LB)*epsilon))+1, 1) 
    print(LB, UB, nm)

    m = Model("milpspca")

    #### Creat coefficients ####
    coeff = [0]*nm
    for i in range(nm):
        coeff[i] = (UB-LB)*(2**(-i))
         
    #### Creat variables ####
    lambdavar = m.addMVar(shape = 1, lb=0.0, ub=UB, name="lambda")
    x = m.addMVar(shape=d, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    R = m.addMVar(shape=d, lb=-(UB-LB)*2**(-nm), ub=(UB-LB)*2**(-nm), vtype=GRB.CONTINUOUS, name="r")
    B = m.addMVar(shape=n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
    
    z = m.addMVar(n, vtype=GRB.BINARY, name="z")

    u1 = m.addMVar((d,n), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u1")
    u2 = m.addMVar((d,n), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u2")
    
    w = m.addMVar((d,d), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name = "w") 
    y = m.addMVar(d, vtype=GRB.BINARY, name="y")
    
    mu1 = m.addMVar((d,nm), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu1")
    mu2 = m.addMVar((d,nm), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu2")
    alpha = m.addMVar(nm, vtype=GRB.BINARY, name="alpha")
    
    #### Set objective ####
    m.setObjective(1*lambdavar, GRB.MAXIMIZE)
    
    
    #### Initialize solution #### 
    for i in range(n):
        z[i].start = bestz[i]

    #### Define constraints ####    
    m.addConstr(z.sum() <= k) # size constraint
    
    ## eigenvalue ##
    for i in range(n):
        m.addConstr(B[i] == sum(V[j,i]*u1[j,i] for j in range(d)))
          
    m.addConstr(V @ B - UB*x + sum(coeff[i] * mu1[:,i] for i in range(nm)) == R) # eigenvalues

    ## lambda approx ###
    m.addConstr(alpha @ np.array(coeff) + lambdavar == UB)
    # m.addConstr(sum(coeff[i]*alpha[i] for i in range(nm)) + lambdavar == UB)
    
    ## inf norm ##
    m.addConstr(sum(w[:,i] for i in range(d)) == x)
    
    for i in range(d):
        m.addConstr(w[i,i] == y[i])
        for j in range(d):
            m.addConstr(w[j,i] <= y[i])
            m.addConstr(w[j,i] >= -y[i])
    m.addConstr(y.sum() == 1)
            
    ## left-hand binary ##  
    #eone = np.array([1.0]*d)    
    for i in range(n):
        m.addConstr(u1[:,i] + u2[:,i] == x)
        for j in range(d):
            m.addConstr(u1[j,i] <= z[i])
            m.addConstr(u1[j,i] >= -z[i])
        
            m.addConstr(u2[j,i] <= 1 - z[i])        
            m.addConstr(u2[j,i] >= -1 + z[i])

    ## right-hand binary ##
    for i in range(nm):
        m.addConstr(mu1[:,i] + mu2[:,i] == x)
        for j in range(d):
            m.addConstr(mu1[j,i] <= alpha[i])
            m.addConstr(mu1[j,i] >= -alpha[i])
            
            m.addConstr(mu2[j,i] <= 1 - alpha[i])
            m.addConstr(mu2[j,i] >= -1 + alpha[i])

    m.Params.MIPGap= 1e-4
    m.params.OutputFlag = 1    
    m.params.timelimit = 3600 
    m.optimize()
    
    end = datetime.datetime.now()
    time = (end-start).seconds
    return m.objval, m.ObjBound, time
            

