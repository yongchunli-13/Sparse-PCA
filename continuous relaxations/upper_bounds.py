import datetime
import os
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
from uci_datasets import Dataset
import cuts
from gurobipy import *
from gurobipy import GRB

benderscut19 = cuts.benderscut19

## Import data
def gen_data(n):
    global S
    global E
    global V
    global A
    global T
    global d
    global A
    
    temp = pd.read_table(os.getcwd()+'/Matrix_Eisen_Data_1_txt',
                header=None,encoding = 'utf-8',sep=',')

    temp = np.array(temp)
    A = np.matrix(temp)
    
    # temp = pd.read_table(os.getcwd()+'/pitdata.csv',
    #                     header=None,encoding = 'utf-8',sep=',')
    
    # temp = np.array(temp)
    # A = np.matrix(temp)

    # data = Dataset("pumadyn32nm")
    # temp = data.x
    # temp = preprocessing.normalize(temp)/10 ## normalize data
    # temp = np.array(temp)
    # A = np.matrix(temp)
    # A = A.T*A
    
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
    b_k = b_k1 / b_k1_norm
    eigenval = 0
    
    while(abs(b_k1_norm-eigenval)/b_k1_norm >= 1e-6):
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
        val = val + (1-mu[i])*S[i]    
    a,b = power_iteration(val) # compute the largest eigenvalue

    z = [0]*nx
    temp = [mu[i]*T[i] for i in range(nx)]
    index = np.argsort(-np.array(temp))
    for i in range(k):
        z[index[i]] = 1
  
    ub = a + sum(mu[i]*z[i]*T[i] for i in range(nx)) # upper bound
    
    subg = [0.0]*nx
    for i in range(nx):
        subg[i] = T[i]*z[i] - ((np.matrix(b)*V[:,i])[0,0])**2 
    
    return  ub, a, subg


## compute the continuous relaxation problem (13)        
def spca_rel_thirteen(n, k):
    start = datetime.datetime.now()
    ltime, LB, bestz = localsearch(n, k)
    zsol = bestz
    mu = [0]*n
    for i in range(n):
        if zsol[i] < 0.5:
            mu[i] = 1  
                
    val = 0.0  
    for i in range(n):
        val = val + (1-mu[i])*S[i]
    a,b = power_iteration(val) # compute the largest eigenvalue

    z = [0]*n
    temp = [mu[i]*T[i] for i in range(n)]
    index = np.argsort(-np.array(temp))
    for i in range(k):
        z[index[i]] = 1
    bestub = a + sum(mu[i]*z[i]*T[i] for i in range(n)) # initial upper bound
    
    itert = 0 # number of iterations
    
    while(itert <= 20000): 
        itert = itert + 1
        
        gamma_t = 1/math.sqrt(8*itert)
        ub, a, new_mu = subgrad(mu, k)
        
        musol = np.array(mu) - gamma_t*np.array(new_mu)
        musol[musol < 0] = 0.0
        musol[musol > 1] = 1.0
        mu = musol

        bestub = min(ub, bestub)
        if itert%100 ==0:  
            print(ub, bestub)

    end = datetime.datetime.now()
    time = (end-start).seconds
    return  mu, bestub, time



##-------------- continuous relaxation (20) --------------
## compute the continuous relaxation problem (20) 
def opt_proj(beta, Lambda, mu1, mu2, nu1, nu2, n):
    #### create model ####
    m = Model()

    # add variables
    mu1var = m.addVars(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="mu1")
    mu2var = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name="mu2")

    nu1var = m.addVars(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="nu1")
    nu2var = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name="nu2")
    
    wvar1 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w")
    wvar2 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w")
    wvar3 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w")
    wvar4 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w")

    # add constraint
    for i in range(n):
        m.addConstr( nu1var[i]**2  <= nu2var[i]**2 - beta[i]**2)
        m.addConstr(sum(Lambda[i,j]**2 for j in range(n)) + mu1var[i]**2 <= mu2var[i]**2)
    
    m.addConstr(sum((mu1var[i]-mu1[i])**2 for i in range(n))  <= wvar1**2)
    m.addConstr(sum((mu2var[i]-mu2[i])**2 for i in range(n))  <= wvar2**2)
    m.addConstr(sum((nu1var[i]-nu1[i])**2 for i in range(n))  <= wvar3**2)
    m.addConstr(sum((nu2var[i]-nu2[i])**2 for i in range(n))  <= wvar4**2)

   
    m.setObjective(wvar1+wvar2+wvar3+wvar4, GRB.MINIMIZE)
    m.update()

    m.params.NonConvex=2
    m.update()
    m.params.OutputFlag= 0
    m.optimize()
    
    # W1sol = np.zeros([n,n])
    # Lambdasol = np.zeros([n,n])
    # W2sol = np.zeros([n,n])
    # for i in range(n):
    #     for j in range(n):
    #         Lambdasol[i, j] = Lambdavar[i,j].x
    #         W1sol[i,j] = W1var[i,j].x
    #         W2sol[i,j] = W2var[i,j].x
            
    nu1sol = [0]*n
    nu2sol = [0]*n
    mu1sol = [0]*n
    mu2sol = [0]*n
    for i in range(n):
        nu1sol[i] = nu1var[i].x
        nu2sol[i] = nu2var[i].x
        mu1sol[i] = mu1var[i].x
        mu2sol[i] = mu2var[i].x
    
    return  mu1sol, mu2sol, nu1sol, nu2sol
 
    
# def opt_proj(Lambda, mu1, mu2, W1, W2, nu1, nu2, n):
#     #### create model ####
#     m = Model()

#     # add variables
#     Lambdavar = m.addVars(n,n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Lambda")
#     mu1var = m.addVars(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="mu1")
#     mu2var = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name="mu2")
    
#     W1var = m.addVars(n,n, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="W1")
#     W2var = m.addVars(n,n, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="W2")
#     nu1var = m.addVars(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="nu1")
#     nu2var = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name="nu2")
    
#     betavar = m.addVars(n, vtype=GRB.CONTINUOUS, name="beta")
#     wvar1 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w")
#     wvar2 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w")
#     wvar3 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w")
    
#     wvar4 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w")
#     wvar5 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w")
#     wvar6 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w")
#     wvar7 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w")

#     # add constraint
#     for i in range(n):
#         for j in range(n):
#             m.addConstr(Lambdavar[i,j] == Lambdavar[j,i])
#             m.addConstr(W1var[i,j] == W1var[j,i])
#             m.addConstr(W2var[i,j] == W2var[j,i])
#             m.addConstr(betavar[i] + W1var[i,j] + W2var[i,j] <= 0)
        
#     for i in range(n):
#         m.addConstr(betavar[i]**2 + nu1var[i]**2 <= nu2var[i]**2)
#         m.addConstr(sum(Lambdavar[i,j]**2 for j in range(n)) + mu1var[i]**2 <= mu2var[i]**2)
    
#     m.addConstr(sum((Lambdavar[i,j] - Lambda[i,j])**2 for i in range(n) for j in range(n)) <= wvar1**2)
#     m.addConstr(sum((mu1var[i]-mu1[i])**2 for i in range(n))  <= wvar2**2)
#     m.addConstr(sum((mu2var[i]-mu2[i])**2 for i in range(n))  <= wvar3**2)
    
#     m.addConstr(sum((W1var[i,j] - W1[i,j])**2 for i in range(n) for j in range(n)) <= wvar4**2)
#     m.addConstr(sum((W2var[i,j] - W2[i,j])**2 for i in range(n) for j in range(n)) <= wvar5**2)
#     m.addConstr(sum((nu1var[i]-nu1[i])**2 for i in range(n))  <= wvar6**2)
#     m.addConstr(sum((nu2var[i]-nu2[i])**2 for i in range(n))  <= wvar7**2)
  
   
#     m.setObjective(wvar1+wvar2+wvar3+wvar4+wvar5+wvar6+wvar7, GRB.MINIMIZE)
#     m.update()

#     m.params.NonConvex=2
#     m.update()
#     m.params.OutputFlag= 0
#     m.optimize()
    
#     W1sol = np.zeros([n,n])
#     Lambdasol = np.zeros([n,n])
#     W2sol = np.zeros([n,n])
#     for i in range(n):
#         for j in range(n):
#             Lambdasol[i, j] = Lambdavar[i,j].x
#             W1sol[i,j] = W1var[i,j].x
#             W2sol[i,j] = W2var[i,j].x
            
#     nu1sol = [0]*n
#     nu2sol = [0]*n
#     mu1sol = [0]*n
#     mu2sol = [0]*n
#     for i in range(n):
#         nu1sol[i] = nu1var[i].x
#         nu2sol[i] = nu2var[i].x
#         mu1sol[i] = mu1var[i].x
#         mu2sol[i] = mu2var[i].x
    
#     return Lambdasol, mu1sol, mu2sol, W1sol, W2sol, nu1sol, nu2sol
 
      
def projection(Lambda, mu1, mu2, W1, W2, nu1, nu2, n):
    
    # ---- conic projection admits the closed form ----
    mu1[mu1 < 0.0] = 0.0
    mu2[mu2 < 0.0] = 0.0
    for i in range(n):
        tempy = 0.0
        tempy = sum(Lambda[i,j]**2 for j in range(n))
        tempy = np.sqrt(tempy + mu1[i]**2)
        
        if tempy > abs(mu2[i]):
            tempval = (abs(mu2[i]) + tempy)/(2*tempy) 
            for j in range(n):
                Lambda[i,j] = tempval*Lambda[i,j]          
            mu1[i] = tempval * mu1[i]
            mu2[i] = tempval * tempy
    Lambda = (Lambda + Lambda.T)/2
    for i in range(n):
        tempy = 0.0
        tempy = sum(Lambda[i,j]**2 for j in range(n))
        tempy = np.sqrt(tempy + mu1[i]**2)
        
        if tempy > abs(mu2[i]):
            mu2[i] = tempy
    # W1sol, W2sol, nu1sol, nu2sol = opt_proj(W1, W2, nu1, nu2, n)
    
    W1 = (W1 + W1.T)/2
    W2 = (W2 + W2.T)/2
    beta = 0.0
    beta = -np.amax(W1+W2, 1)
    
    nu1[nu1 < 0.0] = 0.0
    nu2[nu2 < 0.0] = 0.0
    for i in range(n):
        tempy = np.sqrt(beta[i]**2 + nu1[i]**2)
        if tempy > abs(nu2[i]):
            tempval = (abs(nu2[i]) + tempy)/(2*tempy)
            beta[i] = tempval * beta[i]
            nu1[i] = tempval * nu1[i]
            nu2[i] = tempval * tempy

    temp = np.repeat(beta, n).reshape(n,n)
    temp = temp + W1 + W2
    
    if np.max(temp) < 0.0:
        print("find projection")
    else:
        temp[temp < 0] = 0
        W1 = W1 - temp/2
        W2 = W2 - temp/2
        for i in range(n):
            for j in range(n):
                if W1[i,j] < 0:
                    W2[i,j] = W2[i,j] + W1[i,j]
                    W1[i,j] = 0.0
                if W2[i,j] < 0:
                    W1[i,j] = W1[i,j] + W2[i,j]
                    W2[i,j] = 0.0             
         
    return Lambda, mu1, mu2, W1, W2, nu1, nu2
 
    
def spca_rel_twenty(n, k): 
    start = datetime.datetime.now()
    ltime, LB, bestz = localsearch(n, k)
    
    Lambda, beta, mu1, mu2, nu1, nu2, W1, W2 = benderscut19(bestz, n, k)
    val = A + Lambda - W1 + W2 +1/2*np.diag(mu1+mu2+nu1+nu2)
    a, b = power_iteration(val)
    
    temp = k/2*(nu2 - nu1) + 1/2*(mu2 - mu1)
    
    z = [0]*n
    index = np.argsort(-np.array(temp))
    for i in range(k):
        z[index[i]] = 1
        
    bestub = a + sum(temp[i]*z[i] for i in range(n))
    
    itert = 0 # number of iterations
    
    while(itert <= 6000): 
        
        itert = itert + 1
        
        gamma_t = 1/math.sqrt(8*itert)
        
        subgrad = np.matrix(b).T * np.matrix(b)
        Lambda = Lambda - gamma_t * subgrad
        W1 = W1 + gamma_t * subgrad
        W2 = W2 - gamma_t * subgrad
        W1[W1 < 0] = 0.0
        W2[W2 < 0] = 0.0
        
        mu1 = mu1 - gamma_t*1/2*np.diag(subgrad) + gamma_t*1/2*np.array(z)
        mu2 = mu2 - gamma_t*1/2*np.diag(subgrad) - gamma_t*1/2*np.array(z)
        nu1 = nu1 - gamma_t*1/2*np.diag(subgrad) + gamma_t*k/2*np.array(z)
        nu2 = nu2 - gamma_t*1/2*np.diag(subgrad) - gamma_t*k/2*np.array(z)
        
        Lambda, mu1, mu2, W1, W2, nu1, nu2 = projection(Lambda, mu1, mu2, W1, W2, nu1, nu2, n)
        
        val = A + Lambda - W1 + W2 +1/2*np.diag(np.array(mu1)+np.array(mu2)+np.array(nu1)+np.array(nu2))
        a, b = power_iteration(val)
        
        temp = k/2*(np.array(nu2) - np.array(nu1)) + 1/2*(np.array(mu2) - np.array(mu1))
        
        z = [0]*n
        index = np.argsort(-np.array(temp))
        for i in range(k):
            z[index[i]] = 1
            
        ub = a + sum(temp[i]*z[i] for i in range(n))

        bestub = min(ub, bestub)
        if itert%10 ==0:  
            print(ub, bestub)
            
    end = datetime.datetime.now()
    time = (end-start).seconds
    
    return bestub, time

