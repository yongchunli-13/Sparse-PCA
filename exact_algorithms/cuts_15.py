
import datetime
import os
import numpy as np
import pandas as pd
import math
from mosek.fusion import *

## Import data
def gen_data(n, data_name):
    global S
    global E, Eh
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
    Eh = E/2
    
 
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


##--------------given z, obtain cut by solving problem (19) --------------
def benderscut19(z, n, k):
    with Model("SDPcut") as M:
        ladavar = M.variable('lambdav', Domain.greaterThan(0.0))
        mu1 = M.variable('mu1v', n, Domain.unbounded())
        mu2 = M.variable('mu2v', n, Domain.greaterThan(0.0))
        nu1 = M.variable('nu1v', n, Domain.unbounded())
        nu2 = M.variable('nu2v', n, Domain.greaterThan(0.0))
        betavar = M.variable('betavar', n, Domain.unbounded()) 
        W1 = M.variable('W1', [n,n], Domain.greaterThan(0.0))
        W2 = M.variable('W2', [n,n], Domain.greaterThan(0.0))
        Lambda = M.variable('Lambda', [n,n], Domain.unbounded())
        
        
        M.constraint(Expr.sub(Expr.add(Expr.mulElm(E, Var.repeat(ladavar, n*n).reshape([n,n])), W1), 
                              Expr.add(A, Expr.add(Lambda, Expr.add(W2, 
                              Expr.mulElm(Eh, Expr.add(Var.repeat(mu1,1,n),Expr.add(Var.repeat(mu2,1,n),
                              Expr.add(Var.repeat(nu1,1,n), Var.repeat(nu2,1,n))))))))),
                     Domain.inPSDCone(n))
        
        for i in range(n):
            for j in range(n):
                M.constraint(Expr.sub(W1.index(i,j), W1.index(j,i)), Domain.equalsTo(0.0))
                M.constraint(Expr.sub(W2.index(i,j), W2.index(j,i)), Domain.equalsTo(0.0))
                # M.constraint(Expr.sub(Lambda.index(i,j), Lambda.index(j,i)), Domain.equalsTo(0.0))
                
                M.constraint(Expr.add(betavar.index(i), Expr.add(W1.index(i,j), W2.index(i,j))), Domain.lessThan(0.0))
                
        for i in range(n):
            M.constraint(Expr.vstack(mu2.index(i), mu1.index(i), Lambda.pick([[i,j] for j in range(n)])), Domain.inQCone(n+2))
            M.constraint(Expr.vstack(nu2.index(i), nu1.index(i), betavar.index(i)), Domain.inQCone(3))
           
 
        M.objective("obj", ObjectiveSense.Minimize, Expr.add(ladavar, 
                                                             Expr.add(Expr.mul(Expr.dot(Expr.sub(mu2,mu1),z), 0.5),
                                                                      Expr.mul(Expr.dot(Expr.sub(nu2,nu1),z), 0.5*k)))) 
        M.solve()
        
        mu1sol = M.getVariable('mu1v').level()
        mu2sol = M.getVariable('mu2v').level()
        nu1sol = M.getVariable('nu1v').level()
        nu2sol = M.getVariable('nu2v').level()
        ladasol = M.getVariable('lambdav').level()
        
        mu = [0]*n
        for i in range(n):
            mu[i] = (mu2sol[i] - mu1sol[i])/2 + (nu2sol[i] - nu1sol[i])/2*k
    
    return ladasol[0], mu


