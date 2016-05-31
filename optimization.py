# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
from scipy.optimize import minimize, fmin_cobyla, linprog, fmin_cg, fmin_l_bfgs_b, differential_evolution, fmin_cg
import math
import time

os.chdir('C:/Users/SYARLAG1/Desktop/LIDC')

data = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1 , usecols = (range(11,76)))
targets = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1, usecols = (84,93,102,111)).astype(int)


def dataPreprocess(data, targetArray):  
    minmax = lambda x: (x-x.min())/(x.max() - x.min())
    X = np.apply_along_axis(minmax, 0, data)
    ##For binary similarities
    global singleTarget
    singleTarget = np.array([np.argmax(np.bincount(i)) for i in targetArray]) #Getting the mode for each target case
    S = np.zeros(shape=[singleTarget.shape[0], singleTarget.shape[0]])
    D = np.zeros(shape=[singleTarget.shape[0], singleTarget.shape[0]])
    for i in range(S.shape[0]):
        for j in range(i+1, S.shape[0]):
        #for j in range(S.shape[0]):
            #if i == j or S[j,i] == 1 or D[j,i] == 1:
                #continue
            if singleTarget[i] == singleTarget[j]:
                S[i,j] = 1                
            else:
                D[i,j] = 1
    return X, S, D

X,S,D = dataPreprocess(data = data, targetArray = targets)

########################################################################################################################
def diagA_Xing(X,S,D):
    #X: the data matrix
    #S: the similarity Matrix
    #D: the dissimilarty Matrix    
    
    #Creating the objective function and the constraints function here
    def constraint(A):
        global values
        values = []
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if D[i,j] == 0:
                    continue
                M = np.array(X[i]-X[j])
                Y = np.array([M[x]*A[x] for x in range(len(M))])
                values.append(np.sqrt(sum(x**2 for x in Y)))
        return np.sum(values)
    
    def objective(A):
        values = []
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i,j] == 0 and D[i,j] == 0:
                    continue
                if S[i,j] == 1:
                    M = np.array(X[i] - X[j])
                    Y = np.array([M[x]*A[x] for x in range(len(M))])
                    values.append(sum(x**2 for x in Y))
        print 'processing'
        return np.sum(values) - math.log(constraint(A))
    
    A0 = np.random.rand(X.shape[1])
    
    #return approx_grad=True(objective, A0, maxiter=20)
    #return differential_evolution(objective, [(0,10) for i in range(len(A0))])
    return fmin_l_bfgs_b(objective, A0, approx_grad=True)
    #return fmin_cobyla(objective, A0,[constraint],maxfun=10**10)
    
timeA = time.time()
result_diag = diagA_Xing(X, S, D) #taking too long
timeB = time.time()

def fullA_Xing(X,S,D):
    #X: the data matrix
    #S: the similarity Matrix
    #D: the dissimilarty Matrix    
    
    #Creating the objective function and the constraints function here
    def objective(A):
        values = []
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i,j] == 0:
                    continue
                M = np.matrix(X[i] - X[j])
                values.append(np.sum((M*A)*M.T))  
        return np.sum(np.array(values))

    def constraint(A):
        values = []
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if D[i,j] == 0:
                    continue
                M = np.matrix(X[i]-X[j])
                values.append(np.sqrt(np.sum((M*A)*M.T)))
        return np.sum(np.array(values))
                    
    A0 = np.random.rand(X.shape[1], X.shape[1])
    
    return fmin_cobyla(objective, A0, [constraint], rhoend=1e-7)

result = fullA_Xing(X, S, D) #not working



def fullA_Xing_NCG(X,S,D):
    #X: the data matrix
    #S: the similarity Matrix
    #D: the dissimilarty Matrix    
    
    #Creating the objective function and the constraints function here
    def constraint(A):
        values = []
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if D[i,j] == 0:
                    continue
                M = np.matrix(X[i]-X[j])
                values.append(np.sqrt(np.sum((M*A)*M.T)))
        return np.sum(np.array(values)) 
    
    
    def objective(A):
        values = []
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i,j] == 0:
                    continue
                M = np.matrix(X[i] - X[j])
                values.append(np.sum((M*A)*M.T))  
        return np.sum(np.array(values)) - math.log(constraint(A))

    A0 = np.random.rand(X.shape[1], X.shape[1])
    
    return fmin_cg(objective, A0, maxiter=20)
    
result_NCG_full = fullA_Xing_NCG(X, S, D) #shape mismatch like before


