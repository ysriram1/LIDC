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

data_lidc = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1 , usecols = (range(11,76)))
targets_lidc = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1, usecols = (84,93,102,111)).astype(int)

data_iris = np.genfromtxt('./iris.txt', delimiter = ',',usecols = (range(4)))
targets_iris = np.genfromtxt('./iris.txt', delimiter = ',', usecols = 4, dtype=None)

def dataPreprocess(data, targetArray, multipleLabels = True):  
    minmax = lambda x: (x-x.min())/(x.max() - x.min())
    X = np.apply_along_axis(minmax, 0, data)
    ##For binary similarities
    if multipleLabels:
        singleTarget = np.array([np.argmax(np.bincount(i)) for i in targetArray]) #Getting the mode for each target case
    else:
        singleTarget = targetArray
    S = np.zeros(shape=[singleTarget.shape[0], singleTarget.shape[0]])
    D = np.zeros(shape=[singleTarget.shape[0], singleTarget.shape[0]])
    for i in range(S.shape[0]):
        for j in range(i+1, S.shape[0]):
            if singleTarget[i] == singleTarget[j]:
                S[i,j] = 1                
            else:
                D[i,j] = 1
    return X, S, D

X,S,D = dataPreprocess(data = data_iris, targetArray = targets_iris, multipleLabels=False)

np.savetxt('./data.csv',X,delimiter=',')
np.savetxt('./S.csv',S, delimiter=',')
np.savetxt('./D.csv',S, delimiter=',')


########################################################################################################################
def diagA_Xing(X,S,D):
    #X: the data matrix
    #S: the similarity Matrix
    #D: the dissimilarty Matrix    

    #pre-calculating d_ij_S and d_ij_D and d_ij
    d_ij_S = np.zeros(shape=[X.shape[0],X.shape[0],X.shape[1]])
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            if S[i,j] == 1:
                d_ij_S[i,j,:] = np.array(X[i]-X[j]) 
    
    d_ij_D = np.zeros(shape=[X.shape[0],X.shape[0],X.shape[1]])
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            if D[i,j] == 1:
                d_ij_D[i,j,:] = np.array(X[i]-X[j])
    
    d_ij_S_sq = np.square(d_ij_S) 
    d_ij_D_sq = np.square(d_ij_D) 
    
    #Creating the objective function and the constraints function here
    def constraint(A):
        square_A = d_ij_D_sq*A
        sqrts_A = np.zeros(shape=[square_A.shape[0],square_A.shape[1]])
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                sqrts_A[i,j] = np.sum(square_A[i,j,:])
        return np.sum(np.sqrt(sqrts_A))
    
    def objective(A):
        print('processed')
        return np.sum(d_ij_S_sq*A) - math.log(constraint(A))
    
#    def constraint(A):
#        global values
#        values = []
#        for i in range(D.shape[0]):
#            for j in range(D.shape[1]):
#               values.append(np.sqrt(d_ij_D[i,j,:].T*np.multiply(d_ij_D[i,j,:],A)))
#        return np.sum(values)
#    
#    def objective(A):
#        values = []
#        for i in range(S.shape[0]):
#            for j in range(S.shape[1]):
#                values.append(d_ij_S[i,j,:].T*np.multiply(d_ij_S[i,j,:],A))
#        print 'processing'
#        return np.sum(values) - math.log(constraint(A))
        
    def grad(A):
        global grads
        grads = []
        for l in range(d_ij_D.shape[2]):
            temp = np.zeros(shape=[1,1,X.shape[1]])
            temp[0,0,l] = 1
            sum_S = np.sum(np.square(d_ij_S*temp))
            sum_D = np.sum(np.square(d_ij_D*temp))
            gradVal = sum_S - (0.5/(constraint(A)**2)*sum_D) #the (0.5/(constraint(A)**2)*sum_D) almost has no impact, can be removed? A has no impact?
            grads.append(gradVal)
        return np.array(grads)
       
  
    A0 = np.random.rand(X.shape[1])
    
    #return approx_grad=True(objective, A0, maxiter=20)
    #return differential_evolution(objective, [(0,10) for i in range(len(A0))])
    return fmin_l_bfgs_b(objective, A0, fprime = grad)
    #return fmin_cobyla(objective, A0,[constraint],maxfun=10**10)
    
timeA = time.time()
result_diag = diagA_Xing(X, S, D) #taking too long
timeB = time.time()



###############################################################################################################
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


