# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize 

os.chdir('C:/Users/SYARLAG1/Desktop/LIDC')

data = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1 , usecols = (range(11,76)))
targets = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1, usecols = (84,93,102,111)).astype(int)

def dataPreprocess(data = data, targetArray = targets):
    minmax = lambda x: (x-x.min())/(x.max() - x.min())
    X = np.apply_along_axis(minmax, 0, data)
    ##For binary similarities
    global singleTarget
    singleTarget = np.array([np.argmax(np.bincount(i)) for i in targetArray]) #Getting the mode for each target case
    S = np.zeros(shape=[singleTarget.shape[0], singleTarget.shape[0]])
    D = np.zeros(shape=[singleTarget.shape[0], singleTarget.shape[0]])
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            if i == j:
                continue
            if singleTarget[i] == singleTarget[j]:
                S[i,j] = S[j, i] = 1                
            if abs(singleTarget[i] - singleTarget[j]) >= 3:
                D[i,j] = D[j,i] = 1
    return X, S, D

X,S,D = dataPreprocess(data = data, targetArray = targets)

def fullA_Xing(X,S,D):
    #X: the data matrix
    #S: the similarity Matrix
    #D: the dissimilarty Matrix    
    
    #Creating the objective function and the constraints function here
    def objective(A):
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i,j] == 0:
                    continue
                M = X[i] - X[j] 
                return np.dot(np.dot(M.T,A),M)
    
    def constraint(A):
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if D[i,j] == 0:
                    continue
                M = X[i]-X[j]
                return np.dot(np.dot(M.T, A), M)
    
    A0 = np.random.rand(X.shape[1], X.shape[1])
    
    return minimize(objective, A0, method='COBYLA', constraints={'type':'ineq', 'fun':constraint})

result = fullA_Xing(X, S, D)
