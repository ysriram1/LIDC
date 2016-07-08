# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
from scipy.optimize import minimize, fmin_cobyla, linprog, fmin_cg, fmin_l_bfgs_b, differential_evolution, fmin_cg, fsolve
import math
import time
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


os.chdir('C:/Users/SYARLAG1/Desktop/LIDC')

data_lidc = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1 , usecols = (range(11,76)))
targets_lidc = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1, usecols = (84,93,102,111)).astype(int)

#data_iris = np.genfromtxt('./iris.txt', delimiter = ',',usecols = (range(4)))
#targets_iris = np.genfromtxt('./iris.txt', delimiter = ',', usecols = 4, dtype=None)

########################################################################################################################
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

X,S,D = dataPreprocess(data = data_lidc, targetArray = targets_lidc, multipleLabels=True)

#X,S,D = dataPreprocess(data = data_iris, targetArray = targets_iris, multipleLabels=False)

#np.savetxt('./data.csv',X,delimiter=',')
#np.savetxt('./S.csv',S, delimiter=',')
#np.savetxt('./D.csv',S, delimiter=',')


########################################################################################################################
def createDistributionLabels(targetArray):
    distributionLabel = []
    for entry in targetArray:
        labelVal = {1:0,2:0,3:0,4:0,5:0}#Initialize all 5 labels as 0s
        for rating in labelVal.keys():
            for radRating in entry:
                if rating == radRating:
                    labelVal[rating] += 0.25
        distributionLabel.append(labelVal.values())
    return np.array(distributionLabel)
        
distLabels = createDistributionLabels(targets_lidc)

#######################################################################################################################
#Test-Train split for probabilistic labels
def splitTrainTest(data,Labels,train_percent,random_state, minmax=False):
    random.seed(random_state)
    indexList = range(len(data))
    random.shuffle(indexList)
    trainIndexList = indexList[:int(len(data)*train_percent)]
    testIndexList = indexList[int(len(data)*train_percent):] 
    train, trainLabels, test, testLabels = data[trainIndexList], Labels[trainIndexList], data[testIndexList], Labels[testIndexList]
    if minmax:
        fit = MinMaxScaler.fit(train)
        train = fit.transform(train)
        test = fit.transform(test)
    return train, trainLabels, test, testLabels

train, trainLabels, test, testLabels = splitTrainTest(X,distLabels,0.7,99)

np.savetxt('./train.csv',train,delimiter=',')
np.savetxt('./trainLabels.csv',trainLabels,delimiter=',')
np.savetxt('./test.csv',test,delimiter=',')
np.savetxt('./testLabels.csv',testLabels,delimiter=',')
########################################################################################################################
pred9Cos = np.genfromtxt('./prediction9Cos.csv', delimiter = ',')
meanPred = np.mean(pred9Cos, axis=0)
meanTest = np.mean(testLabels, axis=0)
cosOfMeanArrays = np.dot(meanPred,meanTest)/(np.linalg.norm(meanPred)*np.linalg.norm(meanTest))

fig, ax = plt.subplots()

ax.plot([1,2,3,4,5],meanPred, 'r', label='Predicted Dist')
ax.plot([1,2,3,4,5],meanTest, 'g', label='Actual Dist')
plt.xlabel('Malignancy Value')
plt.ylabel('Probability')
plt.title('Average Predicted Dist vs Acutal Dist')
ax.legend()

###########################################################################################################################
resultD = {}

for i, label in enumerate(testLabels):
    if tuple(label) in resultD.keys():
        resultD[tuple(label)].append(pred9Cos[i])
        continue
    else:
        resultD[tuple(label)] = []
        resultD[tuple(label)].append(pred9Cos[i])

for key in resultD.keys():
    mean = np.mean(np.array(resultD[key]),axis=0)
    
    fig, ax = plt.subplots()
    ax.plot([1,2,3,4,5],mean, 'r', label='Predicted Dist')
    ax.plot([1,2,3,4,5],list(key), 'g', label='Actual Dist')
    text = 'the average predicted vs actual dist for' + str(list(key)) + '(Pred Count: ' + str(len(resultD[key])) + ')'
    plt.title(text)
    ax.legend()
    


########################################################################################################################
def diagA_Xing(X,S,D, maxIter = 100):
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
        #square_A = d_ij_D_sq*A
        #sqrts_A = np.zeros(shape=[square_A.shape[0],square_A.shape[1]])
        #for i in range(D.shape[0]):
            #for j in range(D.shape[1]):
                #sqrts_A[i,j] = np.sum(square_A[i,j,:])
        #return np.sum(np.sqrt(sqrts_A))        
        return np.sum(np.sqrt(np.tensordot(d_ij_D_sq, A, ([2],[0]))))
    
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
            #temp = np.zeros(shape=[1,1,X.shape[1]])
            #temp[0,0,l] = 1
            temp = np.zeros(X.shape[1])
            temp[l] = 1
            sum_S = np.sum(np.tensordot(d_ij_S_sq,temp,([2],[0])))
            sum_D = np.sum(np.tensordot(d_ij_D_sq,temp,([2],[0])))
            gradVal = sum_S - (0.5/(constraint(A)**2)*sum_D) #the (0.5/(constraint(A)**2)*sum_D) almost has no impact, can be removed? A has no impact?
            grads.append(gradVal)
        return np.array(grads)
       
  
    A0 = np.random.rand(X.shape[1])
    
    
    return fsolve(objective, A0)
    #return approx_grad=True(objective, A0, maxiter=20)
    #return differential_evolution(objective, [(0,10) for i in range(len(A0))])
    #return fmin_l_bfgs_b(objective, A0, fprime=grad, maxiter = maxIter)
    #return fmin_l_bfgs_b(objective, A0, approx_grad=True, epsilon=0.01)
    #return fmin_cobyla(objective, A0,[constraint],maxfun=10**10)
    
timeA = time.time()
result_diag = diagA_Xing(X, S, D) #taking too long
timeB = time.time()
print 'time taken: ', timeB-timeA
result_diag


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


