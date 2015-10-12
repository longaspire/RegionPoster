import os
import string
import numpy as np
import cPickle
import random
import stevenNN
import stevenFS
import math

def conventItemRow(itemrow, cursor, FingerPrintmatrix):
    signallist = itemrow[:len(itemrow)-1].split(' ')
    FingerPrintmatrix[cursor-1] = signallist

def conventResult(label, cursor, Resultmatrix):
    Resultmatrix[cursor-1] = label
       
#----------------------- Test ---------------------------------------------
fileHandle = open ('../../0dataset_fingerprints/fingerprint_20130918.txt')
fileList = fileHandle.readlines()
total = len(fileList)
totalline = total
dim = 18
cursor = 0
testFingerPrintmatrix = np.zeros((total,dim))
testResultmatrix = np.zeros((total,1))

for fileLine in fileList:
    cursor += 1
    itemlist = fileLine.split('\t')
    conventItemRow(itemlist[1], cursor, testFingerPrintmatrix)
    conventResult(itemlist[2], cursor, testResultmatrix)

print testFingerPrintmatrix , '\n'
print testResultmatrix , '\n'
    
testResultmatrixList = [testResultmatrix[i,0] for i in range(testResultmatrix.shape[0])]
testFingerPrintmatrixList = [[testFingerPrintmatrix[i,j] for j in range(len(testFingerPrintmatrix[i]))] for i in range(testFingerPrintmatrix.shape[0])]

#----------------------- Training ---------------------------------------------
fileHandle = open ('../../0dataset_fingerprints/fingerprint_Thu_Nov_28_213537_CST_2013.txt')
fileList = fileHandle.readlines()
total = len(fileList)
totalline = total
dim = 18
cursor = 0
traininFingerPrintmatrix = np.zeros((total,dim))
traininResultmatrix = np.zeros((total,1))

for fileLine in fileList:
    cursor += 1
    itemlist = fileLine.split('\t')
    conventItemRow(itemlist[1], cursor, traininFingerPrintmatrix)
    conventResult(itemlist[2], cursor, traininResultmatrix)

print traininFingerPrintmatrix , '\n'
print traininResultmatrix , '\n'
    
traininResultmatrixList = [traininResultmatrix[i,0] for i in range(traininResultmatrix.shape[0])]
traininFingerPrintmatrixList = [[traininFingerPrintmatrix[i,j] for j in range(len(traininFingerPrintmatrix[i]))] for i in range(traininFingerPrintmatrix.shape[0])]

FSmodel = stevenFS.FSmodel('ISOMAP', traininFingerPrintmatrixList, traininResultmatrixList, testFingerPrintmatrixList, _n_feature=2, _alpha = 0.6)
fractraininFingerPrintmatrix ,fractestFingerPrintmatrix, select = FSmodel.transform()
print fractraininFingerPrintmatrix
print fractestFingerPrintmatrix
print select

fractraininFingerPrintmatrixList = [[fractraininFingerPrintmatrix[i,j] for j in range(len(fractraininFingerPrintmatrix[i]))] for i in range(fractraininFingerPrintmatrix.shape[0])]
fractestFingerPrintmatrixList = [[fractestFingerPrintmatrix[i,j] for j in range(len(fractestFingerPrintmatrix[i]))] for i in range(fractestFingerPrintmatrix.shape[0])]
label_count = max(set(testResultmatrixList))


#------------------------------------------Validation--------------------------------------------------

options = [0] * 3  

options[0]  = 0  #  DecisionTree
options[1]  = 1  #  scikit-learnSVM(Linear)
options[2]  = 0  #  scikit-learn NN 


# With DesicionTree
if(options[0] == 1):
    # initial 
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_leaf = 8)
    traininFingerPrintmatrixList = [[int(x) for x in w] for w in traininFingerPrintmatrixList]
    traininResultmatrixList = [int(x) for x in traininResultmatrixList]
    clf = clf.fit(traininFingerPrintmatrixList, traininResultmatrixList) 
    # parameters
    sum_DecisionTree = 0
    sum_eachNumber_DecisionTree = [0] * label_count
    
    frac_clf = tree.DecisionTreeClassifier(min_samples_leaf = 8)
    fractraininFingerPrintmatrixList = [[int(x) for x in w] for w in fractraininFingerPrintmatrixList]
    traininResultmatrixList = [int(x) for x in traininResultmatrixList]
    frac_clf = frac_clf.fit(fractraininFingerPrintmatrixList, traininResultmatrixList) 
    # parameters
    frac_sum_DecisionTree = 0
    frac_sum_eachNumber_DecisionTree = [0] * label_count
    

# With SVM linear
if(options[1] == 1):
    # initial 
    from sklearn import svm
    clf_scikitL = svm.SVC(C=0.0004, cache_size=200, class_weight=None, coef0=0.0, degree=3,
        gamma=0.0, kernel='linear', max_iter=-1, probability=False, random_state=None,
        shrinking=True, tol=0.001, verbose=False)
    traininFingerPrintmatrixList = [[int(x) for x in w] for w in traininFingerPrintmatrixList]
    traininResultmatrixList = [int(x) for x in traininResultmatrixList]
    clf_scikitL.fit(traininFingerPrintmatrixList, traininResultmatrixList) 
    # parameters
    sum_skSVM_L = 0
    sum_eachNumber_skSVM_L = [0] * label_count
    
    frac_clf_scikitL = svm.SVC(C=0.0004, cache_size=200, class_weight=None, coef0=0.0, degree=3,
            gamma=0.0, kernel='linear', max_iter=-1, probability=False, random_state=None,
            shrinking=True, tol=0.001, verbose=False)
    fractraininFingerPrintmatrixList = [[int(x) for x in w] for w in fractraininFingerPrintmatrixList]
    traininResultmatrixList = [int(x) for x in traininResultmatrixList]    
    frac_clf_scikitL.fit(fractraininFingerPrintmatrixList, traininResultmatrixList) 
    # parameters
    frac_sum_skSVM_L = 0
    frac_sum_eachNumber_skSVM_L = [0] * label_count
    
    
# With scikit-learn NN
if(options[2] == 1):
    # initial 
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    traininFingerPrintmatrixList = [[int(x) for x in w] for w in traininFingerPrintmatrixList]
    traininResultmatrixList = [int(x) for x in traininResultmatrixList]
    skNN = NearestCentroid()
    skNN.fit(traininFingerPrintmatrixList, traininResultmatrixList)
    NearestCentroid(metric='minkowski', shrink_threshold=None)
    # parameters
    sum_skNN = 0
    sum_eachNumber_skNN = [0] * label_count
    
    fractraininFingerPrintmatrixList = [[int(x) for x in w] for w in fractraininFingerPrintmatrixList]
    traininResultmatrixList = [int(x) for x in traininResultmatrixList]    
    frac_skNN = NearestCentroid()
    frac_skNN.fit(fractraininFingerPrintmatrixList, traininResultmatrixList)
    NearestCentroid(metric='minkowski', shrink_threshold=None)
    # parameters
    frac_sum_skNN = 0
    frac_sum_eachNumber_skNN = [0] * label_count
        

FingerprintCountList = [0] * label_count
SumfracFingerprintCountList = [[0 for x in range(2)] for y in range(int(label_count))]

for k in range(len(testFingerPrintmatrixList)):
    #print testFingerPrintmatrixList[k]
    t = FingerprintCountList[int(testResultmatrixList[k]) - 1]
    FingerprintCountList[int(testResultmatrixList[k]) - 1] = t + 1
    
    if(options[0] == 1):
        predict_DecisionTree = clf.predict(testFingerPrintmatrixList[k])
        if(predict_DecisionTree == testResultmatrixList[k]):
            sum_DecisionTree += 1
            temp = sum_eachNumber_DecisionTree[predict_DecisionTree[0] - 1]
            sum_eachNumber_DecisionTree[predict_DecisionTree[0] - 1] = temp + 1
            
    if(options[1] == 1):
        predict_skSVM_L = clf_scikitL.predict(testFingerPrintmatrixList[k])
        if(predict_skSVM_L == testResultmatrixList[k]):
            sum_skSVM_L += 1
            temp = sum_eachNumber_skSVM_L[predict_skSVM_L[0] - 1]
            sum_eachNumber_skSVM_L[predict_skSVM_L[0] - 1] = temp + 1     
    
    if(options[2] == 1):
        predict_skNN = skNN.predict(testFingerPrintmatrixList[k])
        if(predict_skNN == testResultmatrixList[k]):
            sum_skNN += 1
            temp = sum_eachNumber_skNN[predict_skNN[0] - 1]
            sum_eachNumber_skNN[predict_skNN[0] - 1] = temp + 1
            
for k in range(len(fractestFingerPrintmatrixList)):
    sx = SumfracFingerprintCountList[int(testResultmatrixList[k]) - 1][0] + fractestFingerPrintmatrixList[k][0]
    sy = SumfracFingerprintCountList[int(testResultmatrixList[k]) - 1][1] + fractestFingerPrintmatrixList[k][1]
    SumfracFingerprintCountList[int(testResultmatrixList[k]) - 1][0] = sx
    SumfracFingerprintCountList[int(testResultmatrixList[k]) - 1][1] = sy
    if(options[0] == 1):
        frac_predict_DecisionTree = frac_clf.predict(fractestFingerPrintmatrixList[k])
        if(frac_predict_DecisionTree == testResultmatrixList[k]):
            frac_sum_DecisionTree += 1
            temp = frac_sum_eachNumber_DecisionTree[frac_predict_DecisionTree[0] - 1]
            frac_sum_eachNumber_DecisionTree[frac_predict_DecisionTree[0] - 1] = temp + 1
            
    if(options[1] == 1):
        frac_predict_skSVM_L = frac_clf_scikitL.predict(fractestFingerPrintmatrixList[k])
        if(frac_predict_skSVM_L == testResultmatrixList[k]):
            frac_sum_skSVM_L += 1
            temp = frac_sum_eachNumber_skSVM_L[frac_predict_skSVM_L[0] - 1]
            frac_sum_eachNumber_skSVM_L[frac_predict_skSVM_L[0] - 1] = temp + 1     
    
    if(options[2] == 1):
        frac_predict_skNN = frac_skNN.predict(fractestFingerPrintmatrixList[k])
        if(frac_predict_skNN == testResultmatrixList[k]):
            frac_sum_skNN += 1
            temp = frac_sum_eachNumber_skNN[frac_predict_skNN[0] - 1]
            frac_sum_eachNumber_skNN[frac_predict_skNN[0] - 1] = temp + 1

           

if(options[0] == 1):
    print "DecisionTree:",sum_DecisionTree, len(testFingerPrintmatrixList), float((float)(sum_DecisionTree)/len(testFingerPrintmatrixList))
    print "DecisionTree:",frac_sum_DecisionTree, len(fractestFingerPrintmatrixList), float((float)(frac_sum_DecisionTree)/len(fractestFingerPrintmatrixList))
    #print sum_eachNumber_DecisionTree
    #print FingerprintCountList
    
if(options[1] == 1):
    print "skSVM with linear:", sum_skSVM_L, len(testFingerPrintmatrixList), float((float)(sum_skSVM_L)/len(testFingerPrintmatrixList))
    print "skSVM with linear:", frac_sum_skSVM_L, len(fractestFingerPrintmatrixList), float((float)(frac_sum_skSVM_L)/len(fractestFingerPrintmatrixList))
    #print sum_eachNumber_skSVM_L
    #print FingerprintCountList
    
if(options[2] == 1):
    print "skNN:", sum_skNN, len(testFingerPrintmatrixList), float((float)(sum_skNN)/len(testFingerPrintmatrixList))
    print "skNN:", frac_sum_skNN, len(fractestFingerPrintmatrixList), float((float)(frac_sum_skNN)/len(fractestFingerPrintmatrixList))
    #print sum_eachNumber_skNN
    #print FingerprintCountList


print "---------DONE--------"

'''
#print SumfracFingerprintCountList
mapPoint = []
for i in range(len(SumfracFingerprintCountList)):
    sx = (float)(SumfracFingerprintCountList[i][0]) / FingerprintCountList[i]
    sy = (float)(SumfracFingerprintCountList[i][1]) / FingerprintCountList[i]
    mapPoint.append([sx, sy])
#print mapPoint

import matplotlib.pyplot as plt

# create a mesh to plot in
x_min, x_max = fractraininFingerPrintmatrix[:, 0].min() - 1, fractraininFingerPrintmatrix[:, 0].max() + 1
y_min, y_max = fractraininFingerPrintmatrix[:, 1].min() - 1, fractraininFingerPrintmatrix[:, 1].max() + 1
h = 2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = frac_clf_scikitL.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('off')

# Plot also the training points
style = ['wo','ws','rp','y*','ms','k+','kD','wd','b+','k*','rH','cs','mp','ys','ko']
from random import choice
import string

def GenPassword(length=6,chars=string.digits):
    return ''.join([choice(chars) for i in range(length)])

style = []
for i in range(len(traininResultmatrixList)):
    flag = 1
    while(flag):
        if(('#'+GenPassword(6) in style) == False):
            style.append(('#'+GenPassword(6)))
            flag = 0
            
for index in range(len(traininResultmatrixList)):
    temp = int(traininResultmatrixList[index])
    plt.plot(fractraininFingerPrintmatrix[index,0], fractraininFingerPrintmatrix[index,1], color=style[temp-1], marker='s')

for index in range(len(mapPoint)):
    plt.annotate((index+1), mapPoint[index], size='14', backgroundcolor='white')
plt.show()
'''