#! /usr/bin/env python
#coding=utf-8

import sys
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
        
def LoadFile(path):
    fileHandle = open (path)
    fileList = fileHandle.readlines()
    total = len(fileList)
    totalline = total
    dim = 18
    cursor = 0
    FingerPrintmatrix = np.zeros((total,dim))
    Resultmatrix = np.zeros((total,1))
    
    for fileLine in fileList:
        cursor += 1
        itemlist = fileLine.split('\t')
        conventItemRow(itemlist[1], cursor, FingerPrintmatrix)
        conventResult(itemlist[2], cursor, Resultmatrix)
        
    #print FingerPrintmatrix , '\n'
    #print Resultmatrix , '\n'
    
    ResultmatrixList = [Resultmatrix[i,0] for i in range(Resultmatrix.shape[0])]
    FingerPrintmatrixList = [[FingerPrintmatrix[i,j] for j in range(len(FingerPrintmatrix[i]))] for i in range(FingerPrintmatrix.shape[0])]
    
    return FingerPrintmatrixList, ResultmatrixList
    
def TrainTestSpilt(FingerPrintmatrixList, ResultmatrixList):
    
    from sklearn.cross_validation import train_test_split
    
    return train_test_split(FingerPrintmatrixList, ResultmatrixList, test_size=0.5, random_state=42)

def Testing(traininFingerPrintmatrixList, testFingerPrintmatrixList, traininResultmatrixList, testResultmatrixList, fractraininFingerPrintmatrix ,fractestFingerPrintmatrix, select):
    
    fractraininFingerPrintmatrixList = [[fractraininFingerPrintmatrix[i,j] for j in range(len(fractraininFingerPrintmatrix[i]))] for i in range(fractraininFingerPrintmatrix.shape[0])]
    fractestFingerPrintmatrixList = [[fractestFingerPrintmatrix[i,j] for j in range(len(fractestFingerPrintmatrix[i]))] for i in range(fractestFingerPrintmatrix.shape[0])]
    label_count = max(set(testResultmatrixList))
    
    from sklearn import svm
        
    frac_clf_scikitL = svm.SVC(C=0.0004, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                gamma=0.0, kernel='linear', max_iter=-1, probability=False, random_state=None,
                shrinking=True, tol=0.001, verbose=False)
    fractraininFingerPrintmatrixList = [[int(x) for x in w] for w in fractraininFingerPrintmatrixList]
    traininResultmatrixList = [int(x) for x in traininResultmatrixList]    
    frac_clf_scikitL.fit(fractraininFingerPrintmatrixList, traininResultmatrixList) 
    # parameters
    frac_sum_skSVM_L = 0
    
    FingerprintCountList = [0] * label_count
    
    for k in range(len(fractestFingerPrintmatrixList)):
        
        frac_predict_skSVM_L = frac_clf_scikitL.predict(fractestFingerPrintmatrixList[k])
        if(frac_predict_skSVM_L == testResultmatrixList[k]):
            frac_sum_skSVM_L += 1
    
    #print "skSVM with linear:", frac_sum_skSVM_L, len(fractestFingerPrintmatrixList), float((float)(frac_sum_skSVM_L)/len(fractestFingerPrintmatrixList))
    return float((float)(frac_sum_skSVM_L)/len(fractestFingerPrintmatrixList))
  
def FeatureSelection(FingerPrintmatrixList, ResultmatrixList, fCount, iter=3):
    i = 0
    AccuracySum = 0
    
    while(i < iter):
        traininFingerPrintmatrixList, testFingerPrintmatrixList, traininResultmatrixList, testResultmatrixList = TrainTestSpilt(FingerPrintmatrixList, ResultmatrixList)
        #print len(traininFingerPrintmatrixList)
        #print len(testFingerPrintmatrixList)
        #print set(traininResultmatrixList)
        #print set(testResultmatrixList)
            
        FSmodel = stevenFS.FSmodel('TreeBased', traininFingerPrintmatrixList, traininResultmatrixList, testFingerPrintmatrixList, _n_feature=fCount, _alpha = 0.6)
        fractraininFingerPrintmatrix ,fractestFingerPrintmatrix, select = FSmodel.transform()
        #print fractraininFingerPrintmatrix
        #print fractestFingerPrintmatrix
        #print select
        Accuracy = Testing(traininFingerPrintmatrixList, testFingerPrintmatrixList, traininResultmatrixList, testResultmatrixList, fractraininFingerPrintmatrix ,fractestFingerPrintmatrix, select)
        AccuracySum = AccuracySum + Accuracy
        preAccuracy = Accuracy
        if(Accuracy >= preAccuracy):
            OptiSelect = select
        i += 1
        #print Accuracy
    return AccuracySum/iter, OptiSelect
    
    

if __name__=="__main__":
    #print sys.argv[1]
    filename = sys.argv[1] #'E:/BaiduYun/Kanbox/9Projects/Room Determination/0dataset_fingerprints/fingerprint_20130918.txt'
    FingerPrintmatrixList, ResultmatrixList = LoadFile(filename)
    
    threshold_FeatureCount = 2
    n = 18
    AccuracyArray = []
    OptiSelectArray = []
    while(n >= threshold_FeatureCount):
        Accuracy, OptiSelect = FeatureSelection(FingerPrintmatrixList, ResultmatrixList, n, iter=3)
        #print n, Accuracy
        AccuracyArray.append(Accuracy)
        OptiSelectArray.append(OptiSelect)
        n -= 1
    print OptiSelectArray[AccuracyArray.index(max(AccuracyArray))]    
    
    
    
    
    
