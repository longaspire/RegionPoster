import numpy as np
import math

def Test():
    print "this is the test function!"


def ManhattanDistance(vector1, vector2):
    d = 0.0
    for i in range(len(vector1)): 
        d += abs(vector1[i]-vector2[i])
    return d

def EuclideanDistance(vector1, vector2): 
    d = 0.0
    for i in range(len(vector1)): 
        d += (vector1[i]-vector2[i])**2
    return math.sqrt(d) 

class KNN():
    def __init__(self, k, trainingset, trainingresultset):
        self.k = k
        self.radiomap = trainingset
        self.result = trainingresultset
    def predict(self, queryvector, func):
        distance = []
        if(func == "ManhattanDistance"):
            for i in range(len(self.radiomap)):
                #print self.radiomap[i]
                distance.append(ManhattanDistance(queryvector, self.radiomap[i]))
        if(func == "EuclideanDistance"):
            for i in range(len(self.radiomap)):
                #print self.radiomap[i]
                distance.append(EuclideanDistance(queryvector, self.radiomap[i]))
            
        sortedDistIndicies = np.argsort(distance)
        #print "minDistance:", distance[sortedDistIndicies[0]]
        countlistZero = [0] * 9
        countlistNonZero = [0] * 9
        for x in range(self.k):
            
            index = (int)(self.result[sortedDistIndicies[x]])
            if(distance[sortedDistIndicies[0]] == 0):
                temp = countlistZero[index-1]
                countlistZero[index-1] = temp + 1
            else:
                temp = countlistNonZero[index-1]
                countlistNonZero[index-1] = temp + 1
                    
        #print countlistNonZero, countlistZero
        #print max(countlistNonZero), max(countlistZero)
        if(max(countlistZero) > 0):
            return countlistZero.index(max(countlistZero)) + 1
        else:
            return countlistNonZero.index(max(countlistNonZero)) + 1