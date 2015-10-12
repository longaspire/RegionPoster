from __future__ import division
import numpy as np
import math


class FSmodel():
    
    def __init__(self, _typeFS, _trainingset, _trainingresultset, _testset, _n_feature=12, _step=1, _alpha=1.2):
        self.typeFS = _typeFS
        self.trainingset = _trainingset
        self.trainingresultset = _trainingresultset
        self.testset = _testset
        self.n_feature = _n_feature
        self.totalfeature = len(_trainingset[0])
        self.percentile = ((float)(self.n_feature * 100))/(self.totalfeature)  
        self.alpha = _alpha
        self.step = _step
        self.transTrainset = np.zeros((np.shape(self.trainingset)[0],self.n_feature))
        self.transTestset = np.zeros((np.shape(self.testset)[0],self.n_feature))
        
        
    def transform(self):
        
        if(self.typeFS == 'SelectPercentile'):
            from sklearn.feature_selection import SelectPercentile, f_classif
            #print self.percentile
            selector = SelectPercentile(f_classif, percentile=self.percentile)
            selector.fit(self.trainingset, self.trainingresultset)
            self.transTrainset = selector.transform(self.trainingset)
            self.transTestset = selector.transform(self.testset)
            #print selector.get_support(True)
            return self.transTrainset, self.transTestset, selector.get_support(True)
        
        elif(self.typeFS == 'LinearModel'):
            from sklearn import linear_model
            clf = linear_model.Lasso(alpha=self.alpha)
            clf.fit(self.trainingset, self.trainingresultset)
            #print np.nonzero(clf.coef_)
            selected = np.array(np.nonzero(clf.coef_)).tolist()[0]
            self.transTrainset = np.zeros((np.shape(self.trainingset)[0],len(selected)))
            self.transTestset = np.zeros((np.shape(self.testset)[0],len(selected)))
            matrix = np.matrix(self.trainingset)
            self.transTrainset = np.array([[matrix[j,i] for i in selected] for j in range(np.shape(matrix)[0])])          
            matrix = np.matrix(self.testset)
            self.transTestset = np.array([[matrix[j,i] for i in selected] for j in range(np.shape(matrix)[0])])  
            return self.transTrainset, self.transTestset, selected
        
        elif(self.typeFS == 'RBF'):
            from sklearn.feature_selection import RFE
            from sklearn.svm import SVR
            estimator = SVR(kernel="linear")
            selector = RFE(estimator, self.n_feature, step=self.step)
            selector = selector.fit(self.trainingset, self.trainingresultset)
            selected = []
            for i in range(len(selector.support_)):
                if(selector.support_[i] == True):
                    selected.append(i)
            self.transTrainset = np.zeros((np.shape(self.trainingset)[0],len(selected)))
            self.transTestset = np.zeros((np.shape(self.testset)[0],len(selected)))
            matrix = np.matrix(self.trainingset)
            self.transTrainset = np.array([[matrix[j,i] for i in selected] for j in range(np.shape(matrix)[0])])          
            matrix = np.matrix(self.testset)
            self.transTestset = np.array([[matrix[j,i] for i in selected] for j in range(np.shape(matrix)[0])])  
            return self.transTrainset, self.transTestset, selected
        
        elif(self.typeFS == 'TreeBased'):
            from sklearn.ensemble import ExtraTreesClassifier
            clf = ExtraTreesClassifier()
            clf.fit(self.trainingset, self.trainingresultset)
            thes = sorted(clf.feature_importances_)[(self.totalfeature - self.n_feature)]
            selected = []
            for i in range(len(clf.feature_importances_)):
                if(clf.feature_importances_[i] >= thes):
                    selected.append(i)
            self.transTrainset = clf.transform(self.trainingset, threshold=thes)
            self.transTestset = clf.transform(self.testset, threshold=thes)
            #print np.shape(self.transTrainset), np.shape(self.transTestset)
            return self.transTrainset, self.transTestset, selected
        
        elif(self.typeFS == 'PCA'):
            from sklearn.decomposition import PCA
            pca_training = PCA(copy=True, n_components=self.n_feature, whiten=False)
            pca_training.fit(self.trainingset)
            self.transTrainset = pca_training.transform(self.trainingset)
            self.transTestset = pca_training.transform(self.testset)
            return self.transTrainset, self.transTestset, 0
        
        elif(self.typeFS == 'KernelPCA'):
            from sklearn.decomposition import KernelPCA
            kpca_training = KernelPCA(kernel='linear', n_components=self.n_feature)
            kpca_training.fit(self.trainingset,  self.trainingresultset)
            self.transTrainset = kpca_training.transform(self.trainingset)
            self.transTestset = kpca_training.transform(self.testset)
            return self.transTrainset, self.transTestset, 0
        
        elif(self.typeFS == 'ICA'):
            from sklearn.decomposition import FastICA
            fica = FastICA(n_components=self.n_feature, algorithm='deflation', max_iter=2000, whiten=True)
            fica.fit(self.trainingset)
            self.transTrainset = fica.transform(self.trainingset)
            self.transTestset = fica.transform(self.testset)
            return self.transTrainset, self.transTestset, 0
        
        elif(self.typeFS == 'ISOMAP'):
            from sklearn.manifold import Isomap
            isomap = Isomap(n_neighbors=5, n_components=self.n_feature)
            isomap.fit(self.trainingset,  self.trainingresultset)
            self.transTrainset = isomap.transform(np.matrix(self.trainingset))
            self.transTestset = isomap.transform(np.matrix(self.testset))
            return self.transTrainset, self.transTestset, 0
        
        else:
            print '----------wrong parameters!!!!'
            return self.transTrainset, self.transTestset, 0
