#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:10:15 2020

@author: aymanjabri
"""

from __future__ import division
from scipy.stats import multivariate_normal
import numpy as np
import sys


## can make more functions if required

class pluginClassifier(object):
    def __init__(self,X_train,y_train):
        self.train = X_train
        self.target = y_train
        self.n_features = X_train.shape[1]
        self.n = y_train.shape
        self.classes = np.unique(y_train)
        self.stats = self.descriptive_stats()
        
    def descriptive_stats(self):
        self.stats = {}
        for c in self.classes:
            mu = list(self.train[self.target == c].mean(axis=0))
            cov = np.cov(self.train[self.target == c].T,ddof=1).tolist()
            p = (sum(self.target==c)/self.n).item()
            self.stats[c] = (mu,cov,p)
        return self.stats
    
    def __normalize(self,x):
        return x/np.sum(x,axis=0)
    
    def probabilities(self,X_test):
        probabilities = []
        for row in X_test:
            vector = []
            for c in self.classes:
                # Likelihood usig cov matrix gives slightly bettr results (98%) than variance matrix (96%)
                vector.append(multivariate_normal.pdf(row,mean=self.stats[c][0],cov=self.stats[c][1])* self.stats[c][2])
            probabilities.append(vector)
        return np.apply_along_axis(self.__normalize,1,probabilities)
    
    def predict(self,X_test):
        return np.argmax(self.probabilities(X_test),axis=1)


# def pluginClassifierII(X_train, y_train, X_test):
#     classes = np.unique(y_train)
#     p_class = np.bincount(y_train)/len(y_train)
#     probabilities = []
#     for row in X_train:
#         vector = []
#         for c in classes:
#             mu = X_train[y_train==c].mean(axis=0)
#             # var = X_train[y_train==c].var(axis=0,ddof=1)
#             cov = np.cov(X_train[y_train==c].T,ddof=1)
#             vector.append(multivariate_normal.pdf(row,
#                               mean=mu,cov=cov)*p_class[c])
#         probabilities.append(vector)
#     y_test = np.argmax(np.array(probabilities),axis=1)
#     return probabilities,y_test
    

if __name__=='__main__':
    X_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2]).astype(int)
    X_test = np.genfromtxt(sys.argv[3], delimiter=",")
    bc = pluginClassifier(X_train,y_train)
    y_test = bc.probabilities(X_test)
    # assuming final_outputs is returned from function
    # final_outputs = pluginClassifier(X_train, y_train, X_test) 
    # write output to file
    np.savetxt("probs_test.csv", y_test, delimiter=",") 

