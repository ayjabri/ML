#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 05:03:02 2020

@author: aymanjabri
"""


# %% Libraries_Data
import numpy as np
# import pandas as pd
# import scipy as sp
import sys
## temp libraries and functions
from sklearn.datasets import make_blobs,make_classification
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X,Y = make_blobs(100)
x,y = make_classification(n_samples=600,
                          n_features=3,
                          n_classes=5,
                          n_clusters_per_class=1,
                          n_redundant=0,
                          n_repeated=0,
                          n_informative=3)

# %% K_means_class

class KMeans(object):
    def __init__(self,n_clusters,n_initiates=10,n_itter=10,plot=False,
                 random_init=True):
        self.K = n_clusters
        self.M = n_initiates
        self.T = n_itter
        self.plot = plot
        self.rand = random_init
        return

    def randomC(self,X):
        # Initiates k random points that are normally distributed around data mean and variance
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        return np.random.normal(loc=mu,scale=std,size=(self.K,X.shape[1]))
    
    def euclidean_matrix(self,X,c):
        # Returns a euclidean distance matri
        m = X.shape[0]
        distance = np.empty((self.K,m))
        for i in range(self.K):
            distance[i] = np.sum((X-c[i])**2,axis=1)
        return distance


    def update(self,X,c):
        d = self.euclidean_matrix(X,c)
        y = np.argmin(d,axis=0)
        for i in range(self.K):
            c[i] = X[y==i].mean(axis=0)
        return c,y
    
    def kmeans(self,X,plot=False):
        c = self.randomC(X)
        c0 = np.array(c)
        centroids = [c0]
        if plot==True: self.plotem(X,c)
        for t in range(self.T):
            c,y = self.update(X,c)
            centroids.append(c.tolist())
            if plot==True: self.plotem(X,c,t,y)
        centroids = np.array(centroids)
        centroids = np.reshape(centroids,(int(centroids.size/2),-1))
        return c,y,centroids
        
    def plotem(X,c,i=0,y=[]):
        plt.scatter(X[:,0],X[:,1],c=y)
        plt.scatter(c[:,0],c[:,1],marker='x',s=300,label=i)
        plt.legend()
        plt.show()
        
    @staticmethod
    def cost(X,C,Y):
        cost = 0
        for i in range(C.shape[0]):
            cost += sum(self.euclidean(X[Y==i],C[i]))
        return cost

    def fit(self,X,plot=False):
        '''
        Runs K-means algorithm M number of times, and returns the instance with the least cost function
        ,which is defined by the sum of euclidean distances of each cluster points to their centroid\n
        M: number of random initiations\n
        T: number of itterations per M\n
        K: number of Clusters\n
        X: your data
        '''
        self.data = X
        clusters = {}
        cost_=[]
        for r in range(self.M):
            clusters[r] = self.kmeans(self.data,plot=plot)
            cost_.append(self.cost(self.data,clusters[r][0],clusters[r][1]))
        i_min = cost_.index(min(cost_))
        self.centroids = dict(clusters[i_min])
        return self.clusters

    
##########
k = KMeans(3)