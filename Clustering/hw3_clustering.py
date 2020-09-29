#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 08:06:13 2020

@author: aymanjabri
"""

# %% Libraries_Data
import numpy as np
import sys
from scipy.stats import multivariate_normal

## temp libraries and functions
from sklearn.datasets import make_blobs,make_classification
import matplotlib.pyplot as plt
X,Y = make_blobs(100)
x,y = make_classification(n_samples=600,
                          n_features=8,
                          n_classes=5,
                          n_clusters_per_class=1,
                          n_redundant=0,
                          n_repeated=0,
                          n_informative=3)
def make_random(mean,var,n=150):
    '''
    Generate a random data set from a multivariate normal distribution
    Equally weighted
    '''
    data =[]
    k = len(mean)
    for i in range(k):
        xi = np.random.normal(loc=mean[i],scale=var[i],size=int(n/k))
        data.append(xi.reshape(-1,1))
    d = np.vstack(data)
    return d

#%% K_means_Functions

def euclidean(X,c0):
        # Returns the euclidean distance between a vecotr and point c0
        return  np.sum((X-c0)**2,axis=1)

def euclidean_matrix(X,c):
        # Returns a euclidean distance matri
        l = c.shape[0]
        m = X.shape[0]
        d = np.empty((l,m))
        for i in range(l):
            d[i] = euclidean(X,c[i])
        return d
    
def randomC(X,k):
    # Initiates k random points that are normally distributed around data mean and variance
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    c = np.random.normal(loc=mu,scale=std,size=(k,X.shape[1]))
    return c

def update(X,c):
    d = euclidean_matrix(X,c)
    y = np.argmin(d,axis=0)
    for i in range(c.shape[0]):
        if X[y==i].size==0:
            c[i]=np.inf
        else:
            c[i] = X[y==i].mean(axis=0)
    return c,y


def k_oneRound(X,K,t,plot=False):
    c = randomC(X,K)
    centroids = []
    if plot==True: plotem(X,c)
    for t in range(t):
        c,y = update(X,c)
        centroids.append(c.tolist())
        if plot==True: plotem(X,c,t,y)
    centroids = np.array(centroids)
    # centroids = np.reshape(centroids,(int(centroids.size/2),-1))
    return c,y,centroids

def plotem(X,c,i=0,y=[]):
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.scatter(c[:,0],c[:,1],marker='x',s=300,label=i)
    plt.legend()
    plt.show()

def cost(X,C,Y):
    d = 0
    for i in range(C.shape[0]):
        d += sum(euclidean(X[Y==i],C[i]))
    return d

def KMeans(X,K=5,M=10,T=10,plot=False):
    '''
    Runs K-means algorithm M number of times, and returns the instance with the least cost function
    ,which is defined by the sum of euclidean distances of each cluster points to their centroid\n
    M: number of random initiations\n
    T: number of itterations per M\n
    K: number of Clusters\n
    X: your data
    '''
    clusters = {}
    cost_=[]
    for r in range(M):
        clusters[r] = k_oneRound(X,K,T,plot=plot)
        cost_.append(cost(X,clusters[r][0],clusters[r][1]))
    i_min = cost_.index(min(cost_))
    return clusters[i_min]

#%% EM_GMM

def emm(x,K,epochs=10):
    global phi,nk,pi,mu,Sigma
    d = x.shape[1]
    n = x.shape[0]
    pi = np.ones(K)/K
    mu = x[np.random.choice(len(x),size=K,replace=False)]
    Sigma = np.zeros((K,d,d))
    Sigma[:,:,:] = np.identity(d)
    posterior = np.zeros((n,K))
    for epoch in range(epochs):
        # The E step: calculate ϕi(k)
        for k in range(K):
            posterior[:,k] = pi[k] * multivariate_normal.pdf(x,mean=mu[k],cov=Sigma[k])
        phi = posterior/posterior.sum(axis=1).reshape(-1,1)
        
        # The M step
        nk = phi.sum(axis=0)
        pi = nk/n
        for k in range(K):
            mu[k] = sum(phi[:,k].reshape(-1,1)*x)/nk[k]
            s = np.zeros((d,d))
            for i in range(n):
                t = (x[i]-mu[k]).reshape(-1,1)
                s += phi[i][k]*(t@t.T)/nk[k]
                # s += phi[i][k]*(x[i]-mu[k])*(x[i]-mu[k]).T/nk[k]
            Sigma[k]=s
            filename = "Sigma-" + str(k+1) + "-" + str(epoch+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, np.diag(Sigma[k]), delimiter=",")
        filename = "pi-" + str(epoch+1) + ".csv" 
        np.savetxt(filename, pi, delimiter=",") 
        filename = "mu-" + str(epoch+1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
    return mu,Sigma

#%%
# 	filename = "pi-" + str(i+1) + ".csv" 
# 	np.savetxt(filename, pi, delimiter=",") 
# 	filename = "mu-" + str(i+1) + ".csv"
# 	np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
    
#   for j in range(k): #k is the number of clusters 
#     filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
#     np.savetxt(filename, sigma[j], delimiter=",")




if __name__=='__main__':
    data = np.genfromtxt(sys.argv[1], delimiter = ",")
    K = 5
    i = 10
    c,targets,centerslist = KMeans(X=data,K=K,M=10,T=i,plot=False)
    for m in range(i):
        filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
        np.savetxt(filename, centerslist[2][i], delimiter=",")
    
# #     # EMGMM(data)
    
