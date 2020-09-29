#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:31:04 2020

@author: aymanjabri
"""

import numpy as np
from scipy.stats import norm,multivariate_normal
import matplotlib.pyplot as plt


class EM(object):
    def __init__(self,K,M,T,plot=False):
        self.K = K
        self.M = M
        self.T = T
        self.plot = plot

    def init(self):
        '''
        Randomly initiate k number of gaussian distributions
        mu is randomly chosen from the dataset
        variance is 
        '''
        # mu = np.random.normal(loc = self.x.mean(),scale = self.x.var(),size=self.K)
        mu = np.random.choice(self.x,size=self.K,replace=False)
        var = np.ones(self.K) * 1
        return mu,var
    
    def posterior(self,x,mu,var):    
        lik = norm.pdf(x.reshape(-1,1),loc=mu,scale=var)
        prior_class = lik.sum(axis=0)
        post = lik*prior_class/(lik@prior_class.reshape(-1,1))
        return post

    def cluster(self,x,mu,var):
        self.x = x
        post = self.posterior(x,mu,var)
        y = post.argmax(axis=1)
        return y
    
    def fit(self,x):
        self.x = x
        mu,var = self.init()
        for t in range(self.T):
            post = self.posterior(self.x, mu, var)
            mu = self.x@post/post.sum(axis=0)
            for k in range(self.K):
                var[k]= ((self.x-mu[k])**2)@post[:,k]/post[:,k].sum()
        self.y = self.cluster(self.x, mu, var)
        self.mu = mu
        self.var = var
        return
    
    def plotem(self):
        l = np.linspace(self.x.min(),self.x.max(),500)
        m = norm.pdf(l.reshape(-1,1),loc=self.mu,scale=self.var)
        plt.scatter(self.x,np.zeros_like(self.x),c=self.y)
        plt.scatter(self.mu,np.zeros_like(self.mu),marker='x',s=200,c='blue')
        plt.plot(l,m[:,0])
        plt.plot(l,m[:,1])
        return

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
     
# %matplotlib
# a = np.linspace(8,-15,200)
# b = np.linspace(0,-15,200)
# A,B = np.meshgrid(a,b)
# pos = np.zeros((200,200,2))
# pos[:,:,0]=A;pos[:,:,1]=B
# Z1 = multivariate_normal.pdf(pos,mean=mu[0],cov=Sigma[0])
# Z2 = multivariate_normal.pdf(pos,mean=mu[1],cov=Sigma[1])
# Z3 = multivariate_normal.pdf(pos,mean=mu[2],cov=Sigma[2])

# fig = plt.figure(num=1)
# ax1 = fig.add_subplot(111,projection='3d')
# #ax1.plot_wireframe(A,B,Z1)
# #ax1.plot_wireframe(A,B,Z2,color='r')
# #ax1.plot_wireframe(A,B,Z3,color='grey')
# ax1.scatter(X[:,0],X[:,1],c=y)
# ax1.contour(A,B,Z1,offset=0)
# ax1.contour(A,B,Z2,offset=0)
# ax1.contour(A,B,Z3,offset=0)