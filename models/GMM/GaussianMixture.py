#Followed http://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/

import numpy as np
import scipy.stats as stats

class GaussianMixture(object):
    def __init__(self, nc):
        super().__init__()
        self.nc = nc

    def get_likeihood(self, X):
        likelihood = np.zeros((X.shape[0], self.nc))
        for i in range(self.nc):
            likelihood[:,i] = stats.multivariate_normal(mean=self.mu[i], cov=self.sigma[i]).pdf(X)
        n = self.phi*likelihood
        d = np.sum(n, axis=1)[:,None]
        return n/d
    
    def Estep(self, X):
        self.weights = self.get_likeihood(X)
        self.phi = np.mean(self.weights)
    
    def Mstep(self, X):
        for i in range(self.nc):
            weight = self.weights[:,i][:,None]
            self.mu[i] = np.sum(X * weight) / np.sum(weight)
            self.sigma[i] = np.cov(X.T,
                                   aweights=(weight/np.sum(weight)).flatten(),
                                   bias=True)

    def fit(self, X, iters=1):
        self.phi = np.full(shape=self.nc,fill_value=1/self.nc)
        self.weights = np.full(shape=X.shape,fill_value=1/self.nc)
        self.mu = [np.random.choice(X.squeeze()) for _ in range(self.nc)]
        self.sigma = [np.cov(X.T) for _ in range(self.nc)]
        for _ in range(iters):
            self.Estep(X)
            self.Mstep(X)
    
    def predict(self, X):
        weights = self.get_likeihood(X)
        return np.argmax(weights, axis=1)