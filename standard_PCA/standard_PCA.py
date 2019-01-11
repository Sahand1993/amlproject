from scipy import linalg
import numpy as np 
import numpy.matlib
import numpy as np
import sklearn.datasets as ds
import os
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col
from matplotlib import cm

class PCA(object):
    def __init__(self, X,K = 2):
        """

        X : with N observations and D dimensions
        K : The number of principal components wanted by the user

        
        """
        N,D = np.shape(X)
        self.N = N
        self.D = D
        self.data = X
        self.K = K
        self.proj_data = np.zeros((N,K))

        
    def fit(self) :
        
        X = self.data
        N,D = np.shape(X)
        
        
        # u = sum(x_i)/N
        mu = X.mean(0)
        
        # X = [x_1 - u, x_2 - u, ..., x_N - u, ]
        Xm = X - np.matlib.repmat(mu,N,1)
        
    
        data = np.matmul(Xm.T, Xm) 
        U,S,V = linalg.svd(data, full_matrices=True)
        
        W = np.matmul(U, np.diag(np.sqrt(S)) )
    
        
        for i, x_i in enumerate(X) :
            self.proj_data[i,:] = np.matmul(W[:,0:self.K].T , Xm[i,:])
            
        self.proj_data /= 100
            
        return self.proj_data, W, mu 
        
    def plot_digits(self, ax=None):
        
        x = self.proj_data
        xx = x[:,0]
        yy = x[:,1]
        
        
        width = np.max(xx) - np.min(xx)
        height = np.max(yy) - np.min(yy)
        ax = plt.gca() if ax is None else ax
        ax.set_xlim([np.min(xx) - 0.1 * width, np.max(xx) + 0.1 * width])
        ax.set_ylim([np.min(yy) - 0.1 * height, np.max(yy) + 0.1 * height])
        
        for digit, x in enumerate (zip(xx, yy)):
            ax.text(x[0], x[1], digit+10, color='k')
            
        
        plt.show()

data = np.loadtxt("tobomovirus.txt")
pca_model = PCA(data, K = 2 )
pca_model.fit()
pca_model.plot_digits()

