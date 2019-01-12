import seaborn as sns
import numpy as np 
import numpy.matlib
import numpy as np
import sklearn.datasets as ds
import os
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col
from matplotlib import cm

class MPPCA(object):
    def __init__(self, X,d,C):
        """
        Do not know why d == C
        X : data matrix NXD, N : Data points D : Dimensions must be transposed X.T
        d : Latent dimensionality
        C : Number of mixtures components
        
        """
        self.epsilon = 1e-3
        N,D = np.shape(X)
        # Initialization of mixtures variables
        self.data = X
        self.dim = D
        self.N = N
        self.C = C
        self.d = d
        self.R = np.zeros((N,C))
        # Initialization of mixtures components 
        self.mix = np.ones(C)/C
        # Initialization of means
        p = np.random.permutation(N)[0:C]
        self.means = X[p,:]
        # Initialization of noise variance
        # self.noise_var = (np.var(X))*np.ones((C, D))
        # self.noise_var =np.array( [np.var(X)] * C)
        self.noise_var = np.var(X) * np.ones(C)
        # Initialization of mixtures Weights W 
        self.W = np.random.randn(self.C,D,d)
        self.W_old = np.random.randn(self.C,D,d)
        
        self.M_i = np.random.randn(self.C,d,d)
        self.M_i_inv = np.random.randn(self.C,d,d)

    def EM(self) :
        C,D,d = np.shape(self.W)
        self.proj = np.zeros((self.C,self.d, self.N))
        self.LogL = np.zeros((self.C,self.N))
        self.temp = np.zeros((self.N,self.C))
        
        # E-step
        for i in range(self.C) :

            # Compute M_i
            self.M_i[i,:,:] = self.noise_var[i]*np.eye(self.d) + np.dot(self.W[i,:,:].T, self.W[i,:,:])
            
            # Compute C_inverse_i
            self.M_i_inv[i,:,:] = np.linalg.pinv(self.M_i[i,:,:])
            C_inv = np.eye(self.dim) - np.dot(np.dot(self.W[i,:,:], self.M_i_inv[i,:,:]),self.W[i,:,:].T )
            C_inv = C_inv / self.noise_var[i]
            
            #print("Covar  = ", C_inv )
            
            # Compute R_ni 
            # TODO Write a new log likehood, new gaussian plot with the data included, 
            
            const = -(self.dim/2)*np.log(2*np.pi) - 0.5*np.linalg.det(np.eye(self.dim) - np.dot(np.dot(self.W[i,:,:], self.M_i_inv[i,:,:]), self.W[i,:,:].T))
            Xm = self.data - np.matlib.repmat(self.means[i,:],self.N,1)
            temp = const - 0.5 * np.diag( np.matmul (Xm , np.matmul(C_inv, Xm.T )) )
            
            
            constant = -(self.dim/2)*np.log(2*np.pi) - 0.5*np.linalg.det(np.eye(self.dim) - np.dot(np.dot(self.W[i,:,:], self.M_i_inv[i,:,:]), self.W[i,:,:].T))
            constant += np.log(self.mix[i]) -(self.dim/2)*np.log(self.noise_var[i])
            # constant += -(self.dim/2)*np.log(self.noise_var[i])
            Xmean = self.data - np.matlib.repmat(self.means[i,:],self.N,1)
            wait = constant - 0.5 * np.diag( np.matmul (Xmean , np.matmul(C_inv, Xmean.T )) )
            # wait = self.R[:,i]*wait #Can be wrong !!! 
            
            
            self.temp[:,i] = temp 
            
            print("Temp 1 = ", self.temp[:,i])
            
            self.temp[:,i] = wait
            
            print("Temp 2 = ", self.temp[:,i])
        
        # Compute Responsability a
        
        self.temp = self.temp + np.matlib.repmat(self.mix, self.N, 1)
        s1,s2 = np.shape(self.temp)
        
        Q = np.exp(self.temp - np.matlib.repmat(self.temp.max(1),1,self.C).reshape(s2,s1).T)
        denom = np.matlib.repmat(Q.sum(1),1,self.C).reshape(s2,s1).T
        Q = Q / denom
        self.R = Q
        print("Resp = ", Q)
        
        # Update of mixtures coefficients 
        self.mix =  Q.mean(0)
        

        # M - step :
        for i in range(self.C) :
            
            ## Update MEAN 
            numerator = Q[:,i].reshape((self.N,1)) * self.data
            numerator = numerator.sum(0)
            denominator = Q[:,i].sum()
            self.means[i,:] = numerator / denominator

            ##Compute S_i
            Xm = self.data - self.means[i,:]

            Rm = Q[:,i].reshape((self.N,1)) * Xm
            # Rm = Q[:,i] * Xm.T
            S_i = (1/(self.mix[i] * self.N)) * np.matmul(Rm.T, Xm)
            # print("S_i = ", S_i)
            
            
            ## Update Weight W
            self.W_old[i,:,:] = self.W[i,:,:]
        
            t1 = np.matmul(self.M_i_inv[i,:,:],self.W[i,:,:].T )
            # print("t111 = ", t1)
            t2 = np.matmul(t1, S_i)
            t3 = np.matmul(t2, self.W[i,:,:])
            t4 = self.noise_var[i]*np.eye(self.d) 
            # print()
            # print("t4 = ", t4)
            # print()
            # print("t3 = ", t3)
            temporary = t3 + t4 

            inverse_cov = np.linalg.pinv(temporary)
            sw = np.matmul(S_i,self.W[i,:,:])
            W_new = np.matmul(sw, inverse_cov)
            self.W[i,:,:] = W_new
            
            ## Update sigma  self.W_old
            # print()
            # print("OLD *** Weight W = ",self.W_old[i,:,:] )
            # print("NEW *** Weight W = ",self.W[i,:,:] )
            
            A = np.matmul(self.M_i_inv[i,:,:], self.W[i,:,:].T )
            B = np.matmul(self.W_old[i,:,:], A) 
            C = np.matmul(S_i,B) 
            sigma = S_i - C
            sigma = (1/(self.dim)) * np.trace(sigma)
            self.noise_var[i] = sigma
            
            # print("Sigma = ", self.noise_var)

    def project_data(self, d = 0):
        
        cluster = np.argmax(self.R[d])
        A = np.matmul(self.M_i_inv[cluster,:,:], self.W[cluster,:,:].T )
        Xm = self.data[d] - self.means[cluster,:]
        B = np.dot(A,Xm)
        print("Cluster : ", cluster)
        print("Data index : ", d, " Resp.: ", self.R[d], " Proj : ", B)
        return B, cluster

    def print_parameter(self) :
    
        print("W = ", self.W)
        print()
        print("Mix coeff = ", self.mix)
        print()
        print("Sigma noise = ", self.noise_var)
        print()
        print("mean = ", self.means)
        print()

    def plot_mixtures(self, x, K) :
        
        for i in range(K):
            data, digits = x[i]
            data = np.array(data)
            couleur = ['b','r', 'y','c']
            if data != [] :
                self.plot_mix(data, digits,i)
    
        plt.show()
        
        return 0
        
    def plot_mix(self, x, digits,img,  ax=None):
        
        # sns.set()
        
        sns.set_style("ticks")
        
        fig = plt.figure()
        if self.C <= 3 :
            col = np.zeros(3)
            col[img] = 244 / 255
        else :
            col = np.random.random(3)
            
        xx = x[:,0]
        yy = x[:,1]
        width = np.max(xx) - np.min(xx)
        height = np.max(yy) - np.min(yy)
        ax = plt.gca() if ax is None else ax
        ax.set_xlim([np.min(xx) - 0.1 * width, np.max(xx) + 0.1 * width])
        ax.set_ylim([np.min(yy) - 0.1 * height, np.max(yy) + 0.1 * height])
    
    
        for index, x in enumerate (zip(xx, yy)):
    
            ax.text(x[0], x[1], digits[index]+10, color=col)
            
        fig.savefig('mix'+ str(img) + '.png')

    def plot_by_clusters(self ) : 
        N, D = np.shape(self.data)
        labels = np.zeros((N,1))
        data_proj = np.zeros((self.N,self.d))
        clusters = []
        for i in range(number_mix) :
            clusters.append([[],[]])
            
        for i, data_point in enumerate(self.data) :
            data_proj[i,:], labels[i,:] = self.project_data(i)
            label = int(labels[i,:])
            clusters[label][0].append(data_proj[i,:].tolist())
            clusters[label][1].append(i)
        
        print()
        print(clusters)  
        print()
        
        self.plot_mixtures(clusters, number_mix )
        
        print(labels)


data = np.loadtxt("tobomovirus.txt")
# data = np.loadtxt("tobomovirus.txt")[0:10]
# data = [[1,2 ,3 ],[ 5, 6 ,9 ],[ -2, 3, 4]]
# data = [[1,2 ,3 ,5],[ 5, 6 ,9 ,10],[ -2, 3, 4, 11]]
# data = np.random.randint(15, size=(10, 4))
# data = [[1,2 ,3 ,5,0],[ 5, 6 ,9 ,10,-7],[ -2, 3, 4, 11, -8]],
latent_dim = 2
number_mix = 3
data = np.array(data)
N, D = np.shape(data)
pca_model = MPPCA(data,latent_dim,number_mix)


print("**************** EM step *****************")
for i in range(D):
    pca_model.EM()
print("**************** END ******************")


pca_model.plot_by_clusters()