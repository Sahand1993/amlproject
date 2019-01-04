import numpy as np
import math

def read_data(file):
	"""reads text file and creates matrix with data"""
	return null

def remove_data(T):
	"""removes 20 procent of data randomly"""
	"""extra testkommentar"""
	return null

def calc_mean_T(T_missing):
	T = T_missing
	D = T.shape[0]
	N = T.shape[1]
	mean = np.zeros((D, 1))
	for i in range(0, D):
		mean_i = 0
		missing_counter = 0
		for j in range(0, N):
			t = T[i, j]
			if math.isnan(t):
				missing_counter += 1
			else:
				mean_i +=t
		mean[i] = mean_i/(N-missing_counter)
	return mean


def EM(T_missing, M):
	"""iteratively calculates W and sigma, treat missing data as latent variables"""
	T = T_missing
	D = T.shape()[1]
	W_init = np.zeros((D, M))
	sigma2 = 1
	mu = calc_mean_T(T)
	M_mat = np.matmul(numpy.transpose(W), W) + sigma2*np.eye(M, M)
	M_mat_inverse = np.inverse(M_mat)
	expected_X = np.matmul(np.matmul(M_mat_inverse, np.transpose(W)) , 
							(T.transpose() - mu).transpose())




	return null

def projection(W, sigma):
	"""equation 12.49 Bishop, calculates the expected value of the latent variable 
	returns 2xN-matrix with projected data"""
	return null

def julias_function(param):
	return 5

	