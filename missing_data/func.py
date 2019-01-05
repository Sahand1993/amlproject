import numpy as np
import math

def read_data(filename, d=18, n=38):
	""" reads text file and creates matrix with data
		input: file name and matrix dimensions
		output: (transposed) matrix with data as column vectors
	"""
	f = open(filename)
	matrix = np.zeros([n,d])

	for i in range(n):
		data_point = np.asarray([int(s) for s in f.readline().split()])
		matrix[i] = data_point

	return np.transpose(matrix)

def remove_data(T):
	"""removes 20 procent of data randomly"""
	"""extra testkommentar"""
	return null

def calc_mean_T(T_missing):
	T = T_missing
	D = T.shape[0]
	N = T.shape[1]
	mean = np.zeros(D)
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
	return null

	T = T_missing
	T_boole = isnan(T)
	D = T.shape()[1]
	W_init = np.zeros((D, M))
	sigma2 = 1
	mu = calc_mean_T(T)
	D = T.shape()[1]
	mu = calc_mean_T(T)
	data_is_missing = np.any(T_boole)

	if data_is_missing:
		t_list = []
		mu_list = []
		for i in range(0, N):
			t_i_missing = T[:, i]
			nan_indices = []
			for j in range(0, D):
				if np.isnan(T[i, j]):
					nan_indices.append(j)
			t_i_removed = np.delete(t_i_missing, nan_indices)
			mu_i_removed = np.delete(mu, nan_indices)
			t_list.append(t_i_removed)
			mu_list.append(mu_i_removed)
	else:
		t_list = []
		mu_list = []
		for i in range(0, N):
			t_list.append(T[:, i])
			mu_list.append(mu)
	
	W_init = np.zeros((D, M))
	sigma2 = 1

	M_mat = np.matmul(numpy.transpose(W), W) + sigma2*np.eye(M)
	M_mat_inverse = np.inverse(M_mat)
	M_inv_W_T = np.matmul(M_mat_inverse, np.transpose(W))
	expected_X = np.zeros((M, N))
	for i in range(0, N):
		expected_X[:, i] = np.matmul( M_inv_W_T, t_list[i]-mu_list[i])
	expected_XX = np.zeros((M, M, N))
	for i in range(0, N):
		expected_XX[:, :, i] = sigma2*M_mat_inverse + np.matmul(expected_X[i], np.transpose(expected_X[i]))



def projection(W, sigma):
	"""equation 12.49 Bishop, calculates the expected value of the latent variable 
	returns 2xN-matrix with projected data"""
	return null
