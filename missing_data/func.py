
import numpy as np

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

def EM(T_missing):
	"""iteratively calculates W and sigma, treat missing data as latent variables"""
	return null

def projection(W, sigma):
	"""equation 12.49 Bishop, calculates the expected value of the latent variable 
	returns 2xN-matrix with projected data"""
	return null
