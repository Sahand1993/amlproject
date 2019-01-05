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

def remove_data(T, num_rmv=136):
    """ removes num_rmv number of data point elements randomly
        input: uncorrupted matrix 
        output: corrupted data matrix with num_rmv random elements set to NaN
    """
    D = T.shape[0]
    N = T.shape[1]
    corrupting = np.ones(N*D)
    corrupting[0:num_rmv] = np.nan
    np.random.shuffle(corrupting)

    count = 0 
    for i in range(D):
        for j in range(N):
            T[i][j] = T[i][j]*corrupting[count]
            count += 1

    return T

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
    D = T.shape()[1]
    W_init = np.zeros((D, M))
    sigma2 = 1
    mu = calc_mean_T(T)
    M_mat = np.matmul(numpy.transpose(W), W) + sigma2*np.eye(M, M)
    M_mat_inverse = np.inverse(M_mat)
    expected_X = np.matmul(np.matmul(M_mat_inverse, np.transpose(W)) , 
                            (T.transpose() - mu).transpose())

def projection(W, sigma):
    """equation 12.49 Bishop, calculates the expected value of the latent variable 
    returns 2xN-matrix with projected data"""
    return null
