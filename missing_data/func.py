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

def get_t_and_mu(T):
	mu = calc_mean_T(T)
	data_is_missing = np.any(T_boole)
	if data_is_missing:
		t_list = []
		mu_list = []
		nan_indicies_list = []
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
			nan_indicies_list.append(nan_indices)
	else:
		t_list = []
		mu_list = []
		for i in range(0, N):
			t_list.append(T[:, i])
			mu_list.append(mu)
	return t_list, mu_list

def calc_S(t, mu, nan_list, D):
	N = len(t)
	S = np.zeros((D, D))
	for i in range(0, N):
		t_i = t[i]
		mu_i = mu[i]
		t_i = np.insert(t_i, nan_list, 0)
		mu_i = np.insert(mu_i, nan_list, 0)
		diff = t_i-mu_i
		mat = np.matmul(diff, np.transpose(diff))
		S += mat
	return S/N

def calc_W_new(S, M, W, sigma2):
	A = np.matmul(S, W) #SW
	B = sigma2*np.eye(M) # sigma2*I
	C = np.matmul(np.inverse(M), np.transpose(W)) #M^(-1)W^T
	D = np.matmul(S, W) #SW
	return np.matmul(A, np.inverse(B + np.matmul(C, D)))

def calc_sigma2_new(S, W, M_inv, W_new, D):
	A = np.matmul(S, W) #SW
	B = np.matmul(M_inv, W_new) #M^(-1)W_new
	return 1/D * np.trace(S - np.matmul(A, B))



def EM(T_missing, M):
	"""iteratively calculates W and sigma, treat missing data as latent variables"""
	pass

	T = T_missing
	T_boole = isnan(T)
	D = T.shape()[1]
	W_init = np.zeros((D, M))
	sigma2 = 1
	D = T.shape()[1]
	t_list, mu_list = get_t_and_mu(T)
	
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
    """equation 12.48 Bishop, calculates the expected value of the latent variable 
    returns 2xN-matrix with projected data"""
    return null
