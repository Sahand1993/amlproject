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
	""" calculates the mean of the data, if corrupted it gives mean anyway"""
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
    return np.transpose(mean)

def get_t_and_mu(T, D):
	""" returns three lists, the first is a list of numpy column vectors of data points t
	 with missing data removed. The second is a list of mean vectors corresponding to the list of t-vectors
	 the third is a list of lists with the indices at which data were missing in the t-vectors"""
	T_boole = np.isnan(T)
	N = T.shape[1]
	print("N"+str(N))
	mu = calc_mean_T(T)
	data_is_missing = np.any(T_boole)
	if data_is_missing:
		t_list = []
		mu_list = []
		nan_indices_list = N*[[]]
		for i in range(0, N):
			t_i_missing = T[:, i]
			for j in range(0, D):
				if np.isnan(T[j, i]):
					copy = nan_indices_list[i].copy()
					copy.append(j)
					nan_indices_list[i] = copy
			nan_indices = nan_indices_list[i]
			t_i_removed = np.delete(t_i_missing, nan_indices)
			mu_i_removed = np.delete(mu, nan_indices)
			t_list.append(t_i_removed)
			mu_list.append(mu_i_removed)
	else:
		t_list = []
		mu_list = []
		nan_indices_list = N*[[]]
		for i in range(0, N):
			t_list.append(T[:, i])
			mu_list.append(mu)
	return t_list, mu_list, nan_indices_list

def calc_S(T, mu, t_list, mu_list, nan_list, D):
	""" calculates the matrix S"""
	N = len(t_list)
	S = np.zeros((D, D))
	for i in range(0, D):
		for j in range(0, N):
			if np.isnan(T[i, j]):
				T[i, j] = 0
	for i in range(0, N):
		t_i = T[:, i]
		mu_i = mu
		t_i = np.insert(t_i, nan_list[i], 0)
		mu_i = np.insert(mu_i, nan_list[i], 0)
		diff = t_i-mu_i
		mat = np.matmul(diff, np.transpose(diff))
		S += mat
	return S/N

def calc_W_new(S, W, M_inv, sigma2, M):
	""" calculates the new version of W"""
	A = np.matmul(S, W) #SW
	B = sigma2*np.eye(M) # sigma2*I
	C = np.matmul(M_inv, np.transpose(W)) #M^(-1)W^T
	D = np.matmul(S, W) #SW
	return np.matmul(A, np.linalg.inv(B + np.matmul(C, D)))

def calc_sigma2_new(S, W, W_new, M_inv, D):
	""" calculates the new sigma^2 """
	A = np.matmul(S, W) #SW
	B = np.matmul(M_inv, np.transpose(W_new)) #M^(-1)W_new
	return 1/D * np.trace(S - np.matmul(A, B))

def calc_M_inv(W, sigma2, M):
	""" calculates the inverse of the matrix M given W and sigma2"""
	M_mat = np.matmul(np.transpose(W), W) + sigma2*np.eye(M)
	M_mat_inverse = np.linalg.inv(M_mat)
	return M_mat_inverse

def calc_M_inv_W_T(W, sigma2, M):
	""" does the calculation M^(-1)W^T, which is needed to calculate expected values of X"""
	M_mat_inverse = calc_M_inv(W, sigma2, M)
	M_inv_W_T = np.matmul(M_mat_inverse, np.transpose(W))
	return M_inv_W_T

def calc_expected_X(M_inv_W_T, t_list, mu_list, M):
	""" calculates the current projections on the principas subspace (the latent variables"""
	N = len(t_list)
	expected_X = np.zeros((M, N))
	for i in range(0, N):
		x = np.matmul( M_inv_W_T, t_list[i]-mu_list[i])
		print(x)
		expected_X[:, i] = x
	return expected_X

def calc_expected_XX(expected_X, sigma2, M_inv):
	""" calculates expression 29 in Tipping Bishop 1999"""
	expected_XX = np.zeros((M, M, N))
	for i in range(0, N):
		expected_XX[:, :, i] = sigma2*M_inv + np.matmul(expected_X[i], np.transpose(expected_X[i]))
	return expected_XX

def EM(T_missing, M):
	"""iteratively calculates W and sigma, treat missing data as latent variables"""

	T = T_missing
	D = T.shape[0]
	W_init = np.zeros((D, M))
	sigma2 = 1
	mu = calc_mean_T(T)
	t_list, mu_list, nan_list = get_t_and_mu(T, D)
	W = 5*np.ones((D, M))
	sigma2 = 1
	S = calc_S(T, mu, t_list, mu_list, nan_list, D)
	M_inv = calc_M_inv(W, sigma2, M)
	repeat = True
	max_iter = 5000
	counter = 0
	while counter < max_iter:

		W_new = calc_W_new(S, W, M_inv, sigma2, M)
		M_inv_new = calc_M_inv(W_new, sigma2, M)
		sigma2_new = calc_sigma2_new(S, W, W_new, M_inv_new, M)
		if abs(sigma2 - sigma2_new) < 0.001:
			repeat = False
		W = W_new
		sigma2 = sigma2_new
		M_inv = M_inv_new
		counter+=136

	return W, sigma2

	

