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


def EM_v1(T, M):
    """ T is 18x38 
    """
    #declare variables&constants
    L_c = 0     #measurement of convergence
    T_bool = np.isnan(T)   
    data_is_missing = np.any(T_bool)
    D = T.shape[0]
    N = T.shape[1]
    mu = calc_mean_T(T)
    sig2_init = 1
    sig2_current = sig2_init
    W_init = np.ones([D,M])
    W_current = W_init
    M_mat = np.dot(W_init.T, W_init) + sig2_init*np.eye(M)
    M_mat_inv = np.linalg.inv(M_mat)
    t_list = []
    mu_list = []
    E_X = np.zeros([N, M])
    E_XX = np.zeros([N, M, M])
    mean_diff = np.zeros([D, N])
    # calculate expected values for x_n and x_n * x_n.T
    if data_is_missing:
        print("data is missing!")


    else:
        print("data is not missing :D ")
        for i in range(N):
            diff = T[:,i] - mu
            mean_diff[:,i] = diff
            #diff = diff.reshape(D,1) 
            E_x_n = np.dot(M_mat_inv, np.dot(W_current.T, diff))
            E_X[i,:] = E_x_n
            E_xx_n = sig2_current*M_mat_inv + np.dot(E_x_n, E_x_n.T)
            E_XX[i,:,:] = E_xx_n

    L_c = conv_calc(T, mu, sig2_current, E_X, E_XX, W_current)
    #calcuclate convergence measurement L_c
    """
    for i in range(N):
        term_1 = 0.5 * D * np.log(sig2_current)
        term_2 = 0.5 * np.trace(E_XX[i])
        term_3 = 0.5 * 1/sig2_current * np.dot(mean_diff[:,i].T, mean_diff[:,i])
        term_4 = -1/sig2_current * np.dot(E_X[i,:].T, np.dot(W_current.T, mean_diff[:,i]))
        term_5 = 0.5*sig2_current * np.trace(np.dot(W_current.T, np.dot(W_current, E_XX[i])))
        L_c += term_1 + term_2 + term_3 + term_4 + term_5
    L_c = -L_c
    """
    print(L_c)


def conv_calc(T, mu, sig2, E_X, E_XX, W):
    """ calculates convergence float L_c
        input: data T, mean values mu, variance sig2, expected latent values E_X, expected latent values product E_XX, projection matrix W
        output: L_c value
    """
    L_c = 0
    N = T.shape[1]
    D = T.shape[0]

    for i in range(N):
        t_mu_diff = T[:,i] - mu
        term_1 = 0.5 * D * np.log(sig2)
        term_2 = 0.5 * np.trace(E_XX[i])
        term_3 = 0.5 * 1/sig2 * np.dot(t_mu_diff.T, t_mu_diff)
        term_4 = -1/sig2 * np.dot(E_X[i,:].T, np.dot(W.T, t_mu_diff))
        term_5 = 0.5*sig2 * np.trace(np.dot(W.T, np.dot(W, E_XX[i])))
        
        L_c += term_1 + term_2 + term_3 + term_4 + term_5

    L_c = -L_c

    return L_c


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
