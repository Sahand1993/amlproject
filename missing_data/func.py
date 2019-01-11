import numpy as np
import copy


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
    T_corrupt = np.zeros([D,N])
    count = 0 
    for i in range(D):
        for j in range(N):
            T_corrupt[i][j] = T[i][j]*corrupting[count]
            count += 1

    return T_corrupt

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
            #math.isnan?
            if np.isnan(t):
                missing_counter += 1
            else:
                mean_i +=t
        mean[i] = mean_i/(N-missing_counter)
    return np.reshape(mean, (D, 1))

def add_zeros(T, N, D):
    T_add = np.zeros([D,N])
    for i in range(D):
        for j in range(N):
            if np.isnan(T[i][j]):
                T_add[i][j] = 0
            else:
                T_add[i][j] = T[i][j]
    return T_add

def conv_calc(T, mu, sig2, E_X, E_XX, W, W_list, diff_list):
    """ calculates convergence float L_c
        input: data T, mean values mu, variance sig2, expected latent values E_X, expected latent values product E_XX, projection matrix W
        output: L_c value
    """
    L_c = 0
    N = T.shape[1]
    D = T.shape[0]

    for i in range(N):
        term_1 = 0.5 * D * np.log(sig2)
        term_2 = 0.5 * np.trace(E_XX[i])
        term_3 = 0.5 * 1/sig2 * np.dot(diff_list[i].T, diff_list[i])
        term_4 = -1/sig2 * np.dot(E_X[:,i].T, np.dot(W_list[i].T, diff_list[i]))
        term_5 = 0.5*sig2 * np.trace(np.dot(W_list[i].T, np.dot(W_list[i], E_XX[i])))
        
        L_c += term_1 + term_2 + term_3 + term_4 + term_5

    L_c = -L_c

    return L_c

def get_list_of_W(W_orig, nan_list, N, D, M):
    """
    Adjusts W to have correct dimensions according to each data point with missing values.
    Input: "normal" W matrix with dimensions (DxM)
    Output: list of (K_nxM) matrices, where K_n is the amount of dimensions in data point n
    """
    W_list = []
    for i in range(N):
        W = W_orig
        W = np.delete(W, nan_list[i], 0)
        W_list.append(W)
        #print(W_list[i].shape)
    return W_list

def get_t_and_mu(T, D):
    """ returns three lists, the first is a list of numpy column vectors of data points t
     with missing data removed. The second is a list of mean vectors corresponding to the list of t-vectors
     the third is a list of lists with the indices at which data were missing in the t-vectors"""
    T_boole = np.isnan(T)
    N = T.shape[1]
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
            t_list.append(T[:, i].reshape(D, 1))
            mu_list.append(mu)
    return t_list, mu_list, nan_indices_list

def calc_S(T, mu, t_list, mu_list, nan_list, D):
    """ calculates the matrix S"""
    mu = mu.reshape(D, 1)
    N = len(t_list)
    S = np.zeros((D, D))
    for i in range(0, D):
        for j in range(0, N):
            if np.isnan(T[i, j]):
                T[i, j] = mu[i]
    for i in range(0, N):
        t_i = T[:, i]
        t_i = np.reshape(t_i, (D, 1))
        mu_i = mu
        diff = t_i-mu_i
        #print(diff)
        mat = np.dot(diff, np.transpose(diff))
        #print(mat)
        S += mat
    S = S/(N*0.8/2)
    return S

def calc_W_new(S, W, M_inv, sigma2, M):
    """ calculates the new version of W"""
    A = np.matmul(S, W) #SW
    B = sigma2*np.eye(M) # sigma2*I
    C = np.matmul(M_inv, np.transpose(W)) #M^(-1)W^T
    return np.matmul(A, np.linalg.inv(B + np.matmul(C, A)))

def calc_sigma2_new(S, W, W_new, M_inv_new, M_inv_old, D):
	""" calculates the new sigma^2 """
	A = np.matmul(S, W) #SW
	B = np.matmul(M_inv_old, np.transpose(W_new)) #M^(-1)W_new^T
	sigma2 = 1.0/D * np.trace(S - np.matmul(A, B))
	return sigma2

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

def calc_W_from_nan_index(W, nan_list):
    W = np.delete(W, nan_list, 0)
    return W

def calc_expected_X(M_inv, W, t_list, mu_list, nan_list, M):
    """ calculates the current projections on the principas subspace (the latent variables"""
    N = len(t_list)
    expected_X = np.zeros((M, N))
    for i in range(0, N):
        W_i = copy.deepcopy(W)
        nan_i = nan_list[i]
        W_i = calc_W_from_nan_index(W_i, nan_i)
        M_inv_W_T = np.dot(M_inv, np.transpose(W_i))
        x = np.dot(M_inv_W_T, t_list[i]-mu_list[i])
        #diff = t_list[i].reshape(t_list[i].shape[0],1) - mu_list[i].reshape(mu_list[i].shape[0],1)
        #x = np.dot(M_inv_W_T, diff)

        print(t_list[i].shape)
        print(mu_list[i].shape)
        diff = t_list[i]-mu_list[i]
        print(diff.shape)
        x = np.dot( M_inv_W_T, diff)
        expected_X[:, i] = x.reshape(2)
    return expected_X

def calc_expected_XX(expected_X, sigma2, M_inv, M, N):
    """ calculates expression 29 in Tipping Bishop 1999"""
    expected_XX = np.zeros((M, M, N))
    for i in range(0, N):
        expected_XX[:, :, i] = sigma2*M_inv + np.matmul(expected_X[i], np.transpose(expected_X[i]))
    return expected_XX

def EM(T_missing, M, probabalistic):
	"""iteratively calculates W and sigma, treat missing data as latent variables"""

	T = T_missing
	D = T.shape[0]
	mu = calc_mean_T(T)
	t_list, mu_list, nan_list = get_t_and_mu(T, D)
	W = (np.random.rand(D, M) - 0.5) * 10
	if probabalistic:
		sigma2 = 1.0
	else:
		sigma2 = 1*10**(-6)
	S = calc_S(T, mu, t_list, mu_list, nan_list, D)
	M_inv = calc_M_inv(W, sigma2, M)
	repeat = True
	max_iter = 100
	counter = 0
	while counter < max_iter:
		W_new = calc_W_new(S, W, M_inv, sigma2, M)
		M_inv_new = calc_M_inv(W_new, sigma2, M)
		if probabalistic:
			sigma2_new = calc_sigma2_new(S, W, W_new, M_inv_new, M_inv, D)
		else: 
			sigma2_new = sigma2
		if abs(sigma2 - sigma2_new) < 0.001:
			repeat = False
		W = W_new
		sigma2 = sigma2_new
		M_inv = M_inv_new
		counter += 1
	return W, sigma2
