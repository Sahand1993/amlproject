import numpy as np
from func import *

def EM_v1(T, M):
    """ T is 18x38 
    """
    #declare variables&constants
    L_c = 0     #measurement of convergence
    max_iter = 5
    iter_count = 0
    T_bool = np.isnan(T)   
    data_is_missing = np.any(T_bool)
    D = T.shape[0]
    N = T.shape[1]
    mu = calc_mean_T(T)
    mu = mu.reshape([D, 1])
    sig2 = .00001
    W = np.random.rand(D,M)
    t_list, mu_list, nan_list = get_t_and_mu(T, D) 
    
    while(iter_count < max_iter):
        iter_count += 1
        print("~~~~~~~~~~~~~~~~~~~ iteration ", iter_count, "~~~~~~~~~~~~~~~~~~~~~~")
        #print(L_c)
        L_c, E_X, E_XX = E_step(T, sig2, W, L_c, t_list, mu_list, nan_list, M, data_is_missing)
        #print('expeced x', E_X)
        W_new, sig2_new = M_step(T, E_X, E_XX, mu, N, D ,M, data_is_missing)
        W = W_new
        sig2 = sig2_new

    return W, sig2

def E_step(T, sig2, W, L_c, t_list, mu_list, nan_list, M, is_missing):

    D = T.shape[0]
    N = T.shape[1]
    mu = calc_mean_T(T)
    mu = mu.reshape([D, 1])
    diff_list = []
    E_X = np.zeros([M, N])
    E_XX = np.zeros([N, M, M])  #update every iteration?
    W_list = get_list_of_W(W, nan_list, N, D, M)
    #print('W_i list!!! 0:', W_list[0])
    #print('W_i list 1', W_list[1])
    # calculate expected values for x_n and x_n * x_n.T
    if is_missing:
        #print("data is missing!")
        for i in range(N):
            diff = t_list[i] - mu_list[i]
            #print('diff at data point ', i, ': ', diff)
            diff_list.append(diff)
            W_i = W_list[i]
            M_inv = calc_M_inv(W_i, sig2, M)
            #print('m_inv', M_inv)
            E_x_n = np.dot(M_inv, np.dot(W_i.T, diff))
            E_xx_n = sig2*M_inv + np.dot(E_x_n, E_x_n.T)
            E_X[:,i] = E_x_n
            E_XX[i,:,:] = E_xx_n 

    else:
        print("data is not missing :D ")
        for i in range(N):
            t_i = T[:,i].reshape(D, 1)
            diff = t_i - mu
            diff_list.append(diff)
            M_mat_inv = calc_M_inv(W, sig2, M) 
            E_x_n = np.dot(M_mat_inv, np.dot(W.T, diff))
            E_X[:,i] = E_x_n[:,0]
            E_xx_n = sig2*M_mat_inv + np.dot(E_x_n, E_x_n.T)
            E_XX[i,:,:] = E_xx_n
    #print('diff at data point 1', diff_list[0])
    #print('diff at data point 2', diff_list[1])

    #calcuclate convergence measurement L_c
    L_c = conv_calc(T, mu, sig2, E_X, E_XX, W, W_list, diff_list)

    print("convergence float: ", L_c)
    return L_c, E_X, E_XX

def M_step(T, E_X, E_XX, mu, N, D, M, is_missing):
    T_zeros = add_zeros(T, N, D)
    W_new = np.zeros([D, M])
    sig2_new = 0
    W_factor1 = np.zeros([D, M])
    W_factor2 = np.zeros([M, M])

    if is_missing:
        for i in range(N):
            E_x_n = E_X[:,i].reshape(M,1)
            t_n = T_zeros[:,i].reshape(D,1)
            t_mu_diff = t_n - mu
            W_factor1 += np.dot(t_mu_diff, E_x_n.T)
            W_factor2 += E_XX[i]
            #W_new += np.dot(np.dot(t_mu_diff, E_x_n.T), E_XX[i])
            #sig2_new += np.linalg.norm(t_mu_diff)**2 - 2 * np.dot(E_x_n.T, np.dot(W_new.T, t_mu_diff)) 
            #+ np.trace(np.dot(E_XX[i], np.dot(W_new.T, W_new)))
        W_new = np.dot(W_factor1, np.linalg.inv(W_factor2))
        for i in range(N):
            E_x_n = E_X[:,i].reshape(M,1)
            t_n = T_zeros[:,i].reshape(D,1)
            t_mu_diff = t_n - mu
            sig2_new += np.linalg.norm(t_mu_diff)**2 - 2 * np.dot(E_x_n.T, np.dot(W_new.T, t_mu_diff)) 
            + np.trace(np.dot(E_XX[i], np.dot(W_new.T, W_new)))
        sig2_new = sig2_new[0][0]/(N * D)

    else:
        for i in range(N):
            E_x_n = E_X[:,i].reshape(M,1)
            t_n = T[:,i].reshape(D, 1)
            t_mu_diff = t_n - mu
            W_factor1 += np.dot(t_mu_diff, E_x_n.T)
            W_factor2 += E_XX[i]

        W_new = np.dot(W_factor1, np.linalg.inv(W_factor2))

        for i in range(N):
            E_x_n = E_X[:,i].reshape(M,1)
            t_n = T[:,i].reshape(D, 1)
            t_mu_diff = t_n - mu
            sig2_new += np.linalg.norm(t_mu_diff)**2 - 2 * np.dot(E_x_n.T, np.dot(W_new.T, t_mu_diff)) 
            + np.trace(np.dot(E_XX[i], np.dot(W_new.T, W_new)))

        sig2_new = sig2_new[0][0]/(N * D)
    print(sig2_new)
    #print('W_new! ', W_new)
    return W_new, sig2_new
