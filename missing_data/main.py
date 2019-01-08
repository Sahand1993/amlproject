from func import *
import matplotlib.pyplot as plt

np.random.seed(1)
f = '../dataset/tobomovirus.txt'
uncorrupted_data = read_data(f)
corrupted_data = remove_data(uncorrupted_data)
corrupted_data.flags.writeable = False
M = 2
W, sig = EM_v1(corrupted_data, M)

"""

D = uncorrupted_data[:, 1]
M = 2
W, sigma2 = EM(corrupted_data, M)

M_inv_W_T = calc_M_inv_W_T(W, sigma2, M)

t_list, mu_list, nan_list = get_t_and_mu(corrupted_data, D)

expected_X = calc_expected_X(M_inv_W_T, t_list, mu_list, M)

plt.scatter(expected_X[0, :], expected_X[1, :])
plt.show()

"""