from func import *
import matplotlib.pyplot as plt

np.random.seed(1)
<<<<<<< HEAD
f = 'tobomovirus.txt'
tobe_corrupted_data = read_data(f)
uncorrupted_data = read_data(f)
=======
f = '../dataset/tobomovirus.txt'
uncorrupted_data = read_data(f)
corrupted_data = remove_data(uncorrupted_data)
M = 2
D = uncorrupted_data[:, 1]
W, sigma2 = EM(corrupted_data, M)

M_inv_W_T = calc_M_inv_W_T(W, sigma2, M)

t_list, mu_list, nan_list = get_t_and_mu(corrupted_data, D)

expected_X = calc_expected_X(M_inv_W_T, t_list, mu_list, M)

plt.scatter(expected_X[0, :], expected_X[1, :])
plt.show()

>>>>>>> fe26439d2bdba7a802b04cd22c54dd3d8816d6eb

corrupted_data = remove_data(tobe_corrupted_data)

t_mean = calc_mean_T(uncorrupted_data)
#todo: test subtraction of matrices!
#print(uncorrupted_data[:,0])
#print(t_mean)
#print(uncorrupted_data[:,0] - t_mean)

EM_v1(uncorrupted_data, 2)