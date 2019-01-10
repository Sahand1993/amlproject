from func import *
from julia_func import *
import matplotlib.pyplot as plt

np.random.seed(1)

f = '../dataset/tobomovirus.txt'
uncorrupted_data = read_data(f)
#corrupted_data = remove_data(uncorrupted_data)
#corrupted_data_copy = corrupted_data.copy()
M = 2
N = uncorrupted_data[0, :].size
D = uncorrupted_data[:, 1].size

"""corrupted"""
<<<<<<< HEAD
W, sigma2 = EM(corrupted_data, M, True)
=======
"""
W, sigma2 = EM_v1(corrupted_data, M)
>>>>>>> 1c0915018a697ad3937688da51aac67289e0846d
M_inv = calc_M_inv(W, sigma2, M)
t_list, mu_list, nan_list = get_t_and_mu(corrupted_data_copy, D)
#print('M inv!', M_inv)
#print('W ! ', W)
expected_X = calc_expected_X(M_inv, W, t_list, mu_list, nan_list, M)
fig1, ax1 = plt.subplots()
plt.scatter(expected_X[0, :], expected_X[1, :])
for i in range(0, N):
    ax1.annotate(str(i), (expected_X[0, i], expected_X[1, i]))
plt.show()
#fig1.savefig("projection_with_missing_data.pdf", bbox_inches='tight')
"""

"""uncorrupted"""
<<<<<<< HEAD
W, sigma2 = EM(uncorrupted_data, M, False)
=======
test_func()
W, sigma2 = EM_v1(uncorrupted_data, M)
>>>>>>> 1c0915018a697ad3937688da51aac67289e0846d
M_inv = calc_M_inv(W, sigma2, M)
t_list, mu_list, nan_list = get_t_and_mu(uncorrupted_data, D)
expected_X = calc_expected_X(M_inv, W, t_list, mu_list, nan_list, M)
fig2, ax2 = plt.subplots()
plt.scatter(expected_X[0, :], expected_X[1, :])
print(sigma2)
for i in range(0, N):
    ax2.annotate(str(i), (expected_X[0, i], expected_X[1, i]))
plt.show()
#fig2.savefig("projection_with_normal_data.pdf", bbox_inches='tight')

