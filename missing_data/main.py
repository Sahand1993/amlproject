from func import *
from julia_func import *
import matplotlib.pyplot as plt
import copy

np.random.seed(1)

f = '../dataset/tobomovirus.txt'
uncorrupted_data = read_data(f)
uncorrupted_data_copy = copy.deepcopy(uncorrupted_data)
corrupted_data = remove_data(uncorrupted_data)
corrupted_data_copy = copy.deepcopy(corrupted_data)

M = 2
N = uncorrupted_data[0, :].size
D = uncorrupted_data[:, 1].size

x_min = -2
x_max = 1
y_min = -1.5
y_max = 1

"""Calculate W and sigma2, and project t-vectors on principal components"""

"""corrupted"""
W, sigma2 = EM(corrupted_data, M, True)									#find W and sigma2 with EM
M_inv = calc_M_inv(W, sigma2, M) 										#Calculate M^(-1)
t_list, mu_list, nan_list = get_t_and_mu(corrupted_data_copy, D) 		#sort out NaN elements and sava indeces
expected_X = calc_expected_X(M_inv, W, t_list, mu_list, nan_list, M) 	#project points
fig1, ax1 = plt.subplots()
plt.scatter(expected_X[0, :], expected_X[1, :]) 						#plot data
for i in range(0, N):
    ax1.annotate(str(i), (expected_X[0, i], expected_X[1, i])) 			#add numbers to points in plots
ax1.set_xlim([x_min,x_max]) 											#set axis
ax1.set_ylim([y_min,y_max])
plt.show()
fig1.savefig("projection_with_missing_data.pdf", bbox_inches='tight')

"""uncorrupted"""
W, sigma2 = EM(uncorrupted_data, M, False)								#find W and sigma2 with EM
M_inv = calc_M_inv(W, sigma2, M)										#Calculate M^(-1)
t_list, mu_list, nan_list = get_t_and_mu(uncorrupted_data_copy, D)		#sort out NaN elements and sava indeces
expected_X = calc_expected_X(M_inv, W, t_list, mu_list, nan_list, M)	#project points
Theta = 1.1*np.pi 														#Set rotation angle
R = np.array([[np.cos(Theta), -np.sin(Theta)],							#Calculate rotation matrix
			  [np.sin(Theta),  np.cos(Theta)]])			
X_rotated = np.dot(R, expected_X)										#rotate projected points
expected_X = X_rotated
fig2, ax2 = plt.subplots()
plt.scatter(expected_X[0, :], expected_X[1, :])							#plot data
for i in range(0, N):
    ax2.annotate(str(i), (expected_X[0, i], expected_X[1, i]))			#add numbers to points in plots
ax2.set_xlim([x_min,x_max])												#set axis
ax2.set_ylim([y_min,y_max])
plt.show()
fig2.savefig("projection_with_normal_data.pdf", bbox_inches='tight')

