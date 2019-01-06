from func import *

np.random.seed(1)
f = 'tobomovirus.txt'
tobe_corrupted_data = read_data(f)
uncorrupted_data = read_data(f)

corrupted_data = remove_data(tobe_corrupted_data)

t_mean = calc_mean_T(uncorrupted_data)
#todo: test subtraction of matrices!
#print(uncorrupted_data[:,0])
#print(t_mean)
#print(uncorrupted_data[:,0] - t_mean)

EM_v1(uncorrupted_data, 2)