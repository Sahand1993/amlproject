from func import *

np.random.seed(1)
f = '../dataset/tobomovirus.txt'
uncorrupted_data = read_data(f)
corrupted_data = remove_data(uncorrupted_data)

