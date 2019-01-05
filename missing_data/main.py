from func import *

f = 'tobomovirus.txt'
complete_data = read_data(f)
#print(complete_data)
#print(complete_data.shape)
"""
testarr = complete_data[0]
corrupting = np.ones(38)
corrupting[0:8] = np.nan
np.random.shuffle(corrupting)
prod = testarr*corrupting
print(testarr)
print(corrupting)
print(prod)
"""
np.random.seed(2)
corrupted = remove_data(complete_data)
print(corrupted)