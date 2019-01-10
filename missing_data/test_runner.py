from func import * 
import numpy as np

T = np.array([[5, 6],
			  [3, np.nan]])
D = 1
t, mu, index = get_t_and_mu(T, D)
print(t)
	