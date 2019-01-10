from func import * 
import numpy as np

T = np.array([[1, 2, np.nan],
			  [2, 3, 4     ],
			  [np.nan, 1, 3]])
D = T.shape[0]

mu = calc_mean_T(T)

t_list, mu_list, index = get_t_and_mu(T, D)

S = calc_S(T, mu, t_list, mu_list, index, D)

print(3*0.8/2*S)
	