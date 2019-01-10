import func as f
import numpy as np

T = np.array([[5, 6],
			  [3, np.nan]])
D = 1
mu = f.calc_mean_T(T)
print(mu)
	