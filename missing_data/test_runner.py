import numpy as np
import func as f

T = np.array([[1, 2],
				[float('nan'), 4]])


mean_T = f.calc_mean_T(T)

print(mean_T)