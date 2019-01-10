import numpy as np
from pca_mixtures.funcs import PCAMixture

data = np.loadtxt("../dataset/tobomovirus.txt")
#print(data)
mixture = PCAMixture(data, 3, 2)
mixture.fit()