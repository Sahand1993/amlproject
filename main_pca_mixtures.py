import numpy as np
from pca_mixtures.funcs import PCAMixture
import matplotlib.pyplot as plt

data = np.loadtxt("dataset/tobomovirus.txt")
#print(data)
mixture = PCAMixture(data, 1, 2)
mixture.fit()


class Plotter(object):
	def __init__(self, mixture):
		self.mixture = mixture

	def plot_all_data(self, skip_improbable):
		for model in self.mixture.models:
			self._plot_data(data, model, skip_improbable)

	def _plot_data(self, data, model, skip_improbable):
		latent_positions = model.posterior(data)
		plt.figure()
		X = latent_positions[:, 0]
		Y = latent_positions[:, 1]
		plt.scatter(X, Y)
		plt.figure()
		nums = [str(i) for i in range(len(data))]
		for x, y, txt in zip(X, Y, nums):
			plt.text(x, y, txt)


plotter = Plotter(mixture)
plotter.plot_all_data(skip_improbable = False)
plt.show()
