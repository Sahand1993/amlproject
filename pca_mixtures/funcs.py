import numpy as np

class PCAModel(object):

	def __init__(self, data):
		self.data = data
		self.data_dimensions = data[0].shape[0]


	def calc_responsibility(prob_t_given_i, prob_t, pi_i):
		"""
		Calculates and returns R_{ni} = p(t | i) * pi_i / p(t) in Tipping & Bishop 2006 (equation 21).
		t is a vector, R_{ni} and pi_i are scalars.

		:param prob_t_given_i: p(t | i)
		:type prob_t_given_i: float

		:param prob_t: p(t)
		:type prob_t: float

		:param pi_i: pi_i
		:type pi_i: float

		:return: R_{ni}
		"""
		return prob_t_given_i * pi_i / prob_t


	def calc_pi(self, responsibilities):
		"""
		Calculates pi_i = \sum_{i=1}^N R_{ni} in Tipping & Bishop 2006 (equation 22).

		:param responsibilities: R_{ni} for all n, fixed i.
		:type responsibilities: array_like

		:return: pi_i
		"""
		return np.mean(responsibilities)


	def calc_mu(self, responsibilities, data):
		"""
		Calculates the mean of a certain PCA model mu_i = \sum_{i=1}^N R_{ni} * t_n / ( \sum_{i=1}^N R_{ni} )
		in Tipping & Bishop 2006 (equation 23).


		:param responsibilities: R_{ni} for all n, fixed i
		:type responsibilities: array_like

		:param data: data vectors t_n, for all n.
		:type data: array_like

		:return:
		"""
		return np.sum(responsibilities * data) / np.sum(responsibilities)


	def calc_sample_cov_matrix(self, responsibilities, mu, pi):
		"""
		Calculates S = 1 / (pi * N) * \sum_{i=1}^N R_{ni} (t_n - mu)(t_n - mu)^T in Tipping & Bishop 2006 (equation 84).
		:param responsibilities: R_{ni}
		:type responsibilities: array_like

		:param mu:
		:type mu: array_like

		:param pi:
		:type pi: float

		:return: S
		"""
		sample_cov_matrix = np.zeros([self.data_dimensions, self.data_dimensions])
		for responsibility, t in zip(responsibilities, self.data):
			sample_cov_matrix += responsibility * np.matmul(t - mu, (t - mu).T) # TODO: Check that orientations are correct
		sample_cov_matrix /= pi * len(self.data)

		return sample_cov_matrix


	def calc_M_inv(self, W, sigma):
		"""
		Calculates M^{-1} = (\sigma^2 * I + W^T * W)^{-1} in Tipping & Bishop 2006 (equation 9)

		:param W: W matrix
		:type W: array_like

		:return: M^{-1}
		"""
		M = sigma**2 + np.matmul(W.T, W)
		return np.linalg.inv(M)
