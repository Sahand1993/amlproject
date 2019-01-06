import numpy as np

def normal_pdf(x, mean, cov_det, cov_inv):
	exp = np.exp(np.matmul(np.matmul((x - mean).T, cov_inv), x - mean))
	out = exp / (2 * np.pi)**(len(x) / 2) / np.sqrt(cov_det)
	return out


class PCAModel(object):

	def __init__(self, mixture):
		"""
		A single PCA model to be used in a PCA mixture


		:param mixture: The mixture object that the model is part of. Used to get data dimensionality.
		"""
		self.mixture = mixture

	def calc_resp(self, prob_t_given_i, prob_t, pi_i):
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

	def calc_mixing_coeff(self, resps):
		"""
		Calculates pi_i = \sum_{i=1}^N R_{ni} in Tipping & Bishop 2006 (equation 22).

		:param resps: R_{ni} for all n, fixed i.
		:type resps: array_like

		:return: pi_i
		"""
		return np.mean(resps)

	def set_mean(self, resps, data):
		"""
		Calculates the mean of a certain PCA model mu_i = \sum_{i=1}^N R_{ni} * t_n / ( \sum_{i=1}^N R_{ni} )
		in Tipping & Bishop 2006 (equation 23).


		:param resps: R_{ni} for all n, fixed i
		:type resps: array_like

		:param data: data vectors t_n, for all n.
		:type data: array_like

		:return:
		"""
		self.mean = np.sum(resps * data) / np.sum(resps)
		return self.mean

	def calc_sample_cov_matrix(self, resps, mu, pi):
		"""
		Calculates S = 1 / (pi * N) * \sum_{i=1}^N R_{ni} (t_n - mu)(t_n - mu)^T in Tipping & Bishop 2006 (equation 84).
		:param resps: R_{ni}
		:type resps: array_like

		:param mu:
		:type mu: array_like

		:param pi:
		:type pi: float

		:return: S
		"""
		sample_cov_matrix = np.zeros([self.mixture.data_dimensions, self.mixture.data_dimensions])
		for resp, t in zip(resps, self.mixture.data):
			sample_cov_matrix += resp * np.matmul(t - mu, (t - mu).T) # TODO: Check that orientations are correct
		sample_cov_matrix /= pi * len(self.mixture.data)

		return sample_cov_matrix

	def calc_M_inv(self, W, var):
		"""
		Calculates M^{-1} = (\sigma^2 * I + W^T * W)^{-1} in Tipping & Bishop 2006 (equation 9)

		:param W: W matrix
		:type W: array_like

		:return: M^{-1}
		"""
		M = var + np.matmul(W.T, W)
		return np.linalg.inv(M)

	def calc_W(self, S, W, M_inv, var):
		"""
		Calculates W = S * W (sigma^2 * I + M^{-1} * W^T * S * W)^{-1} in Tipping & Bishop (equation 82)

		:param S:
		:type S: array_like

		:param W:
		:type W: array_like

		:param M_inv: M^{-1}
		:type M_inv: array_like

		:param var: sigma^2
		:type var: float

		:return: W
		"""
		left = np.matmul(S, W)

		right = np.matmul(M_inv, W.T)
		right = np.matmul(right, S)
		right = np.matmul(right, W)
		len_diag = right.shape[0]
		right += np.diag(np.ones(len_diag)) * var

		W = np.matmul(left, right)
		return W

	def calc_sigma(self, S, W_old, W_new, M_inv):
		"""
		Calculates sigma^2 = 1 / d * Tr[S - S * W * M^{-1} * W^T] in Tipping & Bishop (equation 83)

		:param S:
		:type S: array_like

		:param W_old:
		:type W_old: array_like

		:param W_new:
		:type W_new: array_like

		:param M_inv:
		:type M_inv: array_like

		:return:
		"""
		matrix_prod = np.matmul(S, W_old)
		matrix_prod = np.matmul(matrix_prod, M_inv)
		matrix_prod = np.matmul(matrix_prod, W_new)
		var = np.sum(np.diag(S - matrix_prod)) / self.mixture.data_dimensions
		return var

	def set_C_inv(self, W, M_inv, var):
		C_inv = np.diag(np.ones(self.mixture.data_dimensions))
		C_inv -= np.matmul(np.matmul(W, M_inv), W.T)
		C_inv /= var
		self.C_inv = C_inv
		return C_inv

	def set_C_det(self, C_inv):
		C = np.linalg.inv(C_inv)
		self.C_det = np.linalg.det(C)
		return self.C_det

	def calc_prob_data(self, data_vector):
		"""
		Computes p(t) = (2 * pi)^{-d/2} * |C_i|^{-1/2} * exp{-1/2 * (t - mu_i)^T * C_i^{-1} * (t - mu_i) }
		in Tipping & Bishop (equation 6)

		Parameters
		--------------------
		:param data_vector: the data vector t
		:type data_vector: ndarray

		:return: p(t)
		:type return: float
		"""
		return normal_pdf(data_vector, self.mean, self.C_det, self.C_inv)


class PCAMixture(object):
	def __init__(self, data, no_models):
		"""

		:param data:
		:param no_models:
		"""
		self.no_models = no_models

		self.models = []
		self.mixing_coeffs = []
		for i in range(no_models):
			self.models.append(PCAModel(self))
			self.mixing_coeffs.append(None)

		self.data = data
		self.data_dimensions = data[0].shape[0]

	def calc_prob_data(self, data_vector, model = None):
		"""
		Computes the probability of a single data vector t.

		With i argument:
		--------------------
		p(t | i) = (2 * pi)^{-d/2} * |C_i|^{-1/2} * exp{-1/2 * (t - mu_i)^T * C_i^{-1} * (t - mu_i) }
		in Tipping & Bishop (equation 6)


		Without i argument:
		--------------------
		p(t) = \sum_{i=1}^M pi_i * p(t | i)

		:param data_vector:
		:type data_vector: ndarray

		:param model: the model
		:type model_idx: PCAModel
		:return: p(t | i) or p(t)
		:type return: float
		"""
		if model:
			return model.calc_prob_data(data_vector)

		probability = 0
		for model, mixing_coeff in zip(self.models, self.mixing_coeffs):
			probability += mixing_coeff * model.calc_prob_data(data_vector)

		return probability

	def set_mixing_coefficients(self):
		"""
		Tipping & Bishop Equation 22 for each model
		:return:
		"""
		resps = self.get_resps()
		for model_idx, (model, model_resps) in enumerate(zip(self.models, resps)):
			self.mixing_coeffs[model_idx] = model.calc_mixing_coeff(resps)

	def get_resps(self):
		"""
		Tipping & Bishop equation 21 for each model
		:return:
		"""
		resps = []
		for model in self.models:
			resps.append(self._get_model_resps(model))
		return resps

	def _get_model_resps(self, model):
		"""
		Tipping & Bishop equation 21 for model

		:param model:
		:type model: PCAModel
		"""
		model_resps = []
		for data_vector in self.data:
			probability_model = self.calc_prob_data(data_vector, model)
			total_probability = self.calc_prob_data(data_vector)
			model_idx = self.models.index(model) # Find model index
			resp = model.calc_resp(probability_model, total_probability, self.mixing_coeffs[model_idx])
			model_resps.append(resp)
		return model_resps
