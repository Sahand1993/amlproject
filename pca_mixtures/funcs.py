import numpy as np
from pca_mixtures.constants import *

def normal_pdf(x, mean, cov_det, cov_inv):
	diff = x - mean
	exponent = np.matmul(diff, cov_inv)
	exponent = -np.dot(exponent, diff) / 2
	exp = np.exp(exponent)
	normalizer = (2 * np.pi)**(len(x) / 2) * np.sqrt(cov_det)
	out = exp / normalizer
	return out

def normal_logpdf(x, mean, cov_det, cov_inv):
	diff = x - mean
	dim = len(x)

	exponent = np.matmul(diff, cov_inv)
	exponent = np.dot(exponent, diff) / 2

	out = - np.log(exponent) - np.log(cov_det) - dim * np.log(2 * np.pi)
	out /= 2
	return out

class PCAModel(object):

	def __init__(self, mixture, latent_dim):
		"""
		A single PCA model to be used in a PCA mixture


		:param mixture: The mixture object that the model is part of. Used to get data dimensionality.
		"""
		self.mixture = mixture

		#self.resps = None
		self.mean = np.zeros(mixture.data_dimensions)
		shape_W = (mixture.data_dimensions, latent_dim)
		W = []
		for i in range(latent_dim):
			col = np.random.multivariate_normal(np.zeros(mixture.data_dimensions), np.eye(mixture.data_dimensions))
			W.append(col)
		self.W = np.array(W).T
		self.M_inv = None
		self.var = INITIAL_VAR
		self.S = None
		self.C_det = None
		self.C_inv = None

	def calc_resp(self, prob_t_given_i, prob_t, mixing_coeff):
		"""
		Calculates and returns R_{ni} = p(t | i) * pi_i / p(t) in Tipping & Bishop 2006 (equation 21).
		t is a vector, R_{ni} and pi_i are scalars.

		:param prob_t_given_i: p(t | i)
		:type prob_t_given_i: float

		:param prob_t: p(t)
		:type prob_t: float

		:param mixing_coeff: pi_i
		:type mixing_coeff: float

		:return: R_{ni}
		"""
		return prob_t_given_i * mixing_coeff / prob_t

	def calc_mixing_coeff(self, resps):
		"""
		Calculates pi_i = \sum_{i=1}^N R_{ni} in Tipping & Bishop 2006 (equation 22).

		:param resps: R_{ni} for all n, fixed i.
		:type resps: array_like

		:return: pi_i
		"""
		return np.mean(resps)

	def set_mean(self, resps):
		"""
		Calculates the mean of a certain PCA model mu_i = \sum_{i=1}^N R_{ni} * t_n / ( \sum_{i=1}^N R_{ni} )
		in Tipping & Bishop 2006 (equation 23).


		:param resps: R_{ni} for all n, fixed i
		:type resps: array_like

		:param data: data vectors t_n, for all n.
		:type data: array_like

		:return:
		"""
		self.mean = np.matmul(resps, self.mixture.data) / np.sum(resps)

	def set_S(self, mixing_coeff, resps):
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
			diff = t - self.mean
			prod = np.outer(diff, diff)
			sample_cov_matrix += resp * prod
		sample_cov_matrix /= mixing_coeff * len(self.mixture.data)

		self.S = sample_cov_matrix

	def update_M_inv(self):
		"""
		Calculates M^{-1} = (\sigma^2 * I + W^T * W)^{-1} in Tipping & Bishop 2006 (equation 9)

		:param W: W matrix
		:type W: array_like

		:return: M^{-1}
		"""
		M = self.var * np.eye(self.W.shape[1]) + np.matmul(self.W.T, self.W)
		self.M_inv = np.linalg.inv(M)

	def set_W(self):
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
		left = np.matmul(self.S, self.W)

		right = np.matmul(self.M_inv, self.W.T)
		right = np.matmul(right, self.S)
		right = np.matmul(right, self.W)
		len_diag = right.shape[0]
		right += np.eye(len_diag) * self.var
		right = np.linalg.inv(right)

		W = np.matmul(left, right)
		self.W = W

	def set_var(self, W_old, M_inv_old):
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
		matrix_prod = np.matmul(self.S, W_old)
		matrix_prod = np.matmul(matrix_prod, M_inv_old)
		matrix_prod = np.matmul(matrix_prod, self.W.T)
		self.var = np.trace(self.S - matrix_prod) / self.mixture.data_dimensions

	def set_C_inv(self):
		"""
		Computes C^{-1} = { I - W * M^{-1} * W^T } / sigma^2 (last paragraph of Tipping & Bishop 2006.
		:param W:
		:param M_inv:
		:param var:
		:return:
		"""
		C_inv = np.diag(np.ones(self.mixture.data_dimensions))
		C_inv -= np.matmul(np.matmul(self.W, self.M_inv), self.W.T)
		C_inv /= self.var
		self.C_inv = C_inv

	def set_C_det(self):
		"""
		Computes and sets the determinant of C given the inverse of C.
		:param C_inv:
		:return:
		"""
		identity = self.var * np.eye(self.W.shape[0])
		prod = np.matmul(self.W, self.W.T)
		C =  identity + prod
		self.C_det = np.linalg.det(C)

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
	def get_model_power(self, x):
		"""
		Gets the exponent in the normal distribution pdf for vector x
		:param x:
		:return:
		"""
		diff = x - self.mean
		power = np.matmul(diff, self.C_inv)
		power = - np.dot(power, diff) / 2
		return power

	def logpdf(self, x):
		return normal_logpdf(x, self.mean, self.C_det, self.C_inv)

	def posterior(self, data):
		diff = data - self.mean
		posteriors = np.dot(diff, np.matmul(self.M_inv, self.W.T).T)
		return posteriors

class PCAMixture(object):
	def __init__(self, data, no_models, latent_dim):
		"""

		:param data:
		:param no_models:
		"""
		self.data = data
		self.data_dimensions = data[0].shape[0]

		self.no_models = no_models

		self.models = []
		mixing_coeff = 1 / no_models
		self.mixing_coeffs = [mixing_coeff]*no_models
		for i in range(no_models):
			model = PCAModel(self, latent_dim)
			self.models.append(model)
		self.resps = []

	def calc_prob_data(self, data_vector):
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

		probability = 0
		for model, mixing_coeff in zip(self.models, self.mixing_coeffs):
			probability += mixing_coeff * model.calc_prob_data(data_vector)

		return probability

	def set_resps(self):
		"""
		Tipping & Bishop equation 21 for each model
		:return:
		"""
		resps = []
		for model in self.models:
			resps.append(self.get_model_resps(model))
		self.resps = resps

	def get_model_resps(self, model):
		"""
		Tipping & Bishop equation 21 for model

		:param model:
		:type model: PCAModel
		"""
		model_resps = []
		for data_vector in self.data:
			max_model = self.get_max_model(data_vector)
			left_factor = self.calc_left_frac(data_vector, max_model, model)
			right_factor = self.calc_right_frac(data_vector, max_model)
			resp = left_factor * right_factor
			model_resps.append(resp)
		return np.array(model_resps)

	def set_mixing_coefficients(self):
		"""
		Tipping & Bishop Equation 22 for each model
		:return:
		"""
		for model_idx, (model, model_resps) in enumerate(zip(self.models, self.resps)):
			self.mixing_coeffs[model_idx] = model.calc_mixing_coeff(model_resps)

	def set_means_of_models(self):
		for model, model_resps in zip(self.models, self.resps):
			model.set_mean(model_resps)

	def set_W_of_models(self):
		for model in self.models:
			model.set_W()

	def set_var_of_models(self, old_Ws, old_M_invs):
		for model, W_old, M_inv_old in zip(self.models, old_Ws, old_M_invs):
			model.set_var(W_old, M_inv_old)

	def get_old_Ws(self):
		out = []
		for model in self.models:
			out.append(model.W)
		return out

	def get_old_M_invs(self):
		out = []
		for model in self.models:
			out.append(model.M_inv)
		return out

	def set_C_det_of_models(self):
		for model in self.models:
			model.set_C_det()

	def set_C_inv_of_models(self):
		for model in self.models:
			model.set_C_inv()

	def set_M_inv_of_models(self):
		for model in self.models:
			model.update_M_inv()

	def set_S_of_models(self):
		for model, mixing_coeff, model_resps in zip(self.models, self.mixing_coeffs, self.resps):
			model.set_S(mixing_coeff, model_resps)

	def calc_right_frac(self, x, max_model):
		sum = 0
		for model in self.models:
			temp = self.calc_left_frac(x, max_model, model)
			sum += temp
		out = 1 / sum
		return out

	def calc_left_frac(self, x, max_model, model):
		"""
		Calculates p(x | model) * pi_{model} / p(x | max_model)
		:param max_model:
		:param model:
		:return:
		"""
		model_idx = self.models.index(model)
		res = self.mixing_coeffs[model_idx]
		res = res * np.sqrt(max_model.C_det / model.C_det)

		#model_power = -np.matmul(model_diff, model.C_inv)
		#model_power = np.dot(model_power, model_diff)

		model_power = model.get_model_power(x)
		max_model_power = max_model.get_model_power(x)

		exp = np.exp(model_power - max_model_power)

		res *= exp

		return res

	def get_max_model(self, x):
		"""
		Return argmax_i[p(x | i)]
		:param x: data vector
		:return: the index of the model with the highest probability
		"""
		curr_max_model = None
		curr_max_likelihood = None
		for model in self.models:
			loglikelihood = model.logpdf(x)
			try:
				if loglikelihood > curr_max_likelihood:
					curr_max_model = model
					curr_max_likelihood = loglikelihood
			except TypeError: # If on the first model
				curr_max_model = model
				curr_max_likelihood = loglikelihood
		return curr_max_model

	def fit(self):
		for i in range(ITERS):
			self.set_M_inv_of_models()
			self.print_M_invs()
			self.set_C_inv_of_models()
			self.print_C_invs()
			self.set_C_det_of_models()
			self.print_C_dets()
			self.set_resps()
			self.print_resps()
			self.set_mixing_coefficients()
			self.set_S_of_models()
			self.set_means_of_models()
			old_M_invs = self.get_old_M_invs()
			old_Ws = self.get_old_Ws()
			self.set_W_of_models()
			self.set_var_of_models(old_Ws, old_M_invs)

	def print_M_invs(self):
		print("\nM_invs:")
		for model in self.models:
			print(model.M_inv)

	def print_C_invs(self):
		print("\nC_invs:")
		for model in self.models:
			print(model.C_inv)

	def print_C_dets(self):
		print("\nC_dets:")
		for model in self.models:
			print(model.C_det)

	def print_resps(self):
		print("\nresps:")
		for model_resps in self.resps:
			print(model_resps)

