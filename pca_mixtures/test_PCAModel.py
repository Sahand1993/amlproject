from unittest import TestCase
from pca_mixtures.funcs import PCAModel, PCAMixture
import numpy as np

class TestPCAModel(TestCase):
	def test_calc_resp(self):
		data = np.random.multivariate_normal([0, 0], [[2, 1],[1, 1]], 2)
		mixture = PCAMixture(data, 1, 2)
		model = PCAModel(mixture, 2)
		self.assertEqual(model.calc_resp(0.5, 2, 0.5), 0.125)

	def test_calc_mixing_coeff(self):
		data = np.random.multivariate_normal([0, 0], [[2, 1],[1, 1]], 3)
		mixture = PCAMixture(data, 1, 2)
		model = PCAModel(mixture, 2)

		resps = np.array([1, 2, 3])
		self.assertEqual(model.calc_mixing_coeff(resps), 2)

	def test_set_mean(self):
		data = np.array([
			[1,3],
			[1,6]
		])
		mixture = PCAMixture(data, 1, 2)
		model = PCAModel(mixture, 2)

		model_resps = np.array([1,2])
		model.set_mean(model_resps)
		self.assertTrue( np.array_equal(model.mean, np.array([1, 5])) )

	def test_set_S(self):
		data = np.array([
			[1, 3],
			[1, 6]
		])
		mixture = PCAMixture(data, 1, 2)
		model = PCAModel(mixture, 2)
		model_resps = np.array([1, 2])
		mixture.resps = [model_resps]
		model.set_mean(model_resps)
		mixture.mixing_coeffs = [3 / 2]

		mixing_coeff = mixture.mixing_coeffs[0]
		model_resps = mixture.resps[0]
		model.set_S(mixing_coeff, model_resps)

		self.assertTrue(np.array_equal(model.S, np.array([
			[0, 0],
			[0, 2]
		])))

	def test_update_M_inv(self):
		data = np.array([
			[1, 3],
			[1, 6]
		])
		mixture = PCAMixture(data, 1, 2)
		model = PCAModel(mixture, 2)
		model.var = 3
		model.W = np.array([
			[1, 2],
			[2, 1],
			[1, 3]
		])
		model.update_M_inv()
		M_inv = np.array([
			[17, -7],
			[-7, 9]
		]) / 104
		res = np.isclose(model.M_inv, M_inv)
		self.assertTrue(res.all())

	def test_set_W(self):
		S = np.array([
			[0, 0],
			[0, 2]
		])
		M_inv = np.array([
			[17, -7],
			[-7, 9]
		]) / 104
		W = np.array([
			[1, 1],
			[3, 2]
		])

		data = np.array([
			[1, 3],
			[1, 6]
		])
		mixture = PCAMixture(data, 1, 2)
		model = PCAModel(mixture, 2)
		model.var = 3
		model.S = S
		model.M_inv = M_inv
		model.W = W

		model.set_W()
		new_W = model.W

		#print(new_W)

		res = np.isclose(new_W, np.array([
			[0, 0],
			[104 / 87, 208 / 261]
		]))
		self.assertTrue(res.all())

	def test_set_var(self):
		S = np.array([
			[0, 0],
			[0, 2]
		])

		data = np.array([
			[1, 3],
			[1, 6]
		])
		mixture = PCAMixture(data, 1, 2)
		model = PCAModel(mixture, 2)
		model.S = S
		W_old = np.array([
			[1, 1],
			[3, 2]
		])
		M_inv_old = np.array([
			[17, -7],
			[-7, 9]
		]) / 104
		model.W = np.array([
			[0, 0],
			[104 / 87, 208 / 261]
		])

		model.set_var(W_old, M_inv_old)

		self.assertAlmostEqual(model.var, 1.02298850575)

	def test_set_C_inv(self):
		self.fail()

	def test_set_C_det(self):
		self.fail()

	def test_calc_prob_data(self):
		self.fail()
