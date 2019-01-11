import numpy as np

from unittest import TestCase
from pca_mixtures.funcs import PCAMixture, PCAModel


class TestPCAMixture(TestCase):
	def test_calc_prob_data(self):
		data = np.array([
			[1, 3],
			[1, 6]
		])
		mean = np.array([1, 5])
		C_det = 0.75
		C_inv = np.array([
			[4, -2],
			[-2, 4]
		]) / 3
		mixture = PCAMixture(data, 1, 2)
		model = PCAModel(mixture, 2)
		mixture.models = [model]
		model.mean = mean
		model.C_det = C_det
		model.C_inv = C_inv

		prob = mixture.calc_prob_data(data[0])
		real_prob = 0.0127694115
		self.assertAlmostEqual(prob, real_prob)

		data = np.array([
			[1, 3],
			[1, 6]
		])
		mean1 = np.array([1, 5])
		C_det1 = 0.75
		C_inv1 = np.array([
			[4, -2],
			[-2, 4]
		]) / 3
		mean2 = np.array([2,2])
		C_det2 = 1
		C_inv2 = np.array([
			[1, 0],
			[0, 1]
		])
		mixture = PCAMixture(data, 2, 2)
		model1 = PCAModel(mixture, 2)
		model2 = PCAModel(mixture, 2)
		mixture.models = [model1, model2]
		model1.mean = mean1
		model1.C_det = C_det1
		model1.C_inv = C_inv1
		model2.mean = mean2
		model2.C_det = C_det2
		model2.C_inv = C_inv2

		prob = mixture.calc_prob_data(data[0])
		real_prob = 0.0356596215
		self.assertAlmostEqual(real_prob, prob)

	def test_set_resps(self):
		data = np.array([
			[1, 3],
			[1, 6]
		])
		mean1 = np.array([1, 5])
		C_det1 = 0.75
		C_inv1 = np.array([
			[4, -2],
			[-2, 4]
		]) / 3
		mean2 = np.array([2, 2])
		C_det2 = 1
		C_inv2 = np.array([
			[1, 0],
			[0, 1]
		])
		mixture = PCAMixture(data, 2, 2)
		model1 = PCAModel(mixture, 2)
		model2 = PCAModel(mixture, 2)
		mixture.models = [model1, model2]
		model1.mean = mean1
		model1.C_det = C_det1
		model1.C_inv = C_inv1
		model2.mean = mean2
		model2.C_det = C_det2
		model2.C_inv = C_inv2

		mixture.set_resps()

		resps1 = mixture.resps[0]
		resps2 = mixture.resps[1]

		self.assertAlmostEqual(resps1[0], 0.1790458078)
		self.assertAlmostEqual(resps1[1], 0.99965691)

		self.assertAlmostEqual(resps2[0], 0.8209541918)
		self.assertAlmostEqual(resps2[1], 3.43090081e-04)

	def test_get_model_resps(self):
		# Test 1
		# ----------------------------------
		data = np.array([
			[1, 3],
			[1, 6]
		])
		mixture = PCAMixture(data, 1, 2)
		model = PCAModel(mixture, 2)
		mean = np.array([1, 5])
		C_det = 0.75
		C_inv = np.array([
			[4, -2],
			[-2, 4]
		]) / 3
		mixture = PCAMixture(data, 1, 2)
		model = PCAModel(mixture, 2)
		mixture.models = [model]
		model.mean = mean
		model.C_det = C_det
		model.C_inv = C_inv

		resps = mixture.get_model_resps(model)
		res = np.array_equal(resps, np.array([1,1]))

		self.assertTrue(res)

		# Test 2
		# ---------------------------------
		data = np.array([
			[1, 3],
			[1, 6]
		])
		mean1 = np.array([1, 5])
		C_det1 = 0.75
		C_inv1 = np.array([
			[4, -2],
			[-2, 4]
		]) / 3
		mean2 = np.array([2, 2])
		C_det2 = 1
		C_inv2 = np.array([
			[1, 0],
			[0, 1]
		])
		mixture = PCAMixture(data, 2, 2)
		model1 = PCAModel(mixture, 2)
		model2 = PCAModel(mixture, 2)
		mixture.models = [model1, model2]
		model1.mean = mean1
		model1.C_det = C_det1
		model1.C_inv = C_inv1
		model2.mean = mean2
		model2.C_det = C_det2
		model2.C_inv = C_inv2

		resps1 = mixture.get_model_resps(model1)
		self.assertAlmostEqual(resps1[0], 0.1790458078)
		self.assertAlmostEqual(resps1[1], 0.99965691)

		resps2 = mixture.get_model_resps(model2)
		self.assertAlmostEqual(resps2[0], 0.8209541918)
		self.assertAlmostEqual(resps2[1], 3.43090081e-04)

	def test_set_mixing_coefficients(self):
		data = np.array([
			[1, 3],
			[1, 6]
		])
		mean1 = np.array([1, 5])
		C_det1 = 0.75
		C_inv1 = np.array([
			[4, -2],
			[-2, 4]
		]) / 3
		mean2 = np.array([2, 2])
		C_det2 = 1
		C_inv2 = np.array([
			[1, 0],
			[0, 1]
		])
		mixture = PCAMixture(data, 2, 2)
		model1 = PCAModel(mixture, 2)
		model2 = PCAModel(mixture, 2)
		mixture.models = [model1, model2]
		model1.mean = mean1
		model1.C_det = C_det1
		model1.C_inv = C_inv1
		model2.mean = mean2
		model2.C_det = C_det2
		model2.C_inv = C_inv2

		mixture.set_resps()

		mixture.set_mixing_coefficients()

		mixing_coeffs = mixture.mixing_coeffs
		self.assertAlmostEqual(mixing_coeffs[0], 0.5893513589)
		self.assertAlmostEqual(mixing_coeffs[1], 0.41064864094)

