import unittest

from mhn.original import ModelConstruction
from mhn.ssr import matrix_exponential, state_storage

import numpy as np
import scipy


class TestMatrixExponential(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(0)

    def test_correctness_expm(self):
        n = 6
        theta = ModelConstruction.random_theta(n)
        q = ModelConstruction.build_q(theta)
        all_mutated_data = np.ones((1, n), dtype=np.int32)
        b = np.random.random(2**n)

        for t in np.arange(0, 6, 0.6):
            container = state_storage.StateAgeStorage(all_mutated_data, np.array([t]))
            result1 = scipy.linalg.expm(t * q).dot(b)
            result2 = matrix_exponential.restricted_expm(theta, b, container, 1e-8)
            np.testing.assert_array_equal(np.around(result1, decimals=3), np.around(result2, decimals=3))

    def test_calc_gamma_numerically(self):
        n = 6
        theta = ModelConstruction.random_theta(n)
        np_data_matrix = np.zeros((1, 6), dtype=np.int32)
        np_data_matrix[:, 2:4] = 1
        container = state_storage.StateAgeStorage(np_data_matrix, np.array([0.5]))
        h = 1e-10
        for i in range(n):
            for j in range(n):
                gamma1, dgamma = matrix_exponential.py_calc_gamma(theta, container, i, j)
                theta_copy = theta.copy()
                theta_copy[i, j] += h
                gamma2, _ = matrix_exponential.py_calc_gamma(theta_copy, container, i, j)
                numerical_deriv = (gamma2 - gamma1) / h
                self.assertAlmostEqual(numerical_deriv, dgamma, 3)

    def test_score_and_gradient_numerically(self):
        n = 6
        sample_num = 5
        assert sample_num < n, "This test needs the sample number to be smaller than n to work because of how the " \
                               "data is constructed "
        theta = ModelConstruction.random_theta(n)
        np_data_matrix = np.zeros((sample_num, n), dtype=np.int32)
        for i in range(sample_num):
            np_data_matrix[i, i:] = 1

        ages = np.linspace(0, 6, sample_num)
        states_and_ages = state_storage.StateAgeStorage(np_data_matrix, ages)
        h = 1e-10

        gradient, old_score = matrix_exponential.cython_gradient_and_score(theta, states_and_ages, 1e-6)
        for i in range(n):
            for j in range(n):
                theta_copy = theta.copy()
                theta_copy[i, j] += h
                _, new_score = matrix_exponential.cython_gradient_and_score(theta_copy, states_and_ages, 1e-6)
                numerical_gradient = (new_score - old_score) / h
                self.assertAlmostEqual(numerical_gradient, gradient[i, j], 3)

    def test_sort_by_age(self):
        """
        StateAgeStorage should automatically sort the samples according to their age.
        """
        n = 6
        sample_num = 5
        assert sample_num < n, "This test needs the sample number to be smaller than n to work because of how the " \
                               "data is constructed "
        theta = ModelConstruction.random_theta(n)
        np_data_matrix = np.zeros((sample_num, n), dtype=np.int32)
        for i in range(sample_num):
            np_data_matrix[i, i:] = 1
        ages = np.linspace(0, 6, sample_num)

        states_and_ages = state_storage.StateAgeStorage(np_data_matrix, ages)
        grad1, score1 = matrix_exponential.cython_gradient_and_score(theta, states_and_ages, 1e-6)

        permutation = np.random.permutation(sample_num)
        np_data_matrix = np_data_matrix[permutation]
        ages = ages[permutation]
        states_and_ages = state_storage.StateAgeStorage(np_data_matrix, ages)
        grad2, score2 = matrix_exponential.cython_gradient_and_score(theta, states_and_ages, 1e-6)

        self.assertAlmostEqual(score1, score2, 8)
        np.testing.assert_array_equal(np.around(grad1, decimals=5), np.around(grad2, decimals=5))





if __name__ == '__main__':
    unittest.main()
