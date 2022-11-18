# by Stefan Vocht
#
# this file contains unittests for state_space_restriction.pyx
#

import unittest
import numpy as np
from mhn.ssr import state_space_restriction
from mhn.ssr.state_storage import StateStorage
from mhn.original import Likelihood, UtilityFunctions, ModelConstruction

np.random.seed(0)  # set random seed for reproducibility


class TestCythonGradient(unittest.TestCase):
    """
    Tests for the function cython_gradient_and_score
    """
    def test_comparison_with_numerical_gradient(self):
        """
        Computes the gradient numerically and compares it with the analytical one
        """
        n = 5
        sample_num = 30
        h = 1e-10
        theta = ModelConstruction.random_theta(n)
        random_sample = np.random.choice([0, 1], (sample_num, n), p=[0.5, 0.5])
        pD = UtilityFunctions.data_to_pD(random_sample)                             # compute the data distribution
        original_score = Likelihood.score(theta, pD)
        numerical_gradient = np.empty((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                theta_copy = theta.copy()
                theta_copy[i, j] += h
                new_score = Likelihood.score(theta_copy, pD)
                numerical_gradient[i, j] = (new_score - original_score) / h

        analytic_gradient, _ = state_space_restriction.cython_gradient_and_score(theta, StateStorage(random_sample))
        np.testing.assert_array_equal(np.around(numerical_gradient, decimals=3), np.around(analytic_gradient, decimals=3))

    def test_gene_position_permutation(self):
        """
        Permutation of the position of genes in the data should lead to a permutation in the gradient
        """
        self.assertEqual(True, False)  # add assertion here

    def test_sample_position_permutation(self):
        """
        Permutation of the cancer samples should not change the gradient
        """
        self.assertEqual(True, False)  # add assertion here


class TestCudaGradient(unittest.TestCase):
    """
    Tests for the function gradient_and_score_with_cuda
    """
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
