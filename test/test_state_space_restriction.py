# by Stefan Vocht
#
# this file contains unittests for state_space_restriction.pyx
#

import unittest
import numpy as np
from mhn.ssr import state_space_restriction
from mhn.ssr.state_storage import StateStorage
from mhn.original import Likelihood, UtilityFunctions, ModelConstruction


class TestCythonGradient(unittest.TestCase):
    """
    Tests for the function cython_gradient_and_score
    """
    def setUp(self) -> None:
        """
        Preparation for each test
        """
        np.random.seed(0)  # set random seed for reproducibility

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
        # compute the gradient numerically
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
        Permutation of the position of genes in the data should lead to a permutation in the gradient,
        but should not change the score
        """
        n = 40  # make n > 32 to make sure that the logic of the "State" stuct in the C implementation works as expected
        sample_num = 30
        theta = ModelConstruction.random_theta(n)
        random_sample = np.random.choice([0, 1], (sample_num, n), p=[0.8, 0.2])
        # make sure that there are mutations in two different "parts" of the "State" C struct
        random_sample[:, 0] = 1
        random_sample[:, -1] = 1
        # compute original gradient and score
        gradient1, score1 = state_space_restriction.cython_gradient_and_score(theta, StateStorage(random_sample))
        # permute the sample and theta, compute gradient and score and reverse the permutation
        permutation = np.random.permutation(n)
        reverse = np.empty(n, int)
        reverse[permutation] = np.arange(n)
        permutation_sample = random_sample[:, permutation]
        permutation_theta = theta[permutation][:, permutation]
        gradient2, score2 = state_space_restriction.cython_gradient_and_score(permutation_theta, StateStorage(permutation_sample))
        reversed_gradient = gradient2[reverse][:, reverse]
        np.testing.assert_array_equal(permutation_sample[:, reverse], random_sample)
        # compare gradients and scores
        np.testing.assert_array_equal(np.around(gradient1, decimals=8), np.around(reversed_gradient, decimals=8))
        self.assertEqual(score1, score2)

    def test_sample_position_permutation(self):
        """
        Permutation of the cancer samples should not change the gradient
        """
        self.assertEqual(True, False)  # add assertion here


class TestCudaGradient(unittest.TestCase):
    """
    Tests for the function gradient_and_score_with_cuda
    """
    def setUp(self) -> None:
        """
        Preparation for each test
        """
        np.random.seed(0)  # set random seed for reproducibility
        if state_space_restriction.cuda_available() != state_space_restriction.CUDA_AVAILABLE:
            self.skipTest("CUDA not available for testing")

    def test_compare_with_cython(self):
        n = 40
        sample_num = 30
        theta = ModelConstruction.random_theta(n)
        random_sample = np.random.choice([0, 1], (sample_num, n), p=[0.7, 0.3])
        # make sure that there are mutations in two different "parts" of the "State" C struct
        random_sample[:, 1] = 1
        random_sample[:, -3] = 1
        state_storage = StateStorage(random_sample)
        gradient1, score1 = state_space_restriction.cython_gradient_and_score(theta, state_storage)
        gradient2, score2 = state_space_restriction.gradient_and_score_with_cuda(theta, state_storage)
        self.assertEqual(round(score1, 8), round(score2, 8))
        np.testing.assert_array_equal(np.around(gradient1, decimals=8), np.around(gradient2, decimals=8))

    def test_gene_position_permutation(self):
        """
        Permutation of the position of genes in the data should lead to a permutation in the gradient,
        but should not change the score
        """
        n = 40  # make n > 32 to make sure that the logic of the "State" stuct in the C implementation works as expected
        sample_num = 30
        theta = ModelConstruction.random_theta(n)
        random_sample = np.random.choice([0, 1], (sample_num, n), p=[0.8, 0.2])
        # make sure that there are mutations in two different "parts" of the "State" C struct
        random_sample[:, 1] = 1
        random_sample[:, -3] = 1
        # compute original gradient and score
        gradient1, score1 = state_space_restriction.gradient_and_score_with_cuda(theta, StateStorage(random_sample))
        # permute the sample and theta, compute gradient and score and reverse the permutation
        permutation = np.random.permutation(n)
        reverse = np.empty(n, int)
        reverse[permutation] = np.arange(n)
        permutation_sample = random_sample[:, permutation]
        permutation_theta = theta[permutation][:, permutation]
        gradient2, score2 = state_space_restriction.gradient_and_score_with_cuda(permutation_theta.copy(), StateStorage(permutation_sample))
        reversed_gradient = gradient2[reverse][:, reverse]
        np.testing.assert_array_equal(permutation_sample[:, reverse], random_sample)
        # compare gradients and scores
        np.testing.assert_array_equal(np.around(gradient1, decimals=8), np.around(reversed_gradient, decimals=8))
        self.assertEqual(score1, score2)


if __name__ == '__main__':
    unittest.main()
