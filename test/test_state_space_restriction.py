# by Stefan Vocht
#
# this file contains unittests for likelihood_cmhn.pyx
#

import unittest

import numpy as np

from mhn.full_state_space import (Likelihood, ModelConstruction,
                                  UtilityFunctions)
from mhn.training import likelihood_cmhn, likelihood_omhn
from mhn.training.state_containers import StateContainer


class TestCythonGradient(unittest.TestCase):
    """
    Tests for the function cpu_gradient_and_score
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
        random_sample = np.random.choice([0, 1], (sample_num, n), p=[0.5, 0.5]).astype(np.int32)
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

        analytic_gradient, score = likelihood_cmhn.cpu_gradient_and_score(theta, StateContainer(random_sample))
        self.assertEqual(round(score, 8), round(original_score, 8))
        np.testing.assert_array_equal(np.around(numerical_gradient, decimals=3), np.around(analytic_gradient, decimals=3))

    def test_scores_match_cmhn(self):
        """
        Tests if the score computed by cpu_score() is the same as by cpu_gradient_and_score()
        """
        n = 40  # make n > 32 to make sure that the logic of the State struct in the C implementation works as expected
        sample_num = 30
        theta = ModelConstruction.random_theta(n)
        random_sample = np.random.choice([0, 1], (sample_num, n), p=[0.8, 0.2]).astype(np.int32)
        # make sure that there are mutations in two different "parts" of the "State" C struct
        random_sample[:, 0] = 1
        random_sample[:, -1] = 1
        # compute gradient and score
        gradient, score = likelihood_cmhn.cpu_gradient_and_score(theta, StateContainer(random_sample))
        # compute only score
        score2 = likelihood_cmhn.cpu_score(theta, StateContainer(random_sample))
        self.assertEqual(score, score2)

    def test_scores_match_omhn(self):
        """
        Tests if the score computed by cpu_score() is the same as by cpu_gradient_and_score()
        """
        n = 40  # make n > 32 to make sure that the logic of the State struct in the C implementation works as expected
        sample_num = 30
        theta = ModelConstruction.random_theta(n)
        theta = np.vstack((theta, np.random.random(n)))
        random_sample = np.random.choice([0, 1], (sample_num, n), p=[0.8, 0.2]).astype(np.int32)
        # make sure that there are mutations in two different "parts" of the "State" C struct
        random_sample[:, 0] = 1
        random_sample[:, -1] = 1
        # compute gradient and score
        gradient, score = likelihood_omhn.cpu_gradient_and_score(theta, StateContainer(random_sample))
        # compute only score
        score2 = likelihood_omhn.cpu_score(theta, StateContainer(random_sample))
        self.assertEqual(score, score2)

    def test_gene_position_permutation(self):
        """
        Permutation of the position of genes in the data should lead to a permutation in the gradient,
        but should not change the score
        """
        n = 40  # make n > 32 to make sure that the logic of the State struct in the C implementation works as expected
        sample_num = 30
        theta = ModelConstruction.random_theta(n)
        random_sample = np.random.choice([0, 1], (sample_num, n), p=[0.8, 0.2]).astype(np.int32)
        # make sure that there are mutations in two different "parts" of the "State" C struct
        random_sample[:, 0] = 1
        random_sample[:, -1] = 1
        # compute full_state_space gradient and score
        gradient1, score1 = likelihood_cmhn.cpu_gradient_and_score(theta, StateContainer(random_sample))
        # permute the sample and theta, compute gradient and score and reverse the permutation
        permutation = np.random.permutation(n)
        reverse = np.empty(n, int)
        reverse[permutation] = np.arange(n)
        permutation_sample = random_sample[:, permutation]
        permutation_theta = theta[permutation][:, permutation]
        gradient2, score2 = likelihood_cmhn.cpu_gradient_and_score(permutation_theta, StateContainer(permutation_sample))
        reversed_gradient = gradient2[reverse][:, reverse]
        np.testing.assert_array_equal(permutation_sample[:, reverse], random_sample)
        # compare gradients and scores
        np.testing.assert_array_equal(np.around(gradient1, decimals=8), np.around(reversed_gradient, decimals=8))
        self.assertEqual(score1, score2)


class TestCudaGradient(unittest.TestCase):
    """
    Tests for the function cuda_gradient_and_score
    """
    def setUp(self) -> None:
        """
        Preparation for each test
        """
        np.random.seed(0)  # set random seed for reproducibility
        if likelihood_cmhn.cuda_available() != likelihood_cmhn.CUDA_AVAILABLE:
            self.skipTest("CUDA not available for testing")

    def test_compare_with_cython(self):
        n = 40
        sample_num = 30
        theta = ModelConstruction.random_theta(n)
        random_sample = np.random.choice([0, 1], (sample_num, n), p=[0.7, 0.3]).astype(np.int32)
        # make sure that there are mutations in two different "parts" of the "State" C struct
        random_sample[:, 1] = 1
        random_sample[:, -3] = 1
        # also test with an empty sample
        random_sample[-1, :] = 0
        state_containers = StateContainer(random_sample)
        gradient1, score1 = likelihood_cmhn.cpu_gradient_and_score(theta, state_containers)
        gradient2, score2 = likelihood_cmhn.cuda_gradient_and_score(theta, state_containers)
        self.assertEqual(round(score1, 8), round(score2, 8))
        np.testing.assert_array_equal(np.around(gradient1, decimals=8), np.around(gradient2, decimals=8))

    def test_gene_position_permutation(self):
        """
        Permutation of the position of genes in the data should lead to a permutation in the gradient,
        but should not change the score
        """
        n = 40  # make n > 32 to make sure that the logic of the "State" struct in the C implementation works as expected
        sample_num = 30
        theta = ModelConstruction.random_theta(n)
        random_sample = np.random.choice([0, 1], (sample_num, n), p=[0.8, 0.2]).astype(np.int32)
        # make sure that there are mutations in two different "parts" of the "State" C struct
        random_sample[:, 1] = 1
        random_sample[:, -3] = 1
        # compute full_state_space gradient and score
        gradient1, score1 = likelihood_cmhn.cuda_gradient_and_score(theta, StateContainer(random_sample))
        # permute the sample and theta, compute gradient and score and reverse the permutation
        permutation = np.random.permutation(n)
        reverse = np.empty(n, int)
        reverse[permutation] = np.arange(n)
        permutation_sample = random_sample[:, permutation]
        permutation_theta = theta[permutation][:, permutation]
        gradient2, score2 = likelihood_cmhn.cuda_gradient_and_score(permutation_theta.copy(), StateContainer(permutation_sample))
        reversed_gradient = gradient2[reverse][:, reverse]
        np.testing.assert_array_equal(permutation_sample[:, reverse], random_sample)
        # compare gradients and scores
        np.testing.assert_array_equal(np.around(gradient1, decimals=8), np.around(reversed_gradient, decimals=8))
        self.assertEqual(score1, score2)

    def test_auto_device_picker(self):
        """
        Tests the gradient_and_score() function, which uses both the CPU and the GPU.
        """
        n = 40
        sample_num = 30
        theta = ModelConstruction.random_theta(n)
        random_sample = np.random.choice([0, 1], (sample_num, n), p=[0.7, 0.3]).astype(np.int32)
        # make sure that there are mutations in two different "parts" of the "State" C struct
        random_sample[:, 1] = 1
        random_sample[:, -3] = 1
        # one sample with more than critical_size mutations -> GPU
        random_sample[0, :] = 0
        random_sample[0, :22] = 1
        # one sample with less than critical_size mutations -> CPU
        random_sample[1, :] = 0
        random_sample[1, :5] = 1
        state_containers = StateContainer(random_sample)
        gradient1, score1 = likelihood_cmhn.gradient_and_score(theta, state_containers)
        gradient2, score2 = likelihood_cmhn.cuda_gradient_and_score(theta, state_containers)
        self.assertEqual(round(score1, 8), round(score2, 8))
        np.testing.assert_array_equal(np.around(gradient1, decimals=8), np.around(gradient2, decimals=8))


if __name__ == '__main__':
    unittest.main()
